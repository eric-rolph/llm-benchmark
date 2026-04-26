"""
benchmark/backends/lm_studio.py — LM Studio backend.

Discovery: GET /v1/models  (OpenAI-compat list)
Loading:   POST /api/v0/models/load  (v0.3.x+)
Chat:      POST /v1/chat/completions
Thinking:  delta.reasoning_content  (Qwen3, DeepSeek-R1, Kimi K2, etc.)

Embedding models (nomic-embed, etc.) are automatically excluded from discovery:
they don't support /v1/chat/completions and LM Studio silently routes the request
to the already-loaded chat model, producing identical (and misleading) results.
"""
from __future__ import annotations

import re
import time

import requests

from benchmark.backends.base import Backend, ModelInfo

# Keyword patterns that identify embedding / encoder-only models by name.
# Used as a fallback when the LM Studio v0 API is unavailable.
_EMBED_RE = re.compile(r"embed(?:ding)?|encoder|e5-|bge-|gte-|minilm", re.IGNORECASE)


class LMStudioBackend(Backend):

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self._v1_url()}/models", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def _get_model_types(self) -> dict[str, str]:
        """Return {model_id: type} from LM Studio v0 API (0.3.x+). Empty dict on failure."""
        try:
            r = requests.get(f"{self._api_root()}/api/v0/models", timeout=3)
            if r.status_code == 200:
                return {
                    item.get("id", ""): item.get("type", "")
                    for item in r.json().get("data", [])
                    if item.get("id")
                }
        except Exception:
            pass
        return {}

    def discover_models(self) -> list[ModelInfo]:
        try:
            r = requests.get(f"{self._v1_url()}/models", timeout=5)
            r.raise_for_status()
            data = r.json().get("data", [])

            # Prefer the v0 API for authoritative model-type info; fall back to
            # keyword matching when the v0 endpoint is unavailable (older LM Studio).
            type_map = self._get_model_types()

            models = []
            skipped = []
            for item in data:
                mid = item.get("id", "")
                if not mid:
                    continue
                # "embeddings" is the type string LM Studio uses for embedding models.
                v0_type = type_map.get(mid, "").lower()
                if v0_type in ("embeddings", "embedding") or _EMBED_RE.search(mid):
                    skipped.append(mid)
                    continue
                models.append(ModelInfo(
                    id=mid,
                    name=mid.split("/")[-1] if "/" in mid else mid,
                    backend_name=self.name or "LM Studio",
                    details=item,
                ))
            if skipped:
                print(
                    f"  [LM Studio] Skipped {len(skipped)} embedding model(s) "
                    f"(not suitable for chat/completion benchmarking): "
                    + ", ".join(skipped)
                )
            return models
        except Exception:
            return []

    def ensure_model_loaded(self, model_id: str) -> None:
        """
        Request LM Studio to load the model via its management API (v0.3.x+).
        Falls back silently on older versions or when the model is already loaded.
        """
        api_root = self._api_root()
        try:
            resp = requests.post(
                f"{api_root}/api/v0/models/load",
                json={"identifier": model_id, "config": {"gpu_offload": "max"}},
                timeout=300,
            )
            if resp.status_code in (200, 201):
                print(f"  Loading {model_id} ", end="", flush=True)
                self._wait_for_model(model_id)
            elif resp.status_code == 400:
                pass  # Already loaded or ID mismatch — that's fine
            else:
                print(f"  (Auto-load HTTP {resp.status_code} — load manually in LM Studio)")
        except requests.exceptions.ConnectionError:
            print(f"  (Cannot reach {self.base_url}. Is LM Studio running?)")
        except Exception as e:
            print(f"  (Auto-load skipped: {e})")

    # ── private ──────────────────────────────────────────────────────────────

    def _wait_for_model(self, model_id: str, timeout: int = 120):
        client = self.get_openai_client()
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                ids = [m.id for m in client.models.list().data]
                if any(model_id in mid for mid in ids):
                    print(" ready.")
                    return
            except Exception:
                pass
            time.sleep(3)
            print(".", end="", flush=True)
        print(" timed out — proceeding anyway.")
