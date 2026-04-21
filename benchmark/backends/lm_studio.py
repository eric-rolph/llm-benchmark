"""
benchmark/backends/lm_studio.py — LM Studio backend.

Discovery: GET /v1/models  (OpenAI-compat list)
Loading:   POST /api/v0/models/load  (v0.3.x+)
Chat:      POST /v1/chat/completions
Thinking:  delta.reasoning_content  (Qwen3, DeepSeek-R1, Kimi K2, etc.)
"""
from __future__ import annotations

import time

import requests

from benchmark.backends.base import Backend, ModelInfo


class LMStudioBackend(Backend):

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self._v1_url()}/models", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def discover_models(self) -> list[ModelInfo]:
        try:
            r = requests.get(f"{self._v1_url()}/models", timeout=5)
            r.raise_for_status()
            data = r.json().get("data", [])
            models = []
            for item in data:
                mid = item.get("id", "")
                if not mid:
                    continue
                models.append(ModelInfo(
                    id=mid,
                    name=mid.split("/")[-1] if "/" in mid else mid,
                    backend_name=self.name or "LM Studio",
                    details=item,
                ))
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
