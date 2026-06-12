"""
run.py — back-compat shim. The real entry point lives in benchmark.cli
(installed as the `llm-bench` console script); this keeps
`python run.py …` working from a repo checkout.
"""
from benchmark.cli import main

if __name__ == "__main__":
    main()
