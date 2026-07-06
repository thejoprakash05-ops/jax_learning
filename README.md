# LLM Serving Portfolio Project

A self-contained, local demonstration of LLM serving concepts:

1. GPU accelerator optimization and profiling (Nsight, xprof)
2. LLM serving architectures (disaggregated serving, continuous batching)
3. XLA / compiler technologies for ML serving
4. JAX framework experience
5. On-prem cloud infrastructure for LLM deployment

Runs entirely on CPU with no Docker/WSL/GPU required — see `docs/` for how each piece maps
to real hardware and infrastructure.

Status: under construction. See `PROJECT_PLAN.md` for the build plan and progress.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Requirements → artifacts

_(filled in as the project progresses — see PROJECT_PLAN.md step 11)_
