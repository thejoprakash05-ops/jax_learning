# Project Overview

Build a portfolio project demonstrating the skills listed in `Job requirements.md`:

1. GPU accelerator optimization and profiling (Nsight, xprof)
2. LLM serving architectures (disaggregated serving, continuous batching)
3. XLA or specialized compiler technologies for ML serving
4. JAX framework experience
5. On-prem cloud infrastructure for LLM deployment

Constraint from user: where real hardware (GPUs, on-prem cluster) isn't available, use a
simulator or emulator to stand in for it (e.g. multi-process/container "nodes" for an
on-prem cluster, CPU-based profiling substitutes if no NVIDIA GPU is present, etc.).

Goal: a working, demonstrable system — not just five disconnected demos — that a reviewer
can read/run to see each requirement addressed.

# Exploration Findings

Environment: Windows 11, empty project directory (only `Job requirements.md` present).

- **Python**: 3.14.5 native (`python`), pip 26.1.1. No `python3` alias.
- **GPU**: No NVIDIA GPU detected (`nvidia-smi` not found). No CUDA toolkit (`nvcc` missing).
  No Nsight Systems (`nsys`) or Nsight Compute (`ncu`) installed.
- **JAX / TensorFlow**: Not installed, but `jax` is pip-installable (CPU wheels available,
  confirmed via `pip download`). TensorFlow not installed (xprof ships with TF/JAX profiler
  tooling).
- **PyTorch**: Installed, CPU-only build (`torch 2.12.0+cpu`), `cuda.is_available() == False`.
- **Containers/orchestration**: No Docker, no WSL (`wsl --status` reports WSL not installed),
  no kubectl/kind/minikube. JAX does not officially support native Windows GPU backends and
  Docker Desktop typically requires WSL2 — so container-based "on-prem cluster" simulation
  needs either WSL2 install or a pure-Python process/thread-based stand-in.

Implications for the plan:
- No real GPU → GPU profiling (Nsight, xprof) must be demonstrated via JAX/XProf's
  CPU-compatible trace viewer and/or a documented "how this maps to real Nsight/xprof usage"
  writeup, not literal Nsight Systems captures.
- Disaggregated serving / on-prem cluster must be simulated with separate OS processes
  (e.g. multiprocessing or standalone Python services communicating over localhost
  sockets/HTTP) standing in for separate physical/on-prem nodes, unless the user opts to
  install WSL2 + Docker for a more realistic container-based simulation.
- JAX + `jax.jit`/XLA is installable and runnable on CPU now — good fit for demonstrating
  XLA compilation (HLO dump inspection) and JAX framework experience without any hardware
  dependency.

# Q&A / Requirements

- **On-prem/disaggregated cluster simulation**: Pure Python processes on localhost
  (HTTP/sockets between processes standing in for separate nodes). No Docker/WSL install
  required.
- **Model**: Small transformer built from scratch in JAX/Flax (attention, MLP, KV cache
  written by hand) — demonstrates real JAX + XLA understanding rather than just loading a
  checkpoint.
- **Serving interface**: HTTP API (FastAPI-style), separate prefill and decode
  processes/services, matching how real disaggregated serving systems are wired up.
- **Scope/depth**: Focused portfolio demo — each of the 5 job requirements gets a clear,
  working, documented demonstration. Days of effort, not weeks; breadth and clarity over
  production hardening.
- **Target audience / deadline**: Not answered by user (asked, no response before moving to
  planning). Assuming general portfolio piece (not tailored to one specific employer) and no
  hard deadline. Revisit if this assumption turns out wrong — it mainly affects tone of the
  README/writeup, not the technical plan.

# Plan Steps

- [x] **1. Scaffolding & environment setup**
  Created `src/`, `tests/`, `scripts/`, `configs/`, `docs/`, `artifacts/`, `checkpoints/`
  directories, `requirements.txt`, `.gitignore`, `git init`, README skeleton, and a `.venv`
  with dependencies installed.

  **Deviation from plan**: dropped `flax`. Installing it pulls in `orbax-checkpoint`, whose
  wheel contains a deeply-nested test-fixture path that exceeds the Windows `MAX_PATH` limit,
  so `pip install` fails on this machine (would require enabling Windows Long Path support
  system-wide, a registry change outside this project's scope — not done without asking).
  Verified `optax` does **not** pull in `orbax-checkpoint`, so it's kept. The transformer
  (Step 2) will be written in raw `jax`/`jax.numpy` (params as a plain pytree of arrays)
  instead of Flax modules — arguably a stronger demonstration of JAX fundamentals anyway.
  Checkpointing (Step 3) will use `numpy.savez` instead of orbax, avoiding the dependency
  entirely.

  Verified: `jax 0.10.2`, `jax.devices() == [CpuDevice(id=0)]`, a `jax.jit`-compiled function
  executes correctly, `optax`/`fastapi`/`uvicorn`/`httpx`/`pytest`/`matplotlib` all import.

- [ ] **2. From-scratch JAX transformer model**
  `src/model/transformer.py`: config, embeddings, multi-head attention with KV-cache support
  (separate prefill/full-sequence vs decode/single-step code paths), MLP, layernorm, output
  head, built in raw JAX (params as a pytree, no Flax — see Step 1 deviation note). Unit
  tests: shape checks, jit vs non-jit numerical equivalence.

- [ ] **3. Tiny training run for real (non-random) weights**
  `scripts/train.py`: char-level tiny public-domain corpus (e.g. tiny-Shakespeare), Optax
  Adam, short CPU training run, checkpoint saved to `checkpoints/` via `numpy.savez`. Gives
  the demo actual generated text instead of noise.

- [ ] **4. XLA compilation & inspection demo**
  `src/compiler_demo/xla_inspect.py`: `jax.jit`, `.lower(...).compile()`, dump HLO text,
  benchmark jit vs non-jit latency, save artifacts. `docs/xla_and_compilation.md` write-up
  mapping this to production XLA/TPU/GPU compiler usage.

- [ ] **5. Profiling demo (JAX profiler = xprof)**
  `src/profiling/run_profile.py`: wraps forward/generate calls with
  `jax.profiler.start_trace`, produces a trace viewable via
  `tensorboard --logdir` (the xprof plugin). `docs/profiling.md` explains what to look for
  and explicitly maps the workflow to Nsight Systems/Compute on real GPU hardware (since none
  is available here).

- [ ] **6. Inference engine core: prefill + decode with KV cache**
  `src/engine/kv_cache.py`, `src/engine/generate.py`: jitted `prefill(tokens) -> kv_cache,
  logits` and `decode_step(kv_cache, token) -> kv_cache, logits`.

- [ ] **7. Continuous batching scheduler**
  `src/engine/scheduler.py`: engine loop maintaining a pool of in-flight sequences; each
  iteration dynamically composes a batch from ready sequences, runs one decode step for the
  whole batch, admits/evicts sequences as they arrive/finish. Padding/masking strategy for
  JAX's static shapes. Unit tests vs a sequential (non-batched) baseline for correctness.

- [ ] **8. Disaggregated serving across simulated on-prem nodes**
  `src/serving/prefill_service.py`, `src/serving/decode_service.py` (runs the continuous
  batching loop), `src/serving/router.py` (gateway: calls prefill, load-balances to a decode
  worker, streams tokens back). `configs/cluster.yaml` describes the node inventory (prefill
  + N decode nodes) as a stand-in for on-prem infra-as-code. `scripts/launch_cluster.py`
  subprocess-launches all services from that config.

- [ ] **9. Client demo + metrics/observability**
  `src/client/demo_client.py` fires concurrent generation requests with varied arrival
  times and logs which node/batch handled each. `src/serving/metrics.py` exposes basic
  counters (throughput, batch size, latency) at `/metrics`. `scripts/plot_metrics.py`
  renders charts to `artifacts/metrics/` for the README.

- [ ] **10. End-to-end integration test**
  Launch the full simulated cluster, fire concurrent requests, assert correct responses and
  that batching actually occurred (observed batch size > 1).

- [ ] **11. Documentation tying requirements to artifacts**
  Top-level `README.md`: architecture overview, run instructions, and an explicit table
  mapping each of the 5 job requirements to where it's demonstrated in the repo plus how it
  would differ with real hardware/on-prem infra.

- [ ] **12. Final polish**
  Clean install from a fresh venv to confirm reproducibility; final review pass; last commit.

**Confirmed on 2026-07-05.**

# Testing

_(pending)_

# Commits

_(pending)_
