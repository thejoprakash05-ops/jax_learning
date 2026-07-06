JAX is a numerical computing library from Google (now Google DeepMind) for high-performance machine learning research. Key features:

NumPy-compatible API — most NumPy code works with minimal changes (jax.numpy mirrors numpy)
Automatic differentiation (grad) — computes gradients of arbitrary Python/NumPy functions, including higher-order derivatives
JIT compilation (jit) — compiles functions via XLA to run fast on CPU/GPU/TPU
Vectorization (vmap) — automatically batches a function without manually rewriting it for batched inputs
Parallelization (pmap/pjit) — distributes computation across multiple devices/hosts
It's functional in style (pure functions, immutable arrays, explicit PRNG state) rather than object-oriented like PyTorch. It's widely used in research (e.g., much of DeepMind's work) and underlies libraries like Flax and Haiku (neural network layers) and Optax (optimizers).

In the context of LLM serving, JAX shows up mainly on the training/research side and in some inference stacks (e.g., Google's PaLM/Gemini serving, or projects like MaxText) that want TPU support and functional transformations like pmap for model/data parallelism.