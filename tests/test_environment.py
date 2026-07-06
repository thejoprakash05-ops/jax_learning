import jax
import jax.numpy as jnp
import optax


def test_jax_has_a_cpu_device():
    devices = jax.devices()
    assert len(devices) >= 1
    assert devices[0].platform == "cpu"


def test_jit_compiles_and_executes_correctly():
    f = jax.jit(lambda x: jnp.sum(x**2))
    result = f(jnp.arange(5.0))
    assert float(result) == 30.0


def test_optax_optimizer_can_be_constructed_and_used():
    optimizer = optax.adam(1e-3)
    params = {"w": jnp.ones((3,))}
    opt_state = optimizer.init(params)
    grads = {"w": jnp.ones((3,))}
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    assert new_params["w"].shape == (3,)
    assert not jnp.allclose(new_params["w"], params["w"])


def test_web_and_test_stack_importable():
    import fastapi
    import uvicorn
    import httpx
    import matplotlib

    assert fastapi.__version__
    assert uvicorn.__version__
