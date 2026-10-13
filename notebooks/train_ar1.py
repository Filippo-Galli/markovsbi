import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import optax

    from markovsbi.models.simple_scoremlp import build_score_mlp, precondition_functions
    from markovsbi.models.train_utils import build_batch_sampler, build_loss_fn
    from markovsbi.tasks.ar1 import AR1
    from markovsbi.utils.sde_utils import init_sde

    return (
        AR1,
        build_batch_sampler,
        build_loss_fn,
        build_score_mlp,
        init_sde,
        jax,
        jnp,
        optax,
        plt,
        precondition_functions,
    )


@app.cell
def _(jax):
    key = jax.random.PRNGKey(0)
    return (key,)


@app.cell
def _(AR1):
    task = AR1()
    prior = task.get_prior()
    simulator = task.get_simulator()
    return prior, simulator, task


@app.cell
def _(jax, key):
    key_1, key_task = jax.random.split(key)
    return key_1, key_task


@app.cell
def _(key_task, task):
    all_data = task.get_data(key_task, 100000, 2)
    return (all_data,)


@app.cell
def _(key_task, plt, prior, simulator):
    # Plot an example AR(1) trajectory
    theta_sample = prior.sample(key_task)
    sample_traj = simulator(key_task, theta_sample, 10)
    _fig, _ax= plt.subplots(figsize=(7, 2.5))
    _ax.plot(sample_traj, lw=1.2, label=f"AR(1) trajectory (a={theta_sample[0]:.2f})")
    _ax.set_xlabel("Time step t")
    _ax.set_ylabel("x_t")
    _ax.set_title("Simulated AR(1) Trajectory")
    _ax.legend()
    return


@app.cell
def _(all_data, init_sde):
    to_include = {"thetas", "xs"}
    data = {k: all_data[k] for k in to_include if k in all_data}
    sde, weight_fn = init_sde(data)
    return data, sde, weight_fn


@app.cell
def _(jax, key_1):
    key_2, key_init = jax.random.split(key_1)
    return key_2, key_init


@app.cell
def _(
    build_batch_sampler,
    build_loss_fn,
    build_score_mlp,
    data,
    precondition_functions,
    sde,
    weight_fn,
):
    c_in, c_noise, c_out = precondition_functions(sde, shring_var=0.5)
    init_fn, score_net = build_score_mlp(
        2,
        hidden_dim=50,
        num_hidden=5,
        c_in=c_in,
        c_noise=c_noise,
        c_out=c_out,
        x_o_processing="linear",
    )
    batch_sampler = build_batch_sampler(data)
    loss_fn = build_loss_fn(
        "dsm",
        score_net,
        sde,
        weight_fn,
        control_variate=True,
        control_variate_optimal_scaling=True,
    )
    return batch_sampler, init_fn, loss_fn, score_net


@app.cell
def _(batch_sampler, key_init):
    theta_batch, x_batch = batch_sampler(key_init, 50)
    d = theta_batch.shape[1]
    print(theta_batch.shape, x_batch.shape)
    return d, theta_batch, x_batch


@app.cell
def _(x_batch):
    x_batch.shape
    return


@app.cell
def _(init_fn, jnp, key_init, theta_batch, x_batch):
    params = init_fn(key_init, jnp.ones((50,)), theta_batch, x_batch)
    return (params,)


@app.cell
def _(optax, params):
    num_batches = 5000
    schedule = optax.cosine_onecycle_schedule(
        num_batches,
        5e-4,
    )
    optimizer = optax.chain(optax.adaptive_grad_clip(100.0), optax.adamw(schedule))
    opt_state = optimizer.init(params)
    return num_batches, opt_state, optimizer


@app.cell
def _(jax, loss_fn, optax, optimizer):
    @jax.jit
    def update(params, rng, opt_state, theta_batch, x_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, rng, theta_batch, x_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    return (update,)


@app.cell
def _(batch_sampler, jax, key_2, num_batches, opt_state, params, update):
    batch_size = 1000
    curr_key = key_2
    curr_params = params
    curr_opt_state = opt_state

    for i in range(num_batches):
        curr_key, key_loss, key_batch = jax.random.split(curr_key, 3)
        theta_batch_1, x_batch_1 = batch_sampler(key_batch, batch_size)
        loss, curr_params, curr_opt_state = update(
            curr_params, key_loss, curr_opt_state, theta_batch_1, x_batch_1
        )
        if i % 100 == 0 or i == num_batches - 1:
            print(f"Loss: {loss:.4f} round: {i}         ", end="\r")
    params_1 = curr_params
    return (params_1,)


@app.cell
def _():
    from markovsbi.sampling.score_fn import (
        CorrectedScoreFn,
        FNPEScoreFn,
        GaussCorrectedScoreFn,
        ScoreFn,
        UncorrectedScoreFn,
    )

    return (CorrectedScoreFn,)


@app.cell
def _(jax, prior, simulator):
    key_4 = jax.random.PRNGKey(2)
    theta_o = prior.sample(key_4)
    num_obs = 11
    x_o = simulator(key_4, theta_o, num_obs)
    print("True theta_o:", theta_o)
    return key_4, theta_o, x_o


@app.cell
def _(plt, theta_o, x_o):
    _fig, _ax= plt.subplots(figsize=(7, 2))
    _ax.plot(x_o, color="black", lw=1.2, label=f"Observation $x_o$ (True $a={theta_o[0]:.3f}$)")
    _ax.set_xlabel("Time step $t$")
    _ax.set_ylabel("$x_t$")
    _ax.set_title("Observed Trajectory for Inference")
    _ax.legend()
    return


@app.cell
def _(CorrectedScoreFn, params_1, prior, score_net, sde):
    # score_fn = GaussCorrectedScoreFn(score_net, params_1, sde, prior, posterior_precission_est_fn=None)
    # score_fn.estimate_hyperparameters(x_o, (1,), key_4, diag=True, precission_nugget=0.1)
    score_fn = CorrectedScoreFn(score_net, params_1, sde, prior)
    return (score_fn,)


@app.cell
def _(d, jnp, score_fn, sde):
    from markovsbi.models.utils import get_windows
    from markovsbi.sampling.kernels import DDIM, EulerMaruyama, PredictorCorrector
    from markovsbi.sampling.sample import Diffuser

    kernel = PredictorCorrector(score_fn)
    length = 500
    time_grid = jnp.linspace(sde.T_min, sde.T_max, length)
    sampler = Diffuser(kernel, time_grid, (d,))
    return (sampler,)


@app.cell
def _(jax, key_4, sampler, x_o):
    traces = jax.vmap(sampler.simulate, in_axes=(0, None))(
        jax.random.split(key_4, (1000,)), x_o
    )
    return (traces,)


@app.cell
def _(plt, theta_o, traces):
    _fig, _ax= plt.subplots(figsize=(7, 3))
    for t in traces:
        _ax.plot(t[:, 0], alpha=0.15, color="tab:blue")
    _ax.axhline(theta_o[0], color="red", linestyle="--", lw=2, label=f"True $\\theta={theta_o[0]:.3f}$")
    #_ax.set_ylim(-1.2, 1.2)
    _ax.set_xlabel("Diffusion step")
    _ax.set_ylabel(r"Parameter $\theta$")
    _ax.set_title("Reverse Diffusion Traces")
    _ax.legend()
    return


@app.cell
def _(jax, jnp, key_4, sampler, x_o):
    samples_dim = 1000
    samples = jax.vmap(sampler.sample, in_axes=(0, None))(
        jax.random.split(key_4, samples_dim), x_o
    )
    print(f"Number of NaN: {jnp.isnan(samples).sum()}")
    return samples, samples_dim


@app.cell
def _(plt, samples, theta_o):
    _fig, _ax= plt.subplots(figsize=(6, 3))
    _ax.hist(samples[:, 0], bins=35, density=True, alpha=0.6, color="tab:blue", label="Diffuser Posterior")
    _ax.axvline(theta_o[0], color="red", linestyle="--", lw=2, label=f"True $\\theta={theta_o[0]:.3f}$")
    #_ax.set_xlim(-1.1, 1.1)
    _ax.set_xlabel(r"$\theta$ (AR(1) parameter $a$)")
    _ax.set_ylabel("Density")
    _ax.set_title("Estimated Posterior Distribution")
    _ax.legend()
    return


@app.cell
def _(task, x_o):
    model = task.get_true_posterior(x_o)
    return (model,)


@app.cell
def _(jax, model, samples_dim):
    thetas = model.sample(jax.random.PRNGKey(0), (samples_dim,))
    return (thetas,)


@app.cell
def _(plt, samples, theta_o, thetas):
    _fig, _ax= plt.subplots(figsize=(7, 3.5))
    _ax.hist(thetas[:, 0], bins=40, density=True, alpha=0.5, color="tab:green", label="Reference Posterior")
    _ax.hist(samples[:, 0], bins=40, density=True, alpha=0.5, color="tab:blue", label="Diffuser Posterior")
    _ax.axvline(theta_o[0], color="red", linestyle="--", lw=2, label=f"True $\\theta={theta_o[0]:.3f}$")
    #_ax.set_xlim(-1.1, 1.1)
    _ax.set_xlabel(r"$\theta$ (AR(1) parameter $a$)")
    _ax.set_ylabel("Density")
    _ax.set_title("Posterior Comparison: Diffuser vs Reference")
    _ax.legend()
    return


@app.cell
def _(samples, thetas):
    import numpy as np
    import torch
    from sbi.utils.metrics import c2st

    score = c2st(torch.tensor(np.array(samples)), torch.tensor(np.array(thetas)))
    print(f"C2ST score: {score.item():.4f} (ideal: 0.50)")
    return


if __name__ == "__main__":
    app.run()
