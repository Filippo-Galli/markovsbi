import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import csv
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import optax

    from markovsbi.models.simple_scoremlp import build_score_mlp, precondition_functions
    from markovsbi.models.train_utils import build_loss_fn, build_masked_batch_sampler
    from markovsbi.models.utils import get_windows
    from markovsbi.sampling.kernels import DDIM, EulerMaruyama, PredictorCorrector
    from markovsbi.sampling.sample import Diffuser
    from markovsbi.sampling.score_fn import (
        CorrectedScoreFn,
        FNPEScoreFn,
        GaussCorrectedScoreFn,
        MaskedCorrectedScoreFn,
        ScoreFn,
        UncorrectedScoreFn,
    )
    from markovsbi.tasks.ar1 import AR1
    from markovsbi.utils.posterior_analysis import MMD, energy_distance
    from markovsbi.utils.sde_utils import init_sde

    return (
        AR1,
        Diffuser,
        MMD,
        MaskedCorrectedScoreFn,
        Path,
        PredictorCorrector,
        build_loss_fn,
        build_masked_batch_sampler,
        build_score_mlp,
        csv,
        energy_distance,
        init_sde,
        jax,
        jnp,
        mo,
        optax,
        plt,
        precondition_functions,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experiment settings
    """)
    return


@app.cell
def _(Path):
    # data parameters
    seed = 0
    timeseries_len = 101
    data_generated = 10_000_000

    # training parameters
    informed_proposal = False
    num_batches = 100_000
    batch_size = 1000

    # sampling parameters
    n_samples = 10_000
    denoising_timestep = 20
    num_traces = 100
    sampling_method = "gauss"
    mode = "masked"

    output_dir = Path(f"results/AR1")
    if informed_proposal:
        output_dir = output_dir.with_name(output_dir.name + "_informedProposal")

    output_dir = (
        output_dir / f"{mode}_{sampling_method}_nb{num_batches}__ts{timeseries_len}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    return (
        batch_size,
        csv_path,
        data_generated,
        denoising_timestep,
        informed_proposal,
        n_samples,
        num_batches,
        num_traces,
        output_dir,
        sampling_method,
        seed,
        timeseries_len,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initialization
    """)
    return


@app.cell
def seeds(jax, seed):
    key = jax.random.PRNGKey(seed)
    key_1, key_task = jax.random.split(key)
    key_2, key_init = jax.random.split(key_1)
    key_4 = jax.random.PRNGKey(2)
    return key, key_2, key_4, key_init, key_task


@app.cell
def simulator(AR1, timeseries_len):
    task = AR1(T_max=timeseries_len)
    prior = task.get_prior()
    simulator = task.get_simulator()
    return prior, simulator, task


@app.cell
def data_creation(data_generated, informed_proposal, key_task, task):
    all_data = task.get_data(
        key_task, data_generated, informed_proposal=informed_proposal
    )
    to_include = {"thetas", "xs"}
    data = {k: all_data[k] for k in to_include if k in all_data}
    return (data,)


@app.cell
def data_visualization(key_task, output_dir, plt, prior, simulator):
    # Plot an example AR(1) trajectory
    theta_sample = prior.sample(key_task)
    sample_traj = simulator(key_task, theta_sample)
    _fig, _ax = plt.subplots(figsize=(7, 2.5))
    _ax.plot(sample_traj, lw=1.2, label=f"AR(1) trajectory (a={theta_sample[0]:.2f})")
    _ax.set_xlabel("Time step t")
    _ax.set_ylabel("x_t")
    _ax.set_title(f"Simulated AR(1) Trajectory")
    _ax.legend()

    _plot_path = output_dir / "simulated_data.png"
    _fig.savefig(_plot_path, dpi=300, bbox_inches="tight")
    _fig
    return


@app.cell
def training_initialization(
    build_loss_fn,
    build_masked_batch_sampler,
    build_score_mlp,
    data,
    init_sde,
    jnp,
    key_init,
    num_batches,
    optax,
    precondition_functions,
):
    sde, weight_fn = init_sde(data)

    c_in, c_noise, c_out = precondition_functions(sde, shring_var=0.5)

    init_fn, score_net = build_score_mlp(
        window_size=2,
        hidden_dim=50,
        num_hidden=5,
        c_in=c_in,
        c_noise=c_noise,
        c_out=c_out,
        x_o_processing="linear",
    )

    batch_sampler = build_masked_batch_sampler(data, p_mask=0.5)

    loss_fn = build_loss_fn(
        "dsm",
        score_net,
        sde,
        weight_fn,
        control_variate=True,
        control_variate_optimal_scaling=True,
    )

    theta_batch, x_batch = batch_sampler(key_init, 50)
    d = theta_batch.shape[1]
    print(f"shape theta: {theta_batch.shape} x shape: {x_batch.shape}")

    params = init_fn(key_init, jnp.ones((50,)), theta_batch, x_batch)

    schedule = optax.cosine_onecycle_schedule(
        num_batches,
        5e-4,
    )
    optimizer = optax.chain(optax.adaptive_grad_clip(100.0), optax.adamw(schedule))
    opt_state = optimizer.init(params)
    return (
        batch_sampler,
        d,
        loss_fn,
        opt_state,
        optimizer,
        params,
        score_net,
        sde,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training
    """)
    return


@app.cell
def single_update(jax, loss_fn, optax, optimizer):
    @jax.jit
    def update(params, rng, opt_state, theta_batch, x_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, rng, theta_batch, x_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    return (update,)


@app.cell
def training(
    batch_sampler,
    batch_size,
    jax,
    key_2,
    num_batches,
    opt_state,
    params,
    update,
):
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Evaluation
    """)
    return


@app.cell
def test_x(key_4, output_dir, plt, prior, simulator, timeseries_len):
    theta_o = prior.sample(key_4)
    x_o = simulator(key_4, theta_o, timeseries_len)
    print(f"True theta_o: {theta_o}, len(x_0): {len(x_o)}")

    _fig, _ax = plt.subplots(figsize=(7, 2))
    _ax.plot(
        x_o,
        color="black",
        lw=1.2,
        label=f"Observation $x_o$ (True $a={theta_o[0]:.3f}$)",
    )
    _ax.set_xlabel("Time step $t$")
    _ax.set_ylabel("$x_t$")
    _ax.set_title("Observed Trajectory for Inference")
    _ax.legend()

    _plot_path = output_dir / "observed_data.png"
    _fig.savefig(_plot_path, dpi=300, bbox_inches="tight")
    _fig
    return theta_o, x_o


@app.cell
def sampling(
    Diffuser,
    MaskedCorrectedScoreFn,
    PredictorCorrector,
    d,
    denoising_timestep,
    jax,
    jnp,
    key_4,
    n_samples,
    params_1,
    prior,
    sampling_method,
    score_net,
    sde,
    x_o,
):
    score_fn = MaskedCorrectedScoreFn(
        score_net, params_1, sde, prior, method=sampling_method
    )

    kernel = PredictorCorrector(score_fn)
    time_grid = jnp.linspace(sde.T_min, sde.T_max, denoising_timestep)
    sampler = Diffuser(kernel, time_grid, (d,))

    samples = jax.vmap(sampler.sample, in_axes=(0, None))(
        jax.random.split(key_4, n_samples), x_o
    )
    return sampler, samples


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Diagnostic
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diffusion step check
    """)
    return


@app.cell
def _(jax, key_4, num_traces, output_dir, plt, sampler, theta_o, x_o):
    traces = jax.vmap(sampler.simulate, in_axes=(0, None))(
        jax.random.split(key_4, (num_traces,)), x_o
    )

    _fig, _ax = plt.subplots(figsize=(7, 3))
    for t in traces:
        _ax.plot(t[:, 0], alpha=0.15, color="tab:blue")
    _ax.axhline(
        theta_o[0],
        color="red",
        linestyle="--",
        lw=2,
        label=f"True $\\theta={theta_o[0]:.3f}$",
    )
    # _ax.set_ylim(-1.2, 1.2)
    _ax.set_xlabel("Diffusion step")
    _ax.set_ylabel(r"Parameter $\theta$")
    _ax.set_title("Reverse Diffusion Traces")
    _ax.legend()

    _plot_path = output_dir / "diffusion_step_trace.png"
    _fig.savefig(_plot_path, dpi=300, bbox_inches="tight")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Posterior plots
    """)
    return


@app.cell
def _(output_dir, plt, samples, theta_o):
    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.hist(
        samples[:, 0],
        bins=35,
        density=True,
        alpha=0.6,
        color="tab:blue",
        label="Diffuser Posterior",
    )
    _ax.axvline(
        theta_o[0],
        color="red",
        linestyle="--",
        lw=2,
        label=f"True $\\theta={theta_o[0]:.3f}$",
    )
    # _ax.set_xlim(-1.1, 1.1)
    _ax.set_xlabel(r"$\theta$ (AR(1) parameter $a$)")
    _ax.set_ylabel("Density")
    _ax.set_title("Estimated Posterior Distribution")
    _ax.legend()

    _plot_path = output_dir / "learned_posterior.png"
    _fig.savefig(_plot_path, dpi=300, bbox_inches="tight")
    _fig
    return


@app.cell
def _(key, n_samples, output_dir, plt, samples, task, theta_o, x_o):
    model = task.get_true_posterior(x_o)
    thetas = model.sample(key, (n_samples,))

    _fig, _ax = plt.subplots(figsize=(7, 3.5))
    _ax.hist(
        thetas[:, 0],
        bins=40,
        density=True,
        alpha=0.5,
        color="tab:green",
        label="Reference Posterior",
    )
    _ax.hist(
        samples[:, 0],
        bins=40,
        density=True,
        alpha=0.5,
        color="tab:blue",
        label="Diffuser Posterior",
    )
    _ax.axvline(
        theta_o[0],
        color="red",
        linestyle="--",
        lw=2,
        label=f"True $\\theta={theta_o[0]:.3f}$",
    )
    # _ax.set_xlim(-1.1, 1.1)
    _ax.set_xlabel(r"$\theta$ (AR(1) parameter $a$)")
    _ax.set_ylabel("Density")
    _ax.set_title("Posterior Comparison: Diffuser vs Reference")
    _ax.legend()

    _plot_path = output_dir / "posterior_comparison.png"
    _fig.savefig(_plot_path, dpi=300, bbox_inches="tight")
    _fig
    return (thetas,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Posterior metrics
    """)
    return


@app.cell
def _(csv, csv_path, samples, thetas):
    import numpy as np
    import torch
    from sbi.utils.metrics import c2st

    score = c2st(torch.tensor(np.array(samples)), torch.tensor(np.array(thetas)))
    print(f"C2ST score: {score.item():.4f} (ideal: 0.50)")

    score_value = float(score.item())

    with csv_path.open("w", newline="") as _f:
        _writer = csv.writer(_f)
        _writer.writerow(["metric", "value"])
        _writer.writerow(["c2st", score_value])
    return


@app.cell
def _(csv, csv_path, energy_distance, samples, thetas):
    ed = energy_distance(samples_p=samples, samples_q=thetas)
    print(f"Energy distance: {ed}")

    with csv_path.open("a", newline="") as _f:
        _writer = csv.writer(_f)
        _writer.writerow(["energy_distance", float(ed)])
    return


@app.cell
def _(MMD, csv, csv_path, jnp, samples, thetas):
    mmd2_mean, mmd2_std = MMD(samples, thetas)
    mmd = jnp.sqrt(jnp.maximum(mmd2_mean, 0.0))

    print(f"MMD: {float(mmd):.6f}")

    with csv_path.open("a", newline="") as _f:
        _writer = csv.writer(_f)
        _writer.writerow(["mmd", float(mmd)])
    return


if __name__ == "__main__":
    app.run()
