import csv
import pickle
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optax
import torch
from sbi.utils.metrics import c2st

from markovsbi.models.simple_scoremlp import (
    build_score_mlp,
    precondition_functions,
)
from markovsbi.models.train_utils import (
    build_batch_sampler,
    build_loss_fn,
    build_masked_batch_sampler,
)
from markovsbi.sampling.kernels import PredictorCorrector
from markovsbi.sampling.sample import Diffuser
from markovsbi.sampling.score_fn import (
    CorrectedScoreFn,
    MaskedCorrectedScoreFn,
)
from markovsbi.tasks.ar1 import AR1
from markovsbi.utils.posterior_analysis import MMD, energy_distance
from markovsbi.utils.sde_utils import init_sde


@dataclass
class ExperimentSetup:
    # Data parameters
    seed_training: int
    num_test_observations: int
    timeseries_len: int
    data_generated: int

    # Training parameters
    informed_proposal: bool
    num_batches: int
    batch_size: int
    mode: str

    # Sampling parameters
    n_samples: int
    denoising_timestep: int
    num_traces: int
    sampling_method: str

    # Output paths
    training_dir: Path
    output_dir: Path
    metrics_file_path: Path


@dataclass
class TrainedModel:
    task: Any
    prior: Any
    simulator: Callable
    sde: Any
    score_net: Callable
    params: Any
    theta_dim: int


def make_settings(**kwargs) -> ExperimentSetup:
    defaults = {
        # Data parameters
        "seed_training": 0,
        "num_test_observations": 3,
        "timeseries_len": 20,
        "data_generated": 1_000_000,
        # Training parameters
        "informed_proposal": False,
        "num_batches": 10_000,
        "batch_size": 1000,
        "mode": "masked",
        # Sampling parameters
        "n_samples": 5_000,
        "denoising_timestep": 20,
        "num_traces": 200,
        "sampling_method": "masked",
        # Output paths
        "training_dir": Path("results/AR1"),
        "output_dir": Path("results/AR1"),
        "metrics_file_path": Path("results/AR1/metrics.csv"),
    }

    defaults.update(kwargs)
    return ExperimentSetup(**defaults)


def construct_output_dirs(settings: ExperimentSetup) -> None:
    """
    Construct separate directories for training and evaluation.

    The training directory does not contain the sampling method because
    sampling_method does not affect training. Each sampling method gets
    a separate evaluation directory.
    """
    experiment_name = "AR1"

    if settings.informed_proposal:
        experiment_name += "_informedProposal"

    base_dir = Path("results") / experiment_name

    training_name = (
        f"{settings.mode}"
        f"_nb{settings.num_batches}"
        f"_ts{settings.timeseries_len}"
        f"_nd{settings.data_generated}"
        f"_bs{settings.batch_size}"
        f"_seed{settings.seed_training}"
    )

    settings.training_dir = base_dir / training_name

    evaluation_name = (
        f"{settings.sampling_method}"
        f"_ns{settings.n_samples}"
        f"_dt{settings.denoising_timestep}"
    )

    settings.output_dir = settings.training_dir / evaluation_name
    settings.metrics_file_path = settings.output_dir / "metrics.csv"

    settings.training_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)


def key_generator(settings: ExperimentSetup):
    """
    Generate independent deterministic keys from seed_training.

    No separate testing seed is necessary. The testing key is one branch
    of the PRNG tree rooted at seed_training.
    """
    root_key = jax.random.PRNGKey(settings.seed_training)

    (
        key_task,
        key_batch_init,
        key_model_init,
        key_train,
        key_test,
    ) = jax.random.split(root_key, 5)

    return (
        key_task,
        key_batch_init,
        key_model_init,
        key_train,
        key_test,
    )


def data_exploration(
    output_dir: Path,
    trajectory: jax.Array,
    filename: str,
    title: str,
) -> None:
    """Plot one AR(1) trajectory."""
    fig, ax = plt.subplots(figsize=(7, 2.5))

    ax.plot(np.asarray(trajectory), lw=1.2)
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"$x_t$")
    ax.set_title(title)

    plot_path = output_dir / filename
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def posterior_plot(
    output_dir: Path,
    samples: jax.Array,
    theta_o: jax.Array,
    gt_samples: jax.Array,
    sampling_method: str,
    filename: str,
) -> None:
    """Plot the learned posterior against the true posterior."""
    samples_np = np.asarray(samples)
    gt_samples_np = np.asarray(gt_samples)
    theta_value = float(np.asarray(theta_o)[0])

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.hist(
        samples_np[:, 0],
        bins=35,
        density=True,
        alpha=0.6,
        color="tab:blue",
        label=f"Estimated posterior ({sampling_method})",
    )

    ax.hist(
        gt_samples_np[:, 0],
        bins=35,
        density=True,
        alpha=0.5,
        color="tab:red",
        label="True posterior",
    )

    ax.axvline(
        theta_value,
        color="black",
        linestyle="--",
        lw=2,
        label=rf"True $\theta={theta_value:.3f}$",
    )

    ax.set_xlabel(r"$\theta$ (AR(1) parameter)")
    ax.set_ylabel("Density")
    ax.set_title("Estimated Posterior Distribution")
    ax.legend()

    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def tracing_plot(
    key: jax.Array,
    sampler: Diffuser,
    theta_o: jax.Array,
    x_o: jax.Array,
    settings: ExperimentSetup,
    test_index: int,
) -> None:
    """Plot reverse-diffusion trajectories."""
    trace_keys = jax.random.split(key, settings.num_traces)

    traces = jax.vmap(
        sampler.simulate,
        in_axes=(0, None),
    )(trace_keys, x_o)

    theta_value = float(np.asarray(theta_o)[0])

    fig, ax = plt.subplots(figsize=(7, 3))

    for trace in np.asarray(traces):
        ax.plot(
            trace[:, 0],
            alpha=0.15,
            color="tab:blue",
        )

    ax.axhline(
        theta_value,
        color="red",
        linestyle="--",
        lw=2,
        label=rf"True $\theta={theta_value:.3f}$",
    )

    ax.set_xlabel("Diffusion step")
    ax.set_ylabel(r"Parameter $\theta$")
    ax.set_title("Reverse Diffusion Traces")
    ax.legend()

    filename = f"diffusion_trace_test_{test_index}.png"
    fig.savefig(
        settings.output_dir / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_update_fn(
    optimizer: optax.GradientTransformation,
    loss_fn: Callable,
) -> Callable:
    """
    Build a jitted update function.

    optimizer and loss_fn are captured in the closure instead of being
    passed as dynamic arguments to jax.jit.
    """

    @jax.jit
    def update(
        params,
        rng,
        opt_state,
        theta_batch,
        x_batch,
    ):
        loss, grads = jax.value_and_grad(loss_fn)(
            params,
            rng,
            theta_batch,
            x_batch,
        )

        updates, new_opt_state = optimizer.update(
            grads,
            opt_state,
            params=params,
        )

        new_params = optax.apply_updates(params, updates)

        return loss, new_params, new_opt_state

    return update


def build_model_components(
    settings: ExperimentSetup,
):
    """
    Generate the training data and construct the score network.

    These objects must be reconstructed even when cached parameters are
    loaded because score_net and sde are needed during evaluation.
    """
    (
        key_task,
        key_batch_init,
        key_model_init,
        key_train,
        key_test,
    ) = key_generator(settings)

    task = AR1(T_max=settings.timeseries_len)
    prior = task.get_prior()
    simulator = task.get_simulator()

    all_data = task.get_data(
        key_task,
        settings.data_generated,
        2,
        informed_proposal=settings.informed_proposal,
    )

    data = {name: all_data[name] for name in ("thetas", "xs") if name in all_data}

    sample_traj = simulator(
        key_task,
        prior.sample(key_task),
    )

    data_exploration(
        output_dir=settings.training_dir,
        trajectory=sample_traj,
        filename="training_trajectory.png",
        title="Example Training Trajectory",
    )

    sde, weight_fn = init_sde(data)

    c_in, c_noise, c_out = precondition_functions(
        sde,
        shring_var=0.5,
    )

    init_fn, score_net = build_score_mlp(
        window_size=2,
        hidden_dim=50,
        num_hidden=5,
        c_in=c_in,
        c_noise=c_noise,
        c_out=c_out,
        x_o_processing="linear",
    )

    # Masked mode should use masked batches.
    if settings.mode == "masked":
        batch_sampler = build_masked_batch_sampler(
            data,
            p_mask=0.5,
        )
    else:
        batch_sampler = build_batch_sampler(data)

    theta_example, x_example = batch_sampler(
        key_batch_init,
        50,
    )

    theta_dim = theta_example.shape[-1]

    return {
        "task": task,
        "prior": prior,
        "simulator": simulator,
        "sde": sde,
        "weight_fn": weight_fn,
        "init_fn": init_fn,
        "score_net": score_net,
        "batch_sampler": batch_sampler,
        "theta_example": theta_example,
        "x_example": x_example,
        "theta_dim": theta_dim,
        "key_model_init": key_model_init,
        "key_train": key_train,
        "key_test": key_test,
    }


def train_model(settings: ExperimentSetup) -> TrainedModel:
    """
    Train the network once or load cached parameters.

    The checkpoint depends only on training parameters and does not depend
    on sampling_method or num_test_observations.
    """
    if settings.informed_proposal and settings.mode != "masked":
        raise ValueError("informed_proposal=True is only supported with mode='masked'.")

    construct_output_dirs(settings)

    components = build_model_components(settings)
    checkpoint_path = settings.training_dir / "params.pkl"

    if checkpoint_path.exists():
        print(f"Loading cached parameters: {checkpoint_path}")

        with checkpoint_path.open("rb") as file:
            trained_params = pickle.load(file)

        trained_params = jax.tree_util.tree_map(
            jnp.asarray,
            trained_params,
        )

        return TrainedModel(
            task=components["task"],
            prior=components["prior"],
            simulator=components["simulator"],
            sde=components["sde"],
            score_net=components["score_net"],
            params=trained_params,
            theta_dim=components["theta_dim"],
        )

    print(f"Training model: {settings.training_dir}")

    params = components["init_fn"](
        components["key_model_init"],
        jnp.ones((components["theta_example"].shape[0],)),
        components["theta_example"],
        components["x_example"],
    )

    loss_fn = build_loss_fn(
        "dsm",
        components["score_net"],
        components["sde"],
        components["weight_fn"],
        control_variate=True,
        control_variate_optimal_scaling=True,
    )

    schedule = optax.cosine_onecycle_schedule(
        transition_steps=settings.num_batches,
        peak_value=5e-4,
    )

    optimizer = optax.chain(
        optax.adaptive_grad_clip(100.0),
        optax.adamw(schedule),
    )

    opt_state = optimizer.init(params)
    update_fn = make_update_fn(optimizer, loss_fn)

    current_key = components["key_train"]
    current_params = params
    current_opt_state = opt_state

    for step in range(settings.num_batches):
        current_key, key_loss, key_batch = jax.random.split(
            current_key,
            3,
        )

        theta_batch, x_batch = components["batch_sampler"](
            key_batch,
            settings.batch_size,
        )

        (
            loss,
            current_params,
            current_opt_state,
        ) = update_fn(
            current_params,
            key_loss,
            current_opt_state,
            theta_batch,
            x_batch,
        )

        if step % 100 == 0 or step == settings.num_batches - 1:
            print(
                f"Step {step + 1}/{settings.num_batches}, loss={float(loss):.4f}",
                end="\r",
            )

    print()

    trained_params = jax.device_get(current_params)

    temporary_path = checkpoint_path.with_suffix(".tmp")

    with temporary_path.open("wb") as file:
        pickle.dump(trained_params, file)

    temporary_path.replace(checkpoint_path)

    print(f"Saved parameters: {checkpoint_path}")

    return TrainedModel(
        task=components["task"],
        prior=components["prior"],
        simulator=components["simulator"],
        sde=components["sde"],
        score_net=components["score_net"],
        params=current_params,
        theta_dim=components["theta_dim"],
    )


def build_sampler(
    settings: ExperimentSetup,
    trained_model: TrainedModel,
) -> Diffuser:
    """Construct the posterior sampler for one sampling method."""
    if settings.mode == "masked":
        score_fn = MaskedCorrectedScoreFn(
            trained_model.score_net,
            trained_model.params,
            trained_model.sde,
            trained_model.prior,
            method=settings.sampling_method,
        )
    else:
        score_fn = CorrectedScoreFn(
            trained_model.score_net,
            trained_model.params,
            trained_model.sde,
            trained_model.prior,
        )

    kernel = PredictorCorrector(score_fn)

    time_grid = jnp.linspace(
        trained_model.sde.T_min,
        trained_model.sde.T_max,
        settings.denoising_timestep,
    )

    return Diffuser(
        kernel,
        time_grid,
        (trained_model.theta_dim,),
    )


def calculate_metrics(
    samples: jax.Array,
    gt_samples: jax.Array,
):
    """Calculate C2ST, MMD, and energy distance."""
    samples_np = np.array(samples, copy=True)
    gt_samples_np = np.array(gt_samples, copy=True)

    c2st_score = c2st(
        torch.from_numpy(samples_np),
        torch.from_numpy(gt_samples_np),
        classifier_kwargs={"n_jobs": -1},  # use all CPU cores
    )

    ed = energy_distance(
        samples_p=samples,
        samples_q=gt_samples,
    )

    subsample_size = min(5000, samples.shape[0])
    mmd2_mean, _ = MMD(samples, gt_samples, subsample_size=subsample_size)
    mmd = jnp.sqrt(jnp.maximum(mmd2_mean, 0.0))

    return (
        float(c2st_score.item()),
        float(mmd),
        float(ed),
    )


def run_evaluation(
    settings: ExperimentSetup,
    trained_model: TrainedModel,
) -> None:
    """
    Evaluate one trained network on multiple test observations.

    Each observation receives independent keys for:
    - true parameter generation
    - trajectory simulation
    - learned posterior sampling
    - true posterior sampling
    - diffusion trace generation
    """
    construct_output_dirs(settings)

    if settings.metrics_file_path.exists():
        print(f"Skipping completed evaluation: {settings.metrics_file_path}")
        return

    sampler = build_sampler(settings, trained_model)

    *_, key_test = key_generator(settings)

    all_metrics = []

    for test_index in range(settings.num_test_observations):
        # fold_in means test i remains unchanged if the total number of
        # test observations is later increased.
        observation_key = jax.random.fold_in(
            key_test,
            test_index,
        )

        (
            key_theta,
            key_simulator,
            key_posterior,
            key_true_posterior,
            key_trace,
        ) = jax.random.split(observation_key, 5)

        print(f"Test observation {test_index}")
        print("-" * 80)

        theta_o = trained_model.prior.sample(key_theta)

        x_o = trained_model.simulator(
            key_simulator,
            theta_o,
        )

        data_exploration(
            output_dir=settings.output_dir,
            trajectory=x_o,
            filename=f"test_trajectory_{test_index}.png",
            title=f"Test Trajectory {test_index}",
        )

        posterior_keys = jax.random.split(
            key_posterior,
            settings.n_samples,
        )

        samples = jax.vmap(
            sampler.sample,
            in_axes=(0, None),
        )(posterior_keys, x_o).block_until_ready()

        print(f"\tSampled {settings.n_samples} posterior samples")

        true_posterior = trained_model.task.get_true_posterior(x_o)

        gt_samples = true_posterior.sample(
            key_true_posterior,
            (settings.n_samples,),
        )

        print(f"\tSampled {settings.n_samples} ground truth samples")

        print(f"\tCalculating metrics...")
        c2st_value, mmd_value, ed_value = calculate_metrics(
            samples,
            gt_samples,
        )

        theta_value = float(np.asarray(theta_o)[0])

        all_metrics.append(
            [
                test_index,
                theta_value,
                c2st_value,
                mmd_value,
                ed_value,
            ]
        )

        posterior_plot(
            output_dir=settings.output_dir,
            samples=samples,
            theta_o=theta_o,
            gt_samples=gt_samples,
            sampling_method=settings.sampling_method,
            filename=f"posterior_test_{test_index}.png",
        )

        tracing_plot(
            key=key_trace,
            sampler=sampler,
            theta_o=theta_o,
            x_o=x_o,
            settings=settings,
            test_index=test_index,
        )

        print(
            f"\tTest {test_index}: "
            f"theta={theta_value:.4f}, "
            f"C2ST={c2st_value:.4f}, "
            f"MMD={mmd_value:.4f}, "
            f"ED={ed_value:.4f}"
        )

    with settings.metrics_file_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "test_index",
                "theta_true",
                "c2st",
                "mmd",
                "energy_distance",
            ]
        )

        writer.writerows(all_metrics)

    print(f"Saved metrics: {settings.metrics_file_path}")


def main():
    # Only parameters that affect training belong in this grid.
    training_grid = {
        "seed_training": [0],
        "timeseries_len": [10],
        "informed_proposal": [False, True],
        "num_batches": [2_000],
        "mode": ["masked", "gloeckler"],
    }

    # These methods share the same trained network.
    sampling_methods = [
        "jac",
    ]

    training_keys = list(training_grid)

    for values in product(*(training_grid[key] for key in training_keys)):
        training_config = dict(zip(training_keys, values))

        base_settings = make_settings(
            **training_config,
            num_test_observations=3,
            n_samples=5_000,
            denoising_timestep=20,
            num_traces=200,
            sampling_method=sampling_methods[0],
        )

        if base_settings.informed_proposal and base_settings.mode != "masked":
            continue

        evaluation_settings = []

        for sampling_method in sampling_methods:
            method_settings = replace(
                base_settings,
                sampling_method=sampling_method,
            )

            construct_output_dirs(method_settings)
            evaluation_settings.append(method_settings)

        # Avoid even loading/training if every evaluation already exists.
        if all(settings.metrics_file_path.exists() for settings in evaluation_settings):
            print(
                "Skipping completed training configuration: "
                f"{base_settings.seed_training=}, "
                f"{base_settings.informed_proposal=}"
            )
            continue

        # Called exactly once for this training configuration.
        trained_model = train_model(base_settings)

        # Evaluate the same trained model with every sampling method.
        for method_settings in evaluation_settings:
            run_evaluation(
                method_settings,
                trained_model,
            )


if __name__ == "__main__":
    main()
