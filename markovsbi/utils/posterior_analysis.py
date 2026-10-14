### Optimized JAX Implementation
from functools import partial
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp


# ==============================================================================
# Helper: Fast Pairwise Squared Euclidean Distances
# ==============================================================================
@jax.jit
def _pairwise_sq_dists(A: jax.Array, B: jax.Array) -> jax.Array:
    """Computes pairwise squared Euclidean distances: ||a_i - b_j||^2."""
    a_sq = jnp.sum(A**2, axis=-1, keepdims=True)  # (N, 1)
    b_sq = jnp.sum(B**2, axis=-1, keepdims=True)  # (M, 1)
    dists_sq = a_sq + b_sq.T - 2.0 * jnp.dot(A, B.T)
    return jnp.maximum(dists_sq, 0.0)


@partial(jax.jit, static_argnames=("is_same", "unbiased"))
def _mean_dist_full(
    A: jax.Array, B: jax.Array, is_same: bool = False, unbiased: bool = False
) -> jax.Array:
    dists = jnp.sqrt(_pairwise_sq_dists(A, B))
    if is_same and unbiased:
        n = A.shape[0]
        # Diagonal is 0, so sum(dists) equals off-diagonal sum
        return jnp.sum(dists) / (n * (n - 1))
    return jnp.mean(dists)


def _mean_dist_chunked(
    A: jax.Array,
    B: jax.Array,
    chunk_size: int,
    is_same: bool = False,
    unbiased: bool = False,
) -> jax.Array:
    """Memory-safe chunked pairwise distance using jax.lax.scan (O(chunk_size * N) memory)."""
    na = A.shape[0]
    n_chunks = (na + chunk_size - 1) // chunk_size
    pad_len = n_chunks * chunk_size - na
    A_pad = jnp.pad(A, ((0, pad_len), (0, 0))) if pad_len > 0 else A
    A_chunks = A_pad.reshape(n_chunks, chunk_size, -1)
    b_sq = jnp.sum(B**2, axis=-1, keepdims=True)

    def scan_step(acc, a_chunk):
        a_sq = jnp.sum(a_chunk**2, axis=-1, keepdims=True)
        dist_sq = jnp.maximum(a_sq + b_sq.T - 2.0 * jnp.dot(a_chunk, B.T), 0.0)
        return acc + jnp.sum(jnp.sqrt(dist_sq)), None

    total_sum, _ = jax.lax.scan(scan_step, 0.0, A_chunks)

    if pad_len > 0:
        # Subtract the padded rows distance to B
        pad_dist_sq = jnp.maximum(
            jnp.sum(A_pad[na:] ** 2, axis=-1, keepdims=True)
            + b_sq.T
            - 2.0 * jnp.dot(A_pad[na:], B.T),
            0.0,
        )
        total_sum = total_sum - jnp.sum(jnp.sqrt(pad_dist_sq))

    if is_same and unbiased:
        return total_sum / (na * (na - 1))
    return total_sum / (na * B.shape[0])


@partial(
    jax.jit, static_argnames=("bandwidths", "kernel", "n_repeats", "subsample_size")
)
def _compute_mmd_batched(
    key: jax.Array,
    x: jax.Array,
    y: jax.Array,
    bandwidths: Tuple[float, ...],
    kernel: str,
    n_repeats: int,
    subsample_size: int,
) -> Tuple[jax.Array, jax.Array]:
    nx, ny = x.shape[0], y.shape[0]
    bw = jnp.asarray(bandwidths, dtype=jnp.float32)[:, None, None]  # (B, 1, 1)

    key_x, key_y = jax.random.split(key)
    keys_x = jax.random.split(key_x, n_repeats)
    keys_y = jax.random.split(key_y, n_repeats)

    # Subsample indices for all repeats in one vectorized step: shape (R, subsample_size)
    idx_x = jax.vmap(
        lambda k: jax.random.choice(k, nx, shape=(subsample_size,), replace=False)
    )(keys_x)
    idx_y = jax.vmap(
        lambda k: jax.random.choice(k, ny, shape=(subsample_size,), replace=False)
    )(keys_y)

    def single_mmd(ix: jax.Array, iy: jax.Array) -> jax.Array:
        xs = x[ix]  # (m, D)
        ys = y[iy]  # (m, D)

        dxx = _pairwise_sq_dists(xs, xs)  # (m, m)
        dyy = _pairwise_sq_dists(ys, ys)  # (m, m)
        dxy = _pairwise_sq_dists(xs, ys)  # (m, m)

        if kernel == "rbf":
            Kxx = jnp.sum(jnp.exp(-0.5 * dxx / bw), axis=0)
            Kyy = jnp.sum(jnp.exp(-0.5 * dyy / bw), axis=0)
            Kxy = jnp.sum(jnp.exp(-0.5 * dxy / bw), axis=0)
        elif kernel == "multiscale":
            bw2 = bw**2
            Kxx = jnp.sum(bw2 / (bw2 + dxx), axis=0)
            Kyy = jnp.sum(bw2 / (bw2 + dyy), axis=0)
            Kxy = jnp.sum(bw2 / (bw2 + dxy), axis=0)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        m = subsample_size
        xx = (jnp.sum(Kxx) - jnp.trace(Kxx)) / (m * (m - 1))
        yy = (jnp.sum(Kyy) - jnp.trace(Kyy)) / (m * (m - 1))
        xy = jnp.mean(Kxy)
        return xx + yy - 2.0 * xy

    # Vectorize across all repeats simultaneously
    vals = jax.vmap(single_mmd)(idx_x, idx_y)
    mean_val = jnp.mean(vals)
    sd_val = jnp.std(vals, ddof=1) if n_repeats > 1 else jnp.array(0.0)
    return mean_val, sd_val


# ==============================================================================
# 1. Optimized Energy Distance
# ==============================================================================
def energy_distance(
    samples_p: Union[np.ndarray, jax.Array],
    samples_q: Union[np.ndarray, jax.Array],
    output_file: str = "",
    max_samples: int = -1,
    key: Optional[jax.Array] = None,
    seed: int = 0,
    unbiased: bool = False,
    chunk_size: Optional[int] = None,
) -> jax.Array:
    """Computes the Energy Distance between two empirical distributions in JAX.

    ED = 2 * E||X - Y|| - E||X - X'|| - E||Y - Y'||

    Parameters:
    - samples_p: Array of shape (N, D) or (N,).
    - samples_q: Array of shape (M, D) or (M,).
    - output_file: Optional path to append results.
    - max_samples: Optional cap on the number of samples to use.
    - key: Optional JAX PRNGKey. If None, generated from `seed`.
    - unbiased: If True, uses the unbiased U-statistic (dividing by n(n-1)).
    - chunk_size: If provided and N > chunk_size, uses memory-safe jax.lax.scan chunking.
    """
    X = jnp.asarray(samples_p, dtype=jnp.float32)
    Y = jnp.asarray(samples_q, dtype=jnp.float32)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)

    nx, ny = X.shape[0], Y.shape[0]
    m = min(nx, ny)
    if max_samples > 0:
        m = min(m, max_samples)

    if key is None:
        key = jax.random.PRNGKey(seed)

    # Subsampling without replacement if necessary
    if nx > m:
        key, kx = jax.random.split(key)
        X = X[jax.random.choice(kx, nx, shape=(m,), replace=False)]
    if ny > m:
        key, ky = jax.random.split(key)
        Y = Y[jax.random.choice(ky, ny, shape=(m,), replace=False)]

    # Compute pairwise mean distance
    if chunk_size is not None and max(X.shape[0], Y.shape[0]) > chunk_size:
        exy = _mean_dist_chunked(X, Y, chunk_size=chunk_size)
        exx = _mean_dist_chunked(
            X, X, chunk_size=chunk_size, is_same=True, unbiased=unbiased
        )
        eyy = _mean_dist_chunked(
            Y, Y, chunk_size=chunk_size, is_same=True, unbiased=unbiased
        )
    else:
        exy = _mean_dist_full(X, Y, is_same=False, unbiased=unbiased)
        exx = _mean_dist_full(X, X, is_same=True, unbiased=unbiased)
        eyy = _mean_dist_full(Y, Y, is_same=True, unbiased=unbiased)

    ed = 2.0 * exy - exx - eyy

    if output_file:
        with open(output_file, "a") as f:
            f.write(f"Energy distance: {float(ed):.6f}\n")

    return ed


# ==============================================================================
# 2. Optimized Maximum Mean Discrepancy (MMD^2)
# ==============================================================================
def MMD(
    x: Union[np.ndarray, jax.Array],
    y: Union[np.ndarray, jax.Array],
    output_file: str = "",
    kernel: str = "rbf",
    bandwidths: Optional[Sequence[float]] = None,
    n_repeats: int = 10,
    subsample_size: int = 1000,
    key: Optional[jax.Array] = None,
    seed: int = 0,
) -> Tuple[jax.Array, jax.Array]:
    """Computes the unbiased U-statistic estimator of squared MMD in JAX.

    Vectorized across `n_repeats` and bandwidths via XLA.
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)

    if bandwidths is None:
        if kernel == "rbf":
            bandwidths = (10.0, 15.0, 20.0, 50.0)
        elif kernel == "multiscale":
            bandwidths = (0.2, 0.5, 0.9, 1.3)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
    else:
        bandwidths = tuple(bandwidths)

    nx, ny = x.shape[0], y.shape[0]
    if subsample_size > min(nx, ny):
        raise ValueError(
            f"subsample_size={subsample_size} must be <= min(len(x), len(y)) = {min(nx, ny)}"
        )

    if key is None:
        key = jax.random.PRNGKey(seed)

    mean_mmd2, sd_mmd2 = _compute_mmd_batched(
        key,
        x,
        y,
        bandwidths,
        kernel=kernel,
        n_repeats=n_repeats,
        subsample_size=subsample_size,
    )

    if output_file:
        with open(output_file, "a") as f:
            f.write(f"MMD: {float(mean_mmd2):.6f} +/- {float(sd_mmd2):.6f}\n")

    return mean_mmd2, sd_mmd2
