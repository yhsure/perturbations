"""Smooth projected vectors and decoded values for plotting on a shared grid."""

import numpy as np
from scipy.spatial import cKDTree


def smooth_field(xy, vectors, values, gridsize=100, neighbors=30, bandwidth=None):
    """Return ``x, y, U, V, field`` for streamplot and a background colour map.

    Inputs are matching (cells, 2) coordinates/vectors and (cells,) values.
    The vectors must already be projected; they are never inferred from the
    scalar values. Vectors are averaged without normalizing them. Unsupported
    grid positions are masked in all three returned arrays, whose shape is
    (len(y), len(x)). Grid cells are square, with about ``gridsize`` positions
    across the longest dimension. ``bandwidth`` is in coordinate units and
    controls smoothing and support. Supply it explicitly to refine the grid
    without shrinking the physical neighbourhood (default: 1.1 grid steps).

    Support is limited by nearby samples and local spacing, capped at twice
    the bandwidth. At least three distinct neighbours must lie within this
    radius (all queried neighbours if fewer). Gaps and isolated cells remain
    unfilled; individual cells can still be shown separately as points.
    The result is a display interpolation, not a new decoder evaluation.
    """
    xy, vectors, values = (np.asarray(a, dtype=float) for a in (xy, vectors, values))
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy):
        raise ValueError("xy must be a nonempty (cells, 2) array")
    if vectors.shape != xy.shape or values.shape != (len(xy),):
        raise ValueError("vectors and values must match the coordinate rows")
    if not all(np.isfinite(a).all() for a in (xy, vectors, values)):
        raise ValueError("All coordinates, vectors and values must be finite")
    for name, value, minimum in [("gridsize", gridsize, 3), ("neighbors", neighbors, 1)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")

    # Average coincident samples so duplicates do not distort neighbour spacing.
    points, inverse, counts = np.unique(xy, axis=0, return_inverse=True, return_counts=True)
    samples = np.column_stack([vectors, values])
    combined = np.zeros((len(points), 3))
    np.add.at(combined, inverse, samples)
    samples = combined / counts[:, None]
    extent = max(np.ptp(points, axis=0).max(), np.finfo(float).eps)
    if len(points) == 1:
        extent = 1.0  # A finite display window for a single coincident position.
    lo, hi = points.min(0) - .045 * extent, points.max(0) + .045 * extent
    step = max(hi - lo) / (gridsize - 1)
    x = np.arange(lo[0], hi[0] + .5 * step, step)
    y = np.arange(lo[1], hi[1] + .5 * step, step)
    if len(x) < 2:
        x = np.mean(points[:, 0]) + np.array([-.5, .5]) * step
    if len(y) < 2:
        y = np.mean(points[:, 1]) + np.array([-.5, .5]) * step
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    bandwidth = 1.1 * step if bandwidth is None else float(bandwidth)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive")
    h = bandwidth / 1.1

    tree = cKDTree(points)
    k = min(neighbors, len(points))
    distances, indices = tree.query(grid, k=k)
    distances, indices = distances.reshape(len(grid), k), indices.reshape(len(grid), k)
    spacing = tree.query(points, k=min(5, len(points)))[0]
    spacing = spacing[:, -1] if spacing.ndim == 2 else spacing
    radius = np.minimum(2.2 * h, .75 * spacing[indices[:, 0]] + h)
    # Keep the interpolation kernel local at any grid resolution.
    weights = np.exp(-.5 * (distances / bandwidth) ** 2)
    weights[distances > 3 * h] = 0
    mass = weights.sum(1)
    support = (distances[:, 0] < radius) & (mass > .25)
    support &= distances[:, min(2, k - 1)] < radius
    weights /= np.maximum(mass[:, None], np.finfo(float).tiny)
    smoothed = (samples[indices] * weights[:, :, None]).sum(1)
    mask = ~support.reshape(xx.shape)
    fields = [np.ma.array(smoothed[:, i].reshape(xx.shape), mask=mask) for i in range(3)]
    return x, y, *fields
