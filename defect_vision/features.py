"""Hand-engineered features for surface-defect images (pure NumPy).

Features are chosen so each defect type leaves a distinct signature:
- scratch: long bright line -> high max-row bright mass, high line score
- dent: compact dark blob near the centre -> high centre-dark mass
- crack: jagged dark polyline -> high dark-pixel count spread over many rows
- clean: low everything
"""

import numpy as np

FEATURE_NAMES = [
    "mean",
    "std",
    "grad_mean",        # mean gradient magnitude (edge energy)
    "grad_max",         # max gradient magnitude
    "bright_frac",      # fraction of pixels well above the mean
    "dark_frac",        # fraction of pixels well below the mean
    "max_row_dark",     # largest dark-pixel count in any single row
    "max_col_dark",     # largest dark-pixel count in any single column
    "max_row_bright",   # largest bright-pixel count in any single row
    "centre_dark",      # dark-pixel fraction inside the central 32x32 patch
    "dark_spread",      # fraction of rows containing at least one dark pixel
    "line_score",       # longest run of bright pixels along any row
]


def _longest_run(mask_row):
    best = cur = 0
    for v in mask_row:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def extract_features(img):
    """Extract the 12 engineered features from one 2-D float image."""
    img = np.asarray(img, dtype=float)
    mean = float(img.mean())
    std = float(img.std())
    gy, gx = np.gradient(img)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad_mean = float(grad.mean())
    grad_max = float(grad.max())
    bright = img > mean + 0.12  # absolute offset: noise (std ~0.03) rarely crosses
    dark = img < mean - 0.12
    bright_frac = float(bright.mean())
    dark_frac = float(dark.mean())
    max_row_dark = int(dark.sum(axis=1).max())
    max_col_dark = int(dark.sum(axis=0).max())
    max_row_bright = int(bright.sum(axis=1).max())
    c = img[16:48, 16:48]
    cm = c.mean()
    centre_dark = float((c < cm - 0.12).mean())
    dark_spread = float((dark.sum(axis=1) > 0).mean())
    line_score = max(_longest_run(r) for r in bright)
    return np.array([
        mean, std, grad_mean, grad_max, bright_frac, dark_frac,
        max_row_dark, max_col_dark, max_row_bright, centre_dark,
        dark_spread, float(line_score),
    ])


def extract_all(images):
    """Extract features for a stack of images -> (N, 12) array."""
    return np.stack([extract_features(im) for im in images])


def standardize(X, mean=None, std=None):
    """Z-score standardize; returns (X_std, mean, std)."""
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std
