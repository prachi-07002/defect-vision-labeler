"""Synthetic surface-defect image generation (pure NumPy).

Classes: clean, scratch, dent, crack.
Images are 64x64 grayscale float arrays in [0, 1]. Saved as ASCII PGM (P2)
so everything on disk stays plain text — no binary files.
"""

import os

import numpy as np

CLASSES = ("clean", "scratch", "dent", "crack")
SIZE = 64


def _base_surface(rng):
    """Clean metal-like surface: smooth illumination gradient + fine noise."""
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    grad = 0.08 * (x / SIZE) + 0.05 * (y / SIZE)
    noise = rng.normal(0.0, 0.02, size=(SIZE, SIZE))
    # faint machining texture: horizontal bands
    bands = 0.012 * np.sin(2 * np.pi * y / 7.0 + rng.uniform(0, 6))
    img = 0.62 + grad + bands + noise
    return np.clip(img, 0.0, 1.0)


def _add_scratch(img, rng):
    """Thin bright line across the surface, slight angle, with soft edges."""
    out = img.copy()
    angle = rng.uniform(-0.5, 0.5)
    y0 = rng.uniform(8, SIZE - 8)
    x = np.arange(SIZE)
    y_center = y0 + np.tan(angle) * (x - SIZE / 2)
    width = rng.uniform(0.8, 1.6)
    depth = rng.uniform(0.25, 0.45)
    yy = np.arange(SIZE)[:, None]
    dist = np.abs(yy - y_center[None, :])
    mask = np.exp(-(dist ** 2) / (2 * width ** 2))
    out = np.clip(out + depth * mask, 0.0, 1.0)
    return out


def _add_dent(img, rng):
    """Circular dent: dark centre fading out, with a bright rim ring."""
    out = img.copy()
    cx, cy = rng.uniform(18, SIZE - 18, size=2)
    r = rng.uniform(7, 13)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    centre = np.exp(-(d ** 2) / 0.35) * rng.uniform(0.28, 0.42)
    rim = np.exp(-((d - 1.0) ** 2) / 0.08) * rng.uniform(0.10, 0.18)
    out = np.clip(out - centre + rim, 0.0, 1.0)
    return out


def _add_crack(img, rng):
    """Jagged branching crack: a random-walk dark polyline, 1-2 px wide."""
    out = img.copy()
    x, y = rng.uniform(6, SIZE - 6), rng.uniform(6, SIZE - 6)
    heading = rng.uniform(0, 2 * np.pi)
    steps = rng.integers(28, 48)
    depth = rng.uniform(0.30, 0.5)
    for _ in range(int(steps)):
        heading += rng.normal(0, 0.45)
        x += 1.6 * np.cos(heading)
        y += 1.6 * np.sin(heading)
        if not (1 <= x < SIZE - 1 and 1 <= y < SIZE - 1):
            break
        ix, iy = int(round(x)), int(round(y))
        out[iy - 1:iy + 2, ix - 1:ix + 2] -= depth * 0.55
        out[iy, ix] -= depth * 0.45
        # occasional branch
        if rng.random() < 0.12:
            bx, by = x, y
            bhead = heading + rng.uniform(0.6, 1.4) * rng.choice([-1, 1])
            for _ in range(rng.integers(4, 9)):
                bhead += rng.normal(0, 0.4)
                bx += 1.4 * np.cos(bhead)
                by += 1.4 * np.sin(bhead)
                if not (1 <= bx < SIZE - 1 and 1 <= by < SIZE - 1):
                    break
                out[int(round(by)), int(round(bx))] -= depth * 0.4
    return np.clip(out, 0.0, 1.0)


_GENERATORS = {
    "clean": lambda img, rng: img,
    "scratch": _add_scratch,
    "dent": _add_dent,
    "crack": _add_crack,
}


def generate_image(label, rng):
    """Generate one synthetic surface image of the given class."""
    if label not in CLASSES:
        raise ValueError(f"unknown label {label!r}; expected one of {CLASSES}")
    img = _base_surface(rng)
    return _GENERATORS[label](img, rng)


def generate_dataset(n_per_class=100, seed=42):
    """Return (images, labels): images shape (N, 64, 64), labels list of str."""
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for label in CLASSES:
        for _ in range(n_per_class):
            images.append(generate_image(label, rng))
            labels.append(label)
    return np.stack(images), labels


def save_pgm(path, img):
    """Write a float image in [0,1] as ASCII PGM (P2, plain text)."""
    arr = np.clip(np.rint(img * 255), 0, 255).astype(int)
    h, w = arr.shape
    with open(path, "w") as f:
        f.write("P2\n# defect-vision-labeler synthetic surface\n")
        f.write(f"{w} {h}\n255\n")
        for row in arr:
            f.write(" ".join(str(v) for v in row) + "\n")


def load_pgm(path):
    """Read an ASCII PGM (P2) back into a float array in [0, 1]."""
    tokens = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                tokens.extend(line.split())
    if tokens[0] != "P2":
        raise ValueError(f"{path}: not an ASCII PGM (P2)")
    w, h, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    vals = np.array([int(t) for t in tokens[4:4 + w * h]], dtype=float)
    if vals.size != w * h:
        raise ValueError(f"{path}: expected {w*h} pixels, got {vals.size}")
    return (vals.reshape(h, w) / maxval)


def save_dataset(directory, n_per_class=100, seed=42):
    """Generate the dataset, save images as .pgm + a manifest CSV.

    Returns the manifest rows: list of dicts with image_id, file, label.
    """
    os.makedirs(directory, exist_ok=True)
    images, labels = generate_dataset(n_per_class, seed)
    rows = []
    for i, (img, label) in enumerate(zip(images, labels)):
        image_id = f"img_{i:04d}"
        fname = f"{image_id}.pgm"
        save_pgm(os.path.join(directory, fname), img)
        rows.append({"image_id": image_id, "file": fname, "label": label})
    import csv
    with open(os.path.join(directory, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "file", "label"])
        w.writeheader()
        w.writerows(rows)
    return rows
