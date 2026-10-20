"""Color target detector in numpy: color match with tolerance, blobs, aim point.

detect(img, colors, tol, aim=...) finds every blob whose pixels are within
`tol` of any of `colors`, picks the blob nearest `prefer` (the tracker's
prediction) or the crosshair, and returns the aim point on it.
"""
import numpy as np


class Detection:
    def __init__(self, found=False, x=0.0, y=0.0, radius=1.0, boxes=(), chosen=-1, box_w=2.0, box_h=2.0):
        self.found, self.x, self.y, self.radius = found, x, y, radius
        self.boxes, self.chosen = list(boxes), chosen
        self.box_w, self.box_h = box_w, box_h      # size of the chosen target's bounding box


def color_mask(img, colors, tol):
    """Pixels within `tol` (about 0-255) of any color, using the 'redmean' color distance."""
    f = img.astype(np.float32)
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    mask = np.zeros(img.shape[:2], bool)
    lim = (3.0 * tol) ** 2
    for c in colors:
        cr, cg, cb = (float(v) for v in c[:3])
        rm = (R + cr) * 0.5
        dr, dg, db = R - cr, G - cg, B - cb
        d2 = (2 + rm / 256) * dr * dr + 4 * dg * dg + (2 + (255 - rm) / 256) * db * db
        mask |= d2 < lim
    return mask


def label(mask):
    """Connected components (4-neighbour). Background gets the value h*w+1."""
    h, w = mask.shape
    big = h * w + 1
    lab = np.where(mask, np.arange(1, h * w + 1).reshape(h, w), big)
    for _ in range(100):
        m = lab.copy()
        np.minimum(m[1:], lab[:-1], out=m[1:])
        np.minimum(m[:-1], lab[1:], out=m[:-1])
        np.minimum(m[:, 1:], lab[:, :-1], out=m[:, 1:])
        np.minimum(m[:, :-1], lab[:, 1:], out=m[:, :-1])
        m[~mask] = big
        # Pointer jumping: take the label of the pixel your label points at.
        flat = m.ravel()
        sel = flat < big
        flat[sel] = flat[flat[sel] - 1]
        if np.array_equal(m, lab):
            break
        lab = m
    return lab


def detect(img, colors, tol, aim="center", head_pct=15.0, prefer=None, min_px=4, block=2):
    if not colors:
        return Detection()
    mask = color_mask(img, colors, tol)
    ys, xs = np.nonzero(mask)
    if len(ys) < min_px:
        return Detection()
    h, w = mask.shape

    # Group pixels on a coarser grid so labelling stays fast at high frame rates.
    coarse = np.zeros(((h + block - 1) // block, (w + block - 1) // block), bool)
    coarse[ys // block, xs // block] = True
    L = label(coarse)[ys // block, xs // block]
    _, inv = np.unique(L, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)
    cx = np.bincount(inv, xs) / cnt
    cy = np.bincount(inv, ys) / cnt
    nb = len(cnt)
    x0, y0 = np.full(nb, w), np.full(nb, h)
    x1, y1 = np.zeros(nb, int), np.zeros(nb, int)
    np.minimum.at(x0, inv, xs)
    np.minimum.at(y0, inv, ys)
    np.maximum.at(x1, inv, xs)
    np.maximum.at(y1, inv, ys)

    good = cnt >= min_px
    if not good.any():
        return Detection()
    boxes = [(int(x0[i]), int(y0[i]), int(x1[i]), int(y1[i])) for i in np.flatnonzero(good)]
    ref = prefer if prefer is not None else (w / 2, h / 2)
    dist = np.hypot(cx - ref[0], cy - ref[1])
    dist[~good] = np.inf
    k = int(np.argmin(dist))

    sel = inv == k
    bx, by = xs[sel], ys[sel]
    height = y1[k] - y0[k] + 1
    if aim == "head":
        band = by <= y0[k] + max(1.0, height * head_pct / 100)
    elif aim == "body":
        band = (by >= y0[k] + 0.3 * height) & (by <= y0[k] + 0.6 * height)
    else:
        band = np.ones(len(bx), bool)
    if not band.any():
        band = np.ones(len(bx), bool)

    if aim == "closest":
        j = int(np.argmin(np.hypot(bx - w / 2, by - h / 2)))
        px, py = float(bx[j]), float(by[j])
        rad = max(1.0, np.sqrt(cnt[k] / np.pi) * 0.35)
    else:
        px, py = float(bx[band].mean()), float(by[band].mean())
        rad = max(1.0, np.sqrt(band.sum() / np.pi))
        # Hollow target (an outline or ring): few pixels for its size. Its size is the inside of
        # the frame, not the pixel count, and its center is the middle of the frame.
        bw, bh = x1[k] - x0[k] + 1, y1[k] - y0[k] + 1
        if cnt[k] < 0.45 * bw * bh and min(bw, bh) >= 6:
            rad = max(1.0, 0.4 * min(bw, bh))
            if aim == "center":
                px, py = (x0[k] + x1[k]) / 2.0, (y0[k] + y1[k]) / 2.0
    chosen = int(np.flatnonzero(np.flatnonzero(good) == k)[0])
    return Detection(True, px + 0.5, py + 0.5, rad, boxes, chosen,
                     float(x1[k] - x0[k] + 1), float(y1[k] - y0[k] + 1))
