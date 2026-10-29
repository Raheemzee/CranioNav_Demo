"""
CranioNav - AR navigation simulator for brain tumour removal on BraTS cases.

Run:   pip install -r requirements.txt
       python app.py
Open:  http://127.0.0.1:5160

Hosted demo (Render, see README): `gunicorn app:app` serves the bundled cases read-only. On Render
(or with CRANIONAV_DEMO=1) nothing is built or written: no BraTS folder, no database, no sessions.

The server finds BraTS cases under DATA_DIR, turns each one into a navigation
package (tumour, ventricle and functional-area labels, 3D meshes, a patient
head fitted around the brain, an estimated skull, dura and venous sinuses, and an automatic
craniotomy plan)
and serves the browser simulator. Packages are cached in cases/.
"""
import base64
import glob
import io
import json
import math
import os
import re
import threading
import time
import traceback

import numpy as np

# ------------------------------------------------------------------ settings (edit these)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "CRANIONAV_DATA")
CASES_DIR = os.path.join(BASE_DIR, "cases")            # built packages (cache, also used by the static preview)
ATLAS_DIR = os.path.join(BASE_DIR, "atlas")
HEAD_GLTF = os.path.join(BASE_DIR, "assets", "head", "LeePerrySmith.gltf.json")
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
SEGRESNET_CHECKPOINT = ""     # optional MONAI SegResNet .pth (4 channels FLAIR,T1,T1CE,T2 -> TC,WT,ET); used when a case has no seg file
LANGUAGE_HEMISPHERE = "left"  # dominant hemisphere for the language areas
HOST, PORT = "127.0.0.1", 5160
# Hosted demo: Render sets RENDER and PORT. Demo mode serves the bundled cases only and writes nothing.
DEMO = os.environ.get("CRANIONAV_DEMO", "").lower() in ("1", "true", "yes") or bool(os.environ.get("RENDER"))
if os.environ.get("PORT"):
    HOST, PORT = "0.0.0.0", int(os.environ["PORT"])
PACKAGE_VERSION = 4

# ------------------------------------------------------------------ constants
L_OUT, L_BRAIN, L_NCR, L_ED, L_ET, L_VENT = 0, 1, 2, 3, 4, 5
LABEL_NAMES = {L_BRAIN: "Brain", L_NCR: "Necrotic / non-enhancing core", L_ED: "Oedema",
               L_ET: "Enhancing tumour", L_VENT: "Lateral ventricle"}
FUNC_NAMES = {1: "Motor cortex (precentral gyrus)", 2: "Sensory cortex (postcentral gyrus)",
              3: "Language - Broca area", 4: "Language - Wernicke area", 5: "Visual cortex (V1)"}
# Harvard-Oxford cortical maxprob indices (FSL order, 1-based)
HO_PRECENTRAL, HO_POSTCENTRAL = 7, 17
HO_BROCA = (5, 6)                 # IFG pars triangularis, pars opercularis
HO_WERNICKE = (10, 20, 46)        # STG posterior, supramarginal posterior, planum temporale
HO_VISUAL = (24, 32, 47, 48)      # intracalcarine, cuneal, supracalcarine, occipital pole

# Head scan (Lee Perry-Smith "Infinite" scan) facts, in scan units
SCAN_MM = 51.0                                   # millimetres per scan unit before fitting
SCAN_TO_RAS = np.array([[-1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])   # scan +x = patient left, +y = up, +z = face
SCAN_LANDMARKS = {
    "nasion": [-0.105, 1.86, 2.128], "nose_tip": [-0.105, 1.10, 2.586],
    "canthus_r": [-1.01, 1.655, 1.735], "canthus_l": [0.80, 1.65, 1.698],
    "tragus_r": [-1.535, 1.17, 0.40], "tragus_l": [1.341, 1.17, 0.25],
}
SCAN_CRANIUM = [-0.105, 2.35, 0.35]
# Scalp-to-brain clearance measured on a real BraTS-space MRI with scalp (TCGA-DU-7294), mm
CLEARANCE_TARGET = {"vertex": 16.0, "upper": 16.0, "frontal": 17.0, "temporal": 12.0, "occipital": 14.0}
# Brain-hull to outer-table distance and CSF/dura gap, measured on the same MRI (median of radial profiles), mm
SKULL_OUTER_MM = {"default": 8.0, "frontal": 10.5, "temporal": 7.0, "occipital": 7.0}
SKULL_GAP_MM = 2.0

MOD_RULES = [
    ("seg", r"(seg|segmentation|label|labels|gt)$"),
    ("t1ce", r"(t1ce|t1c|t1gd|t1_gd|t1post|t1_ce)$"),
    ("flair", r"(flair|t2f|fla|t2flair)$"),
    ("t2", r"(t2|t2w)$"),
    ("t1", r"(t1|t1n|t1w)$"),
]
MOD_ORDER = ["t1ce", "flair", "t2", "t1"]
BUILD_STATUS = {}
BUILD_LOCK = threading.Lock()


# ================================================================== case discovery
def split_modality(path):
    stem = re.sub(r"\.nii(\.gz)?$", "", os.path.basename(path), flags=re.I)
    low = stem.lower()
    for key, rx in MOD_RULES:
        m = re.search(r"(?:^|[_\-.])" + rx, low)
        if m:
            return key, stem[:m.start()].rstrip("_-.")
    return None, None


def case_key(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:80] or "case"


_DISCOVERY = {"t": 0, "cases": {}}


def discover_cases(force=False):
    if not force and time.time() - _DISCOVERY["t"] < 20:
        return _DISCOVERY["cases"]
    cases = {}
    if os.path.isdir(DATA_DIR):
        for root, dirs, files in os.walk(DATA_DIR):
            depth = root[len(DATA_DIR):].count(os.sep)
            if depth > 3:
                dirs[:] = []
                continue
            groups = {}
            for f in files:
                if not re.search(r"\.nii(\.gz)?$", f, re.I):
                    continue
                mod, prefix = split_modality(f)
                if not mod:
                    continue
                groups.setdefault(prefix or os.path.basename(root), {})[mod] = os.path.join(root, f)
            for prefix, mods in groups.items():
                if not any(m in mods for m in MOD_ORDER):
                    continue
                name = prefix if len(groups) > 1 else (os.path.basename(root) or prefix)
                key = case_key(name)
                while key in cases:
                    key += "_"
                cases[key] = {"key": key, "name": name, "dir": root, "files": mods}
    _DISCOVERY.update(t=time.time(), cases=cases)
    return cases


# ================================================================== volume helpers
def _nib():
    import nibabel as nib
    return nib


def load_ras(path):
    nib = _nib()
    img = nib.as_closest_canonical(nib.load(path))
    return np.asanyarray(img.dataobj).astype(np.float32), img.affine


def largest_cc(mask, keep_ratio=0.0, min_size=0):
    from scipy import ndimage as ndi
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    big = sizes.max()
    keep = [i + 1 for i, s in enumerate(sizes) if s >= max(min_size, 1) and (s == big or (keep_ratio and s >= keep_ratio * big))]
    return np.isin(lab, keep)


def robust_z(v, mask):
    x = v[mask]
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-6
    return ((v - med) / mad).astype(np.float32)


def to_u8(v, mask, lo_p=0.5, hi_p=99.5):
    x = v[mask]
    lo, hi = np.percentile(x, [lo_p, hi_p])
    out = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1) * 254 + 1
    out[~mask] = 0
    return out.astype(np.uint8)


def fill_slices(m):
    from scipy import ndimage as ndi
    out = m.copy()
    for k in range(m.shape[2]):
        if out[:, :, k].any():
            out[:, :, k] = ndi.binary_fill_holes(out[:, :, k])
    return out


# ================================================================== tumour
def tumour_from_seg(seg):
    s = np.rint(seg).astype(np.int16)
    u = set(np.unique(s).tolist())
    t = np.zeros(s.shape, np.uint8)
    if 4 in u:
        t[(s == 1) | (s == 3)] = L_NCR
        t[s == 2] = L_ED
        t[s == 4] = L_ET
        scheme = "BraTS 2017-2021 labels (1 NCR, 2 ED, 4 ET)"
    else:
        t[s == 1] = L_NCR
        t[s == 2] = L_ED
        t[s == 3] = L_ET
        scheme = "BraTS 2023 labels (1 NCR, 2 ED, 3 ET)"
    return t, scheme


def tumour_auto(z, mask):
    """Intensity heuristic used when a case has no segmentation file.
    Oedema: FLAIR and T2 bright. Cyst/necrosis: very bright T2, not FLAIR-suppressed, dark on T1CE.
    Enhancing tumour: bright T1CE at the lesion margin."""
    from scipy import ndimage as ndi
    t = np.zeros(mask.shape, np.uint8)
    fl, t2, t1c = z.get("flair"), z.get("t2"), z.get("t1ce")
    if fl is None and t2 is None:
        return t
    er = ndi.binary_erosion(mask, iterations=2)
    oed = er & ((fl > 1.7) if fl is not None else (t2 > 2.5))
    if fl is not None and t2 is not None:
        oed &= t2 > 0.8
    cyst = np.zeros_like(oed)
    if t2 is not None:
        cyst = er & (t2 > 3.0)
        if fl is not None:
            cyst &= fl > -0.5
        if t1c is not None:
            cyst &= t1c < 0.5
    cand = ndi.binary_opening(oed | cyst, iterations=2)
    wt = largest_cc(cand, keep_ratio=0.25, min_size=300)
    if wt.sum() < 300:
        return t
    wt = fill_slices(ndi.binary_closing(wt, iterations=3)) & mask
    cyst &= wt
    et = np.zeros_like(wt)
    if t1c is not None:
        near = ndi.binary_dilation(wt, iterations=2) & er & (t1c > 2.0)
        if cyst.any():
            near &= ~ndi.binary_erosion(cyst, iterations=1)
        lab, n = ndi.label(near)
        if n:
            sizes = ndi.sum(near, lab, range(1, n + 1))
            touch = ndi.maximum(wt, lab, range(1, n + 1))
            et = np.isin(lab, [i + 1 for i in range(n) if sizes[i] >= 20 and touch[i]])
    ncr = cyst & ~et
    if et.any():
        hull = fill_slices(ndi.binary_closing(et, iterations=2))
        ncr |= hull & ~et & wt & ((t1c < 1.0) if t1c is not None else True)
    wt |= et
    t[wt] = L_ED
    t[ncr] = L_NCR
    t[et] = L_ET
    return t


def tumour_segresnet(raw, mask):
    """Optional MONAI SegResNet (BraTS layout). Returns None when unavailable."""
    if not SEGRESNET_CHECKPOINT or not os.path.isfile(SEGRESNET_CHECKPOINT):
        return None
    if not all(m in raw for m in ("flair", "t1", "t1ce", "t2")):
        return None
    try:
        import torch
        from monai.inferers import sliding_window_inference
        from monai.networks.nets import SegResNet
    except Exception:
        return None
    net = SegResNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16, in_channels=4, out_channels=3, dropout_prob=0.2)
    state = torch.load(SEGRESNET_CHECKPOINT, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state)
    net.eval()
    chans = []
    for m in ("flair", "t1", "t1ce", "t2"):
        v = raw[m]
        x = v[mask]
        chans.append(np.where(mask, (v - x.mean()) / (x.std() + 1e-6), 0).astype(np.float32))
    with torch.no_grad():
        inp = torch.from_numpy(np.stack(chans)[None])
        out = sliding_window_inference(inp, (128, 128, 128), 1, net, overlap=0.5)
        p = (torch.sigmoid(out)[0].numpy() > 0.5)
    tc, wt, et = p[0], p[1], p[2]
    t = np.zeros(mask.shape, np.uint8)
    t[wt] = L_ED
    t[tc] = L_NCR
    t[et] = L_ET
    return t


# ================================================================== atlas registration
def _euler(rx, ry, rz):
    a, b, c = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def register_atlas(case_img, case_mask, exclude, affine, status=None):
    """Affine (9 dof) registration of the MNI152 brain to the case. Returns a function case_mm -> mni_mm, plus atlas volumes."""
    from scipy import ndimage as ndi
    from scipy.optimize import minimize
    paths = {k: os.path.join(ATLAS_DIR, f) for k, f in {
        "t1": "MNI152NLin6_T1w_brain_1mm.nii.gz", "mask": "MNI152NLin6_brainmask_1mm.nii.gz",
        "ho": "HarvardOxford_cortical_maxprob_thr25_1mm.nii.gz", "vent": "HarvardOxford_lateral_ventricles_1mm.nii.gz"}.items()}
    if not all(os.path.isfile(p) for p in paths.values()):
        return None
    mt1, maff = load_ras(paths["t1"])
    mmask = load_ras(paths["mask"])[0] > 0.5
    ho = np.rint(load_ras(paths["ho"])[0]).astype(np.int16)
    vent = load_ras(paths["vent"])[0] > 0.5
    minv = np.linalg.inv(maff)

    def mm(ijk, aff):
        return ijk @ aff[:3, :3].T + aff[:3, 3]

    c_ijk = np.argwhere(case_mask).astype(np.float64)
    m_ijk = np.argwhere(mmask).astype(np.float64)
    c_mm, m_mm = mm(c_ijk, affine), mm(m_ijk, maff)
    cc, mc = c_mm.mean(0), m_mm.mean(0)
    s0 = m_mm.std(0) / (c_mm.std(0) + 1e-6)

    case_in = case_img.astype(np.float32)
    mni_in = mt1.astype(np.float32)
    valid = case_mask & ~exclude

    def build(p):
        R = _euler(*p[3:6])
        S = np.diag(s0 * np.exp(p[6:9] / 100.0))
        L = R @ S
        return L, mc + p[0:3] - L @ cc

    def run(step, sigma, x0):
        ci = ndi.gaussian_filter(case_in, sigma)
        mi = ndi.gaussian_filter(mni_in, sigma)
        mmk = ndi.gaussian_filter(mmask.astype(np.float32), sigma)
        cmk = ndi.gaussian_filter(case_mask.astype(np.float32), sigma)
        grid = np.zeros_like(valid)
        grid[::step, ::step, ::step] = True
        pts = np.argwhere(valid & grid).astype(np.float64)
        ring = np.argwhere(grid & (ndi.binary_dilation(case_mask, iterations=4) ^ ndi.binary_erosion(case_mask, iterations=4))).astype(np.float64)
        a = ci[tuple(pts.T.astype(int))]
        a = (a - a.mean()) / (a.std() + 1e-6)
        am = cmk[tuple(ring.T.astype(int))]
        am = (am - am.mean()) / (am.std() + 1e-6)
        P, Q = mm(pts, affine), mm(ring, affine)

        def cost(p):
            L, t = build(p)
            q = (P @ L.T + t) @ minv[:3, :3].T + minv[:3, 3]
            b = ndi.map_coordinates(mi, q.T, order=1, mode="constant")
            b = (b - b.mean()) / (b.std() + 1e-6)
            q2 = (Q @ L.T + t) @ minv[:3, :3].T + minv[:3, 3]
            bm = ndi.map_coordinates(mmk, q2.T, order=1, mode="constant")
            bm = (bm - bm.mean()) / (bm.std() + 1e-6)
            return -(np.mean(a * b) + 0.6 * np.mean(am * bm))

        r = minimize(cost, x0, method="Powell", options={"xtol": 1e-2, "ftol": 1e-5, "maxfev": 2500})
        return r.x, -r.fun

    x = np.zeros(9)
    x, score = run(4, 2.0, x)
    if status:
        status("Registering brain atlas (fine)")
    x, score = run(2, 1.0, x)
    L, t = build(x)
    return {"L": L, "t": t, "score": float(score), "ho": ho, "vent": vent, "maff": maff, "minv": minv}


def atlas_to_case(reg, shape, affine, mask):
    """Nearest-neighbour atlas labels on the case grid (inside the brain bounding box)."""
    from scipy import ndimage as ndi
    func = np.zeros(shape, np.uint8)
    vent = np.zeros(shape, bool)
    mni_x = np.zeros(shape, np.float32)
    idx = np.argwhere(mask)
    lo, hi = idx.min(0), idx.max(0) + 1
    ventu8 = reg["vent"].astype(np.uint8)
    for z0 in range(lo[2], hi[2], 16):
        z1 = min(z0 + 16, hi[2])
        gi = np.stack(np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), np.arange(z0, z1), indexing="ij"), -1).reshape(-1, 3).astype(np.float64)
        q_mm = (gi @ affine[:3, :3].T + affine[:3, 3]) @ reg["L"].T + reg["t"]
        q = q_mm @ reg["minv"][:3, :3].T + reg["minv"][:3, 3]
        ho = ndi.map_coordinates(reg["ho"], q.T, order=0, mode="constant").astype(np.int16)
        ve = ndi.map_coordinates(ventu8, q.T, order=0, mode="constant") > 0
        f = np.zeros(len(gi), np.uint8)
        left = q_mm[:, 0] < 0
        dominant = left if LANGUAGE_HEMISPHERE == "left" else ~left
        f[ho == HO_PRECENTRAL] = 1
        f[ho == HO_POSTCENTRAL] = 2
        f[np.isin(ho, HO_BROCA) & dominant] = 3
        f[np.isin(ho, HO_WERNICKE) & dominant] = 4
        f[np.isin(ho, HO_VISUAL)] = 5
        sub = (hi[0] - lo[0], hi[1] - lo[1], z1 - z0)
        sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(z0, z1))
        func[sl] = f.reshape(sub)
        vent[sl] = ve.reshape(sub)
        mni_x[sl] = q_mm[:, 0].reshape(sub).astype(np.float32)
    func[~mask] = 0
    # grow labels a few mm so the brain surface picks them up
    lab = func > 0
    if lab.any():
        dist, ind = ndi.distance_transform_edt(~lab, return_indices=True)
        grow = mask & ~lab & (dist <= 3.0)
        func[grow] = func[tuple(i[grow] for i in ind)]
    return func, vent, mni_x


# ================================================================== ventricles
def ventricles(z, mask, tumour, atlas_vent=None, midline_x=None, affine=None):
    from scipy import ndimage as ndi
    s = np.zeros(mask.shape, np.float32)
    w = 0.0
    if "t2" in z:
        s += np.clip(z["t2"], -6, 10); w += 1
    if "flair" in z:
        s -= np.clip(z["flair"], -10, 6); w += 1
    if "t1" in z:
        s -= np.clip(z["t1"], -10, 6); w += 1
    if "t1ce" in z:
        s -= 0.7 * np.clip(z["t1ce"], -10, 6); w += 0.7
    if w == 0:
        return np.zeros(mask.shape, bool)
    s /= w
    depth = ndi.distance_transform_edt(mask)
    deep = mask & (depth > 9) & ~ndi.binary_dilation(tumour > 0, iterations=2)
    thr = max(float(np.percentile(s[deep], 93)), 1.2)
    c = deep & (s > thr)
    if atlas_vent is not None and atlas_vent.any():
        c &= ndi.binary_dilation(atlas_vent, iterations=8)
    c = ndi.binary_opening(c, iterations=1)
    lab, n = ndi.label(c)
    if n == 0:
        return np.zeros(mask.shape, bool)
    sizes = ndi.sum(c, lab, range(1, n + 1))
    cents = ndi.center_of_mass(c, lab, range(1, n + 1))
    keep = []
    big = sizes.max()
    cx = midline_x if midline_x is not None else np.argwhere(mask)[:, 0].mean()
    for i in np.argsort(-sizes):
        if sizes[i] < 600 or len(keep) >= 3:
            break
        if abs(cents[i][0] - cx) > 40 or depth[tuple(int(round(v)) for v in cents[i])] < 14:
            continue
        if sizes[i] >= 0.25 * big:
            keep.append(i + 1)
    v = np.isin(lab, keep)
    v = ndi.binary_closing(v, iterations=2) & mask
    return fill_slices(v)


# ================================================================== head scan fitting
def load_head_mesh():
    g = json.load(open(HEAD_GLTF))
    buf = base64.b64decode(g["buffers"][0]["uri"].split(",", 1)[1])

    def acc(i):
        a = g["accessors"][i]
        bv = g["bufferViews"][a["bufferView"]]
        n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
        dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[a["componentType"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        r = np.frombuffer(buf, dt, count=a["count"] * n, offset=off)
        return r.reshape(-1, n) if n > 1 else r
    prim = g["meshes"][0]["primitives"][0]
    return acc(prim["attributes"]["POSITION"]).astype(np.float64), acc(prim["indices"]).reshape(-1, 3).astype(np.int64)


def raycast(P, T, orig, dirs, eps=1e-9, batch=48):
    """Moller-Trumbore against all triangles. Returns nearest positive hit distance per ray (inf if none)."""
    v0 = P[T[:, 0]]
    e1 = P[T[:, 1]] - v0
    e2 = P[T[:, 2]] - v0
    dirs = np.atleast_2d(np.asarray(dirs, float))
    N = len(dirs)
    orig = np.broadcast_to(np.asarray(orig, float), (N, 3))
    out = np.full(N, np.inf)
    for s in range(0, N, batch):
        d = dirs[s:s + batch, None, :]
        o = orig[s:s + batch, None, :]
        h = np.cross(d, e2[None])
        a = np.einsum("nmk,mk->nm", h, e1)
        ok = np.abs(a) > eps
        f = np.where(ok, 1.0 / np.where(ok, a, 1), 0)
        sv = o - v0[None]
        u = f * np.einsum("nmk,nmk->nm", sv, h)
        q = np.cross(sv, e1[None])
        v = f * np.einsum("nmk,nmk->nm", d, q)
        t = f * np.einsum("mk,nmk->nm", e2, q)
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > eps)
        out[s:s + batch] = np.where(hit, t, np.inf).min(1)
    return out


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], -1)


def fit_head(brain_mask, affine, status=None):
    """Scale and place the head scan so the brain sits inside it with realistic scalp/skull clearance."""
    from scipy import ndimage as ndi
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    P, T = load_head_mesh()
    bd = brain_mask & ~ndi.binary_erosion(brain_mask)
    B = np.argwhere(bd).astype(np.float64) @ affine[:3, :3].T + affine[:3, 3]
    c = (np.argwhere(brain_mask).astype(np.float64) @ affine[:3, :3].T + affine[:3, 3]).mean(0)
    dirs = fib_sphere(900)
    tree = cKDTree(dirs)
    rel = B - c
    rb = np.linalg.norm(rel, axis=1)
    _, bi = tree.query(rel / rb[:, None])
    rbk = np.full(len(dirs), np.nan)
    for k in np.unique(bi):
        rbk[k] = rb[bi == k].max()
    el = np.degrees(np.arcsin(np.clip(dirs[:, 2], -1, 1)))
    az = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0]))   # 90 = anterior
    target = np.full(len(dirs), np.nan)
    target[el > 50] = CLEARANCE_TARGET["vertex"]
    target[(el > 20) & (el <= 50)] = CLEARANCE_TARGET["upper"]
    low = (el > -12) & (el <= 20)
    target[low & (az > 45) & (az < 135)] = CLEARANCE_TARGET["frontal"]
    target[low & ((np.abs(az) < 35) | (np.abs(az) > 145))] = CLEARANCE_TARGET["temporal"]
    target[low & (az < -45) & (az > -135)] = CLEARANCE_TARGET["occipital"]
    use = ~np.isnan(rbk) & ~np.isnan(target)
    lowband = ~np.isnan(rbk) & (el > -35) & (el <= -12)
    # dense head samples: vertices + triangle centroids + edge midpoints
    samp = np.concatenate([P, P[T].mean(1), (P[T[:, 0]] + P[T[:, 1]]) / 2, (P[T[:, 1]] + P[T[:, 2]]) / 2])
    H0 = samp @ (SCAN_TO_RAS * SCAN_MM).T
    h0 = SCAN_TO_RAS @ (np.array(SCAN_CRANIUM) * SCAN_MM)

    def head_radii(p):
        s = np.exp(p[3:6] / 100.0)
        Hf = c + p[0:3] + (H0 - h0) * s
        r = Hf - c
        d = np.linalg.norm(r, axis=1)
        _, hi = tree.query(r / d[:, None])
        rh = np.full(len(dirs), np.inf)
        np.minimum.at(rh, hi, d)
        return rh

    def resid(p):
        rh = head_radii(p)
        clr = rh - rbk
        r1 = np.where(use, np.where(np.isfinite(clr), clr - target, 30.0), 0.0)[use]
        r2 = np.where(lowband, np.minimum(np.where(np.isfinite(clr), clr, 30.0) - 8.0, 0.0), 0.0)[lowband] * 2.0
        reg = np.concatenate([p[3:6] * 0.6, [(p[3] - p[5]) * 0.4]])
        return np.concatenate([r1, r2, reg])

    best = None
    for zoff in (-15.0, 0.0, 15.0):
        r = least_squares(resid, np.array([0.0, 0.0, zoff, 0.0, 0.0, 0.0]), diff_step=0.05, max_nfev=300)
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    s = np.exp(p[3:6] / 100.0)
    L = np.diag(s) @ SCAN_TO_RAS * SCAN_MM
    t = c + p[0:3] - s * h0
    M = np.eye(4)
    M[:3, :3] = L
    M[:3, 3] = t
    rh = head_radii(p)
    clr = (rh - rbk)[use]
    clr = clr[np.isfinite(clr)]
    Pf = P @ L.T + t
    return {"matrix": M, "scale": s.tolist(), "P": Pf, "T": T, "brain_center": c,
            "clearance": {"median": float(np.median(clr)), "p05": float(np.percentile(clr, 5)), "min": float(clr.min())}}


def head_landmarks(M):
    return {k: (M[:3, :3] @ np.array(v) + M[:3, 3]) for k, v in SCAN_LANDMARKS.items()}


# ================================================================== planning
def sample_line(vol, a_mm, b_mm, origin, spacing, step=1.0, order=0):
    from scipy import ndimage as ndi
    n = max(2, int(np.linalg.norm(b_mm - a_mm) / step) + 1)
    pts = a_mm[None] + (b_mm - a_mm)[None] * np.linspace(0, 1, n)[:, None]
    q = (pts - origin) / spacing
    return ndi.map_coordinates(vol, q.T, order=order, mode="constant", cval=0), pts


def plan_corridor(head, label, func, origin, spacing, lms, target_mask, max_candidates=2500):
    """Choose a straight approach to the target that minimises brain transit and avoids eloquent cortex and ventricles."""
    P, T = head["P"], head["T"]
    tidx = np.argwhere(target_mask)
    if len(tidx) == 0:
        return None
    tc = tidx.mean(0) * spacing + origin
    # candidate scalp points above the nasion/tragus line
    up = np.array([0, 0, 1.0])
    pl_a = lms["nasion"] + up * 15
    pl_b = lms["tragus_l"] + up * 15
    pl_c = lms["tragus_r"] + up * 15
    nrm = np.cross(pl_b - pl_a, pl_c - pl_a)
    if nrm[2] < 0:
        nrm = -nrm
    nrm /= np.linalg.norm(nrm)
    above = (P - pl_a) @ nrm > 0
    vn = np.zeros_like(P)
    tri_n = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    for k in range(3):
        np.add.at(vn, T[:, k], tri_n)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9
    cand = np.where(above & (np.linalg.norm(P - tc, axis=1) < 140))[0]
    if len(cand) > max_candidates:
        cand = cand[np.linspace(0, len(cand) - 1, max_candidates).astype(int)]
    best = None
    for vi in cand:
        e = P[vi]
        d = tc - e
        L = np.linalg.norm(d)
        u = d / L
        ang = math.degrees(math.acos(np.clip(-u @ vn[vi], -1, 1)))
        if ang > 50:
            continue
        lab, _ = sample_line(label, e, tc, origin, spacing)
        fun, _ = sample_line(func, e, tc, origin, spacing)
        inbrain = lab > 0
        if not inbrain.any():
            continue
        first_brain = int(np.argmax(inbrain))
        tum = np.isin(lab, (L_NCR, L_ET, L_ED))
        first_tum = int(np.argmax(tum)) if tum.any() else len(lab) - 1
        transit = max(0, first_tum - first_brain)
        seg_l = lab[first_brain:first_tum + 1]
        seg_f = fun[first_brain:first_tum + 1]
        eloq_mm = float(np.isin(seg_f, (1, 2, 3, 4)).sum())
        vent_mm = float((seg_l == L_VENT).sum())
        cost = transit + 6.0 * eloq_mm + 20.0 * vent_mm + 0.25 * ang
        if best is None or cost < best["cost"]:
            best = {"cost": cost, "entry": e, "target": tc, "transit_mm": float(transit), "eloquent_mm": eloq_mm,
                    "ventricle_mm": vent_mm, "angle_deg": ang, "length_mm": float(L), "vertex": int(vi)}
    return best


def craniotomy_outline(head, target_mask, origin, spacing, entry, target, margin=10.0, npts=72):
    from scipy.spatial import ConvexHull
    P, T = head["P"], head["T"]
    d = target - entry
    d /= np.linalg.norm(d)
    a = np.cross(d, [0, 0, 1.0])
    if np.linalg.norm(a) < 1e-3:
        a = np.cross(d, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    pts = np.argwhere(target_mask) * spacing + origin - target
    uv = np.stack([pts @ a, pts @ b], -1)
    hull = uv[ConvexHull(uv).vertices]
    circ = np.stack([np.cos(np.linspace(0, 2 * math.pi, 24, endpoint=False)), np.sin(np.linspace(0, 2 * math.pi, 24, endpoint=False))], -1) * margin
    big = (hull[:, None, :] + circ[None]).reshape(-1, 2)
    poly = big[ConvexHull(big).vertices]
    # resample evenly by arc length
    loop = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, cum[-1], npts, endpoint=False)
    res = np.stack([np.interp(s, cum, loop[:, 0]), np.interp(s, cum, loop[:, 1])], -1)
    orig = target[None] + res[:, :1] * a + res[:, 1:] * b - d * 260.0
    t = raycast(P, T, orig, np.repeat(d[None], len(res), 0))
    ok = np.isfinite(t)
    out = orig + d[None] * t[:, None]
    area = 0.5 * abs(np.dot(res[:, 0], np.roll(res[:, 1], 1)) - np.dot(res[:, 1], np.roll(res[:, 0], 1)))
    return out[ok], float(area / 100.0), float(np.sqrt(area / math.pi) * 2)


# ================================================================== skull, dura and venous sinuses
def _block_mean(m, f):
    nx, ny, nz = (np.array(m.shape) // f) * f
    return m[:nx, :ny, :nz].reshape(nx // f, f, ny // f, f, nz // f, f).mean((1, 3, 5))


def _outer_table_mm(el, az):
    """Brain-hull to outer-table distance by region (degrees; az 90 = anterior)."""
    w_front = np.clip(1 - np.abs(az - 90) / 50, 0, 1) * np.clip((35 - el) / 25, 0, 1) * np.clip((el + 25) / 15, 0, 1)
    w_temp = np.clip(1 - np.minimum(np.abs(az), np.abs(np.abs(az) - 180)) / 40, 0, 1) * np.clip((25 - el) / 20, 0, 1)
    w_occ = np.clip(1 - np.abs(az + 90) / 50, 0, 1) * np.clip((25 - el) / 25, 0, 1)
    o = SKULL_OUTER_MM
    return (o["default"] + (o["frontal"] - o["default"]) * w_front + (o["temporal"] - o["default"]) * w_temp
            + (o["occipital"] - o["default"]) * w_occ)


def skull_model(mask, origin, spacing, head, lms, midline_x):
    """Estimate the dura, the skull vault and the main venous sinuses around a skull-stripped brain.
    The brain is closed into a hull (sulci filled); the inner table sits SKULL_GAP_MM outside it and
    the outer table follows regional distances measured on a real MRI with scalp (TCGA-DU-7294).
    The skull is kept at least 3 mm inside the fitted scalp. Works on a 2 mm grid for speed."""
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    f = 2
    pad = 12
    m = np.pad(mask, pad * f)
    org = origin - pad * f * spacing
    mb = _block_mean(m.astype(np.float32), f) > 0.4
    sp2 = spacing * f
    org2 = org + (f - 1) / 2.0 * spacing
    d_out = ndi.distance_transform_edt(~mb, sampling=sp2)
    closed = d_out <= 6.0
    hull = ndi.binary_fill_holes((ndi.distance_transform_edt(closed, sampling=sp2) > 6.0) | mb)
    dh = ndi.distance_transform_edt(~hull, sampling=sp2)
    # no skull below the foramen magnum: lowest brain level with a substantial cross-section
    area = mask.sum((0, 1))
    z_cut = origin[2] + int(np.argmax(area >= 0.25 * area.max())) * spacing[2]
    gi = np.argwhere(dh <= 14.0)
    X = gi * sp2 + org2
    c = np.asarray(head["brain_center"], float)
    rel = X - c
    r = np.linalg.norm(rel, axis=1) + 1e-6
    u = rel / r[:, None]
    el = np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1)))
    az = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
    # scalp radius per direction from the fitted head scan
    P, T = head["P"], head["T"]
    S = np.concatenate([P, P[T].mean(1)])
    sr = S - c
    sd = np.linalg.norm(sr, axis=1)
    dirs = fib_sphere(2500)
    tree = cKDTree(dirs)
    _, bi = tree.query(sr / sd[:, None])
    rh = np.full(len(dirs), np.inf)
    np.minimum.at(rh, bi, sd)
    nb = tree.query(dirs, k=8)[1]
    rh_f = np.array([np.min(rh[nb[k]]) for k in range(len(dirs))])
    _, vb = tree.query(u)
    scalp_r = rh_f[vb]
    d = dh[tuple(gi.T)]
    outer = _outer_table_mm(el, az)
    inside_scalp = r <= scalp_r - 3.0
    skull_pts = (d > SKULL_GAP_MM) & (d <= outer) & inside_scalp & (X[:, 2] >= z_cut)
    skull = np.zeros(dh.shape, bool)
    skull[tuple(gi[skull_pts].T)] = True
    dura = (dh <= SKULL_GAP_MM - 0.4)
    zi = np.arange(dh.shape[2]) * sp2[2] + org2[2]
    dura[:, :, zi < z_cut - 6] = False

    def surf(vol, sigma):
        from skimage import measure
        v = ndi.gaussian_filter(np.pad(vol.astype(np.float32), 2), sigma)
        if v.max() <= 0.5:
            return None
        vv, ff, _, _ = measure.marching_cubes(v, 0.5, step_size=1, allow_degenerate=False)
        return (vv - 2) * sp2 + org2, ff

    out = {"skull": surf(skull, 0.7), "dura": surf(dura, 0.8), "z_cut": float(z_cut)}
    # superior sagittal sinus: along the hull in the midsagittal plane, nasion level to the torcula
    ix = int(round((midline_x - org2[0]) / sp2[0]))
    sl = hull[max(0, ix - 1):ix + 2].any(0)
    edge = sl & ~ndi.binary_erosion(sl)
    pts = np.argwhere(edge) * sp2[1:] + org2[1:]
    sinus = []
    if len(pts) > 20:
        yc, zc = (np.argwhere(sl) * sp2[1:] + org2[1:]).mean(0)
        ang = np.degrees(np.arctan2(pts[:, 1] - zc, pts[:, 0] - yc))      # 0 anterior, 90 superior, 180 posterior
        rad = np.hypot(pts[:, 0] - yc, pts[:, 1] - zc)
        z_front = lms["nasion"][2] + 14.0
        z_back = (lms["tragus_l"][2] + lms["tragus_r"][2]) / 2 + 14.0
        curve = []
        for a in np.arange(-30, 215, 3.0):
            sel = np.abs(ang - a) < 2.0
            if not sel.any():
                continue
            k = np.argmax(np.where(sel, rad, -1))
            y, zz = pts[k]
            if (a < 90 and zz < z_front) or (a >= 90 and zz < z_back):
                continue
            curve.append([y, zz, a])
        if len(curve) > 8:
            cv = np.array(curve)
            n = len(cv)
            for i, (y, zz, a) in enumerate(cv):
                inward = np.array([yc - y, zc - zz])
                inward /= np.linalg.norm(inward) + 1e-9
                t = i / max(1, n - 1)
                sinus.append([float(midline_x), float(y + inward[0] * 1.0), float(zz + inward[1] * 1.0), round(2.0 + 3.0 * t, 2)])
    # transverse sinuses: from the torcula laterally along the occipital inner surface
    transverse = {}
    if sinus:
        tor = np.array(sinus[-1][:3])
        kz = int(round((tor[2] - org2[2]) / sp2[2]))
        if 0 <= kz < hull.shape[2]:
            ax = hull[:, :, kz]
            e2 = np.argwhere(ax & ~ndi.binary_erosion(ax)) * sp2[:2] + org2[:2]
            if len(e2) > 20:
                xc, yc2 = (np.argwhere(ax) * sp2[:2] + org2[:2]).mean(0)
                a2 = np.degrees(np.arctan2(e2[:, 1] - yc2, e2[:, 0] - xc))
                r2 = np.hypot(e2[:, 0] - xc, e2[:, 1] - yc2)
                for side, sgn in (("right", 1), ("left", -1)):
                    line = []
                    for k, a in enumerate(np.arange(0, 75, 4.0)):
                        aa = -90 + sgn * a if sgn > 0 else -90 - a
                        sel = np.abs(a2 - aa) < 3.0
                        if not sel.any():
                            continue
                        j = np.argmax(np.where(sel, r2, -1))
                        x, y = e2[j]
                        inward = np.array([xc - x, yc2 - y])
                        inward /= np.linalg.norm(inward) + 1e-9
                        line.append([float(x + inward[0]), float(y + inward[1]), float(tor[2] - k * 0.6), round(4.5 - k * 0.12, 2)])
                    if len(line) > 4:
                        transverse[side] = line
    out["sinus"] = {"superior_sagittal": sinus, "transverse": transverse}
    return out


# ================================================================== meshes and packaging
def mesh_from_mask(mask, sigma, step, origin, spacing, level=0.5):
    from scipy import ndimage as ndi
    from skimage import measure
    if mask.sum() < 20:
        return None
    pad = 3
    m = np.pad(mask.astype(np.float32), pad)
    f = ndi.gaussian_filter(m, sigma)
    if f.max() <= level:
        return None
    v, faces, _, _ = measure.marching_cubes(f, level, step_size=step, allow_degenerate=False)
    v = (v - pad) * spacing + origin
    return v, faces


def pack_mesh(name, kind, color, v, f, extra=None):
    off = v.min(0)
    q = np.clip(np.rint((v - off) / 0.05), 0, 65535).astype(np.uint16)
    d = {"name": name, "kind": kind, "color": color, "quant": 0.05, "offset": off.round(3).tolist(),
         "count": int(len(v)), "tris": int(len(f)),
         "pos": base64.b64encode(q.tobytes()).decode(), "idx": base64.b64encode(f.astype(np.uint32).tobytes()).decode()}
    if extra:
        d.update(extra)
    return d


def mosaic(channels, dims):
    from PIL import Image
    nx, ny, nz = dims
    cols = int(math.ceil(math.sqrt(nz * ny / nx)))
    rows = int(math.ceil(nz / cols))
    img = np.zeros((rows * ny, cols * nx, 3), np.uint8)
    for k in range(nz):
        r, cidx = divmod(k, cols)
        for ch in range(3):
            if channels[ch] is not None:
                img[r * ny:(r + 1) * ny, cidx * nx:(cidx + 1) * nx, ch] = channels[ch][:, :, k].T
    bio = io.BytesIO()
    Image.fromarray(img, "RGB").save(bio, "PNG", compress_level=6)
    return bio.getvalue(), cols, rows


def r3(v):
    return [round(float(x), 2) for x in v]


def build_case(case, status=lambda s: None):
    from scipy import ndimage as ndi
    t0 = time.time()
    out_dir = os.path.join(CASES_DIR, case["key"])
    os.makedirs(out_dir, exist_ok=True)
    status("Loading MRI")
    raw, affine, shape = {}, None, None
    for m in MOD_ORDER:
        if m in case["files"]:
            v, aff = load_ras(case["files"][m])
            if shape is None:
                shape, affine = v.shape, aff
            elif v.shape != shape:
                raise ValueError(f"{m} has shape {v.shape}, expected {shape}")
            raw[m] = v
    spacing = np.abs(np.diag(affine[:3, :3])).astype(np.float64)
    off_diag = np.abs(affine[:3, :3] - np.diag(np.diag(affine[:3, :3]))).max()
    if off_diag > 1e-3:
        raise ValueError("Oblique volumes are not supported; BraTS volumes are axis-aligned.")
    status("Brain mask and tumour")
    nz = np.zeros(shape, bool)
    for v in raw.values():
        nz |= v != 0
    mask = ndi.binary_fill_holes(largest_cc(ndi.binary_opening(nz, iterations=1)))
    z = {m: robust_z(v, mask) for m, v in raw.items()}
    seg_source, scheme = "auto", "Intensity heuristic (no segmentation file)"
    if "seg" in case["files"]:
        segv, _ = load_ras(case["files"]["seg"])
        tum, scheme = tumour_from_seg(segv)
        seg_source = "file"
    else:
        tum = None
        try:
            tum = tumour_segresnet(raw, mask)
        except Exception:
            traceback.print_exc()
        if tum is not None:
            seg_source, scheme = "segresnet", "MONAI SegResNet"
        else:
            tum = tumour_auto(z, mask)
    tum[~mask] = 0
    status("Registering brain atlas")
    atl_img = raw.get("t1", raw.get("t1ce", next(iter(raw.values()))))
    reg = None
    try:
        reg = register_atlas(atl_img, mask, ndi.binary_dilation(tum > 0, iterations=4), affine, status)
    except Exception:
        traceback.print_exc()
    if reg:
        func, atlas_vent, mni_x = atlas_to_case(reg, shape, affine, mask)
        near_mid = mask & (np.abs(mni_x) < 1.5)
        midline_x = float(np.argwhere(near_mid)[:, 0].mean()) if near_mid.any() else float(np.argwhere(mask)[:, 0].mean())
    else:
        func, atlas_vent, midline_x = np.zeros(shape, np.uint8), None, float(np.argwhere(mask)[:, 0].mean())
    status("Segmenting ventricles")
    vent = ventricles(z, mask, tum, atlas_vent, midline_x)
    label = np.zeros(shape, np.uint8)
    label[mask] = L_BRAIN
    label[vent] = L_VENT
    label[tum == L_ED] = L_ED
    label[tum == L_NCR] = L_NCR
    label[tum == L_ET] = L_ET
    # crop to brain + margin
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - 4, 0)
    hi = np.minimum(idx.max(0) + 5, np.array(shape))
    sl = tuple(slice(lo[d], hi[d]) for d in range(3))
    dims = (hi - lo).tolist()
    origin = (affine[:3, :3] @ lo + affine[:3, 3]).astype(np.float64)
    lab_c, func_c, mask_c = label[sl], func[sl], mask[sl]
    mid_c = (midline_x - lo[0]) * spacing[0] + origin[0]
    status("Fitting patient head")
    affc = np.eye(4)
    affc[:3, :3] = np.diag(spacing)
    affc[:3, 3] = origin
    head = fit_head(mask_c, affc, status)
    lms = head_landmarks(head["matrix"])
    status("Planning the craniotomy")
    plans = {}
    core = np.isin(lab_c, (L_NCR, L_ET))
    et_cc = float((lab_c == L_ET).sum() * spacing.prod() / 1000)
    core_cc = float(core.sum() * spacing.prod() / 1000)
    # enhancing / cystic lesions (HGG, metastasis): resect the core; non-enhancing (LGG-like): resect the FLAIR abnormality
    resect = core if (et_cc >= 0.3 and core_cc >= 1.0) else np.isin(lab_c, (L_NCR, L_ET, L_ED))
    cr = plan_corridor(head, lab_c, func_c, origin, spacing, lms, resect)
    if cr:
        margin = 7.0 if resect is core else 10.0
        outline, area_cm2, diam = craniotomy_outline(head, resect, origin, spacing, cr["entry"], cr["target"], margin=margin)
        plans["craniotomy"] = {"entry": r3(cr["entry"]), "target": r3(cr["target"]), "outline": [r3(p) for p in outline],
                               "area_cm2": round(area_cm2, 1), "diameter_mm": round(diam, 1), "margin_mm": margin,
                               "transit_mm": round(cr["transit_mm"], 1), "eloquent_mm": cr["eloquent_mm"], "ventricle_mm": cr["ventricle_mm"],
                               "target_region": "tumour core" if resect is core else "whole tumour (FLAIR)"}
    status("Building 3D models")
    meshes = []
    brain_m = mesh_from_mask(mask_c, 1.2, 2, origin, spacing)
    if brain_m:
        v, f = brain_m
        q = np.clip(np.rint((v - origin) / spacing).astype(int), 0, np.array(dims) - 1)
        fl = func_c[tuple(q.T)]
        # sample a little inward too (vertices sit on the boundary)
        nrm = np.zeros_like(v)
        tn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        for k in range(3):
            np.add.at(nrm, f[:, k], tn)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        for depth in (1.5, 3.0):
            q2 = np.clip(np.rint((v - nrm * depth - origin) / spacing).astype(int), 0, np.array(dims) - 1)
            f2 = func_c[tuple(q2.T)]
            fl = np.where(fl == 0, f2, fl)
        meshes.append(pack_mesh("Brain", "brain", "#c9c4c0", v, f, {"func": base64.b64encode(fl.astype(np.uint8).tobytes()).decode()}))
    for name, kind, color, m, sig in (("Oedema (FLAIR)", "ed", "#3fb68b", np.isin(lab_c, (L_NCR, L_ET, L_ED)), 1.0),
                                      ("Tumour core", "ncr", "#f2efe6", np.isin(lab_c, (L_NCR, L_ET)), 0.9),
                                      ("Enhancing tumour", "et", "#e0607e", lab_c == L_ET, 0.8),
                                      ("Lateral ventricles", "vent", "#4aa8e8", lab_c == L_VENT, 1.0)):
        r = mesh_from_mask(m, sig, 1, origin, spacing)
        if r:
            meshes.append(pack_mesh(name, kind, color, r[0], r[1]))
    status("Estimating skull, dura and sinuses")
    sk = skull_model(mask_c, origin, spacing, head, lms, mid_c)
    if sk["skull"]:
        meshes.append(pack_mesh("Skull (estimated)", "skull", "#e9dcc4", *sk["skull"]))
    if sk["dura"]:
        meshes.append(pack_mesh("Dura (estimated)", "dura", "#b9a7b0", *sk["dura"]))
    status("Writing package")
    u8 = {m: to_u8(raw[m][sl], mask_c, 0.5, 99.8 if m == "t1ce" else 99.5) for m in raw}
    pa, cols, rows = mosaic([u8.get("t1ce"), u8.get("flair"), u8.get("t2")], dims)
    pb, _, _ = mosaic([u8.get("t1"), lab_c, func_c], dims)
    open(os.path.join(out_dir, "volA.png"), "wb").write(pa)
    open(os.path.join(out_dir, "volB.png"), "wb").write(pb)
    json.dump({"meshes": meshes}, open(os.path.join(out_dir, "meshes.json"), "w"))
    vox_cc = float(spacing.prod() / 1000)
    vols = {"brain": (lab_c > 0).sum(), "oedema": (lab_c == L_ED).sum(), "core": core.sum(), "enhancing": (lab_c == L_ET).sum(), "ventricles": (lab_c == L_VENT).sum()}
    meta = {
        "version": PACKAGE_VERSION, "key": case["key"], "name": case["name"], "folder": case["dir"],
        "modalities": [m for m in MOD_ORDER if m in raw], "seg_source": seg_source, "label_scheme": scheme,
        "dims": dims, "spacing": spacing.tolist(), "origin": origin.round(3).tolist(),
        "mosaic": {"cols": cols, "rows": rows, "volA": ["t1ce", "flair", "t2"], "volB": ["t1", "label", "func"]},
        "label_codes": {str(k): v for k, v in LABEL_NAMES.items()}, "func_codes": {str(k): v for k, v in FUNC_NAMES.items()},
        "volumes_cc": {k: round(float(v) * vox_cc, 2) for k, v in vols.items()},
        "midline_x": round(float(mid_c), 2),
        "atlas": {"registered": bool(reg), "score": round(reg["score"], 3) if reg else None, "language_hemisphere": LANGUAGE_HEMISPHERE},
        "head_fit": {"matrix": np.asarray(head["matrix"]).round(5).tolist(), "scale": [round(x, 4) for x in head["scale"]],
                     "clearance_mm": {k: round(v, 1) for k, v in head["clearance"].items()}},
        "landmarks": {k: r3(v) for k, v in lms.items()},
        "anatomy": {"sinus": sk["sinus"], "skull_base_z": round(sk["z_cut"], 1),
                    "skull_model": {"gap_mm": SKULL_GAP_MM, "outer_mm": SKULL_OUTER_MM, "source": "regional distances measured on TCGA-DU-7294 with scalp"}},
        "plans": plans,
        "build_seconds": round(time.time() - t0, 1),
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    write_index()
    return meta


def bundled_cases():
    """The prepared cases in cases/ (the ones a hosted demo can show)."""
    rows = []
    for d in sorted(glob.glob(os.path.join(CASES_DIR, "*", "meta.json"))):
        try:
            m = json.load(open(d))
            rows.append({"key": m["key"], "name": m["name"], "modalities": m["modalities"], "seg_source": m["seg_source"],
                         "volumes_cc": m["volumes_cc"], "built": True})
        except Exception:
            pass
    return rows


def write_index():
    rows = []
    for d in sorted(glob.glob(os.path.join(CASES_DIR, "*", "meta.json"))):
        try:
            m = json.load(open(d))
            rows.append({"key": m["key"], "name": m["name"], "modalities": m["modalities"], "seg_source": m["seg_source"],
                         "volumes_cc": m["volumes_cc"], "built": True})
        except Exception:
            pass
    json.dump({"cases": rows}, open(os.path.join(CASES_DIR, "index.json"), "w"), indent=1)


def package_ready(key):
    p = os.path.join(CASES_DIR, key, "meta.json")
    if not os.path.isfile(p):
        return False
    try:
        return json.load(open(p)).get("version") == PACKAGE_VERSION
    except Exception:
        return False


def start_build(key):
    with BUILD_LOCK:
        st = BUILD_STATUS.get(key)
        if st and st.get("state") == "running":
            return st
        case = discover_cases().get(key)
        if not case:
            return {"state": "error", "error": "Case not found in DATA_DIR."}
        st = {"state": "running", "stage": "Queued", "started": time.time()}
        BUILD_STATUS[key] = st

    def run():
        try:
            build_case(case, status=lambda s: st.update(stage=s))
            st.update(state="done", stage="Ready")
        except Exception as e:
            traceback.print_exc()
            st.update(state="error", error=str(e))
    threading.Thread(target=run, daemon=True).start()
    return st


# ================================================================== web server
def create_app():
    import gzip
    import mimetypes
    from flask import Flask, Response, abort, jsonify, request, send_from_directory
    mimetypes.add_type("application/javascript", ".js")
    app = Flask(__name__, static_folder=None)
    long_cache = 86400 if DEMO else 0          # the demo's files never change between deploys
    gz_cache = {}                               # path -> (mtime, gzipped bytes)

    def send_file_gz(folder, name, max_age):
        """Send a text file gzipped when the browser accepts it (the case meshes shrink about 4x), else as is."""
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            abort(404)
        if name.endswith((".json", ".js", ".html")) and "gzip" in request.headers.get("Accept-Encoding", ""):
            mt = os.path.getmtime(path)
            hit = gz_cache.get(path)
            if not hit or hit[0] != mt:
                with open(path, "rb") as f:
                    hit = (mt, gzip.compress(f.read(), 6))
                gz_cache[path] = hit
            ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
            if name.endswith(".js"):
                ctype = "application/javascript"
            r = Response(hit[1], mimetype=ctype)
            r.headers["Content-Encoding"] = "gzip"
            r.headers["Vary"] = "Accept-Encoding"
            r.headers["Cache-Control"] = f"public, max-age={max_age}" if max_age else "no-cache"
            return r
        return send_from_directory(folder, name, max_age=max_age)

    @app.get("/")
    def index():
        return send_file_gz(BASE_DIR, "index.html", 0)

    @app.get("/healthz")
    def healthz():                              # for the host's health check: always cheap, never touches data
        return "ok", 200, {"Content-Type": "text/plain"}

    @app.get("/favicon.ico")
    def favicon():
        return "", 204

    @app.get("/<path:path>")
    def static_files(path):
        if not path.startswith(("vendor/", "assets/")) or ".." in path:
            abort(404)
        return send_file_gz(BASE_DIR, path, long_cache)

    @app.get("/api/health")
    def health():
        if DEMO:   # the page then runs on the bundled cases, as a static preview does
            return jsonify(ok=False, demo=True)
        return jsonify(ok=True, data_dir=DATA_DIR, data_dir_found=os.path.isdir(DATA_DIR))

    @app.get("/api/cases")
    def cases():
        if DEMO:
            abort(404)
        found = discover_cases(force=request.args.get("refresh") == "1")
        rows = []
        for k, c in sorted(found.items(), key=lambda kv: kv[1]["name"]):
            rows.append({"key": k, "name": c["name"], "modalities": [m for m in MOD_ORDER if m in c["files"]],
                         "has_seg": "seg" in c["files"], "built": package_ready(k),
                         "state": BUILD_STATUS.get(k, {}).get("state")})
        for m in json.load(open(os.path.join(CASES_DIR, "index.json"))).get("cases", []) if os.path.isfile(os.path.join(CASES_DIR, "index.json")) else []:
            if m["key"] not in found and package_ready(m["key"]):
                rows.append({"key": m["key"], "name": m["name"], "modalities": m["modalities"], "has_seg": m["seg_source"] == "file", "built": True, "bundled": True})
        return jsonify(cases=rows, data_dir=DATA_DIR, data_dir_found=os.path.isdir(DATA_DIR))

    @app.post("/api/cases/<key>/build")
    def build(key):
        if DEMO:
            return jsonify(state="error", error="The demo only has the bundled cases."), 403
        if package_ready(key) and request.args.get("force") != "1":
            return jsonify(state="done", stage="Ready")
        st = start_build(key)
        return jsonify({k: v for k, v in st.items() if k != "started"})

    @app.get("/api/cases/<key>/status")
    def status(key):
        if package_ready(key) and BUILD_STATUS.get(key, {}).get("state") != "running":
            return jsonify(state="done", stage="Ready")
        st = BUILD_STATUS.get(key) or {"state": "idle"}
        return jsonify({k: v for k, v in st.items() if k != "started"})

    @app.get("/cases/<key>/<name>")
    def case_file(key, name):
        if name not in ("meta.json", "meshes.json", "volA.png", "volB.png") or "/" in key or ".." in key:
            abort(404)
        return send_file_gz(os.path.join(CASES_DIR, key), name, long_cache)

    @app.get("/cases/index.json")
    def case_index():
        if not os.path.isfile(os.path.join(CASES_DIR, "index.json")):
            if DEMO:   # read-only: list the prepared case folders without writing the index
                return jsonify(cases=bundled_cases())
            write_index()
        return send_file_gz(CASES_DIR, "index.json", 0)

    @app.post("/api/sessions")
    def save_session():
        if DEMO:
            return jsonify(error="The demo does not save sessions. Use Copy JSON."), 403
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or data.get("app") != "CranioNav":
            return jsonify(error="The request did not contain a CranioNav session."), 400
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        name = time.strftime("%Y%m%d-%H%M%S") + "_" + case_key(str(data.get("case", "case"))) + "_" + case_key(str(data.get("workflow", "nav"))) + ".json"
        json.dump(data, open(os.path.join(SESSIONS_DIR, name), "w"), indent=1)
        return jsonify(file=name)

    return app


# Module-level app for gunicorn (`gunicorn app:app`). It needs nothing set up first: there is no database to create, and the
# bundled cases are read from cases/ on demand, so the host's first request (a HEAD /) and its health checks always succeed.
app = create_app()
if DEMO:   # one line in the host's log says what the demo will show
    _n = len(bundled_cases())
    print(f"CranioNav demo: {_n} bundled case(s) in {CASES_DIR}" if _n else
          f"CranioNav demo: WARNING - no prepared cases in {CASES_DIR}. Push the cases folder to the repository.", flush=True)


if __name__ == "__main__":
    os.makedirs(CASES_DIR, exist_ok=True)
    print(f"CranioNav running at http://{HOST}:{PORT}" + (" (demo: bundled cases only)" if DEMO else ""))
    if not DEMO:
        print(f"BraTS folder: {DATA_DIR} ({'found' if os.path.isdir(DATA_DIR) else 'NOT FOUND - edit DATA_DIR in app.py'})")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
