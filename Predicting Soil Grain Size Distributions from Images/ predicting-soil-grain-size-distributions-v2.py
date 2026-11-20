# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-11-20T03:20:50.045685Z","iopub.execute_input":"2026-11-20T03:20:50.046476Z","iopub.status.idle":"2026-11-20T03:20:51.647095Z","shell.execute_reply.started":"2026-11-20T03:20:50.046440Z","shell.execute_reply":"2026-11-20T03:20:51.646168Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

# %% [code] {"execution":{"iopub.status.busy":"2026-11-20T03:20:51.648678Z","iopub.execute_input":"2026-11-20T03:20:51.649190Z","iopub.status.idle":"2026-11-20T03:20:53.188438Z","shell.execute_reply.started":"2026-11-20T03:20:51.649160Z","shell.execute_reply":"2026-11-20T03:20:53.187511Z"}}
# Cell 1: Setup, labels, and the metric

import os, glob, re, time
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
import warnings; warnings.filterwarnings("ignore")

BASE = "/kaggle/input/competitions/soil-grain-size-from-photos"

lab = pd.read_csv(f"{BASE}/Training_labels_updated.csv")
sub = pd.read_csv(f"{BASE}/sample_submission.csv")
COLS = list(sub.columns[1:])            # the 11 column names, in the right order
lab.columns = ["sample_id"] + COLS
Y = lab[COLS].values.astype(float)

SUPPORT = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
W = np.diff(np.log10(SUPPORT))          # width of each interval on the log axis

def emd(Y_true, Y_pred):
    """Competition metric, one score per sample (lower is better)."""
    return (np.abs(Y_true - Y_pred)[:, :-1] * W).sum(axis=1)

uniform = np.tile(np.cumsum([100/11]*11), (len(Y), 1))
mean_loo = np.array([np.delete(Y, i, axis=0).mean(axis=0) for i in range(len(Y))])
print("Uniform baseline :", emd(Y, uniform).mean().round(1), "(leaderboard says ~102)")
print("Mean-curve (LOO) :", emd(Y, mean_loo).mean().round(1), "(leaderboard says ~90)")

# %% [code] {"execution":{"iopub.status.busy":"2026-11-20T03:21:16.943025Z","iopub.execute_input":"2026-11-20T03:21:16.943796Z","iopub.status.idle":"2026-11-20T03:21:18.972659Z","shell.execute_reply.started":"2026-11-20T03:21:16.943763Z","shell.execute_reply":"2026-11-20T03:21:18.971830Z"}}
# Cell 2: Link photos to samples, and work out the real pixels-per-mm
def clean(text):
    text = text.lower()
    for a, b in [("ü","ue"), ("ö","oe"), ("ä","ae"), ("ß","ss")]:
        text = text.replace(a, b)
    return re.sub(r"[^a-z0-9]", "", text)

def find_sample(path, ids):
    name = clean(os.path.basename(path))
    for s in sorted(ids, key=len, reverse=True):
        if clean(s) in name:
            return s
    return None

all_imgs = [p for p in glob.glob(f"{BASE}/**/*", recursive=True) if p.lower().endswith((".jpg", ".jpeg"))]
train_df = pd.DataFrame({"path": sorted(p for p in all_imgs if "Training" in p.replace(BASE, ""))})
test_df  = pd.DataFrame({"path": sorted(p for p in all_imgs if "Test" in p.replace(BASE, ""))})

train_ids = lab["sample_id"].astype(str).tolist()
test_ids  = sub["sample_id"].astype(str).tolist()
train_df["sample_id"] = train_df["path"].apply(lambda p: find_sample(p, train_ids))
test_df["sample_id"]  = test_df["path"].apply(lambda p: find_sample(p, test_ids))

# camera name = the text in the file name before the sample ID
train_df["camera"] = [clean(os.path.basename(p).split(s)[0]) for p, s in zip(train_df["path"], train_df["sample_id"])]
test_df["camera"]  = [clean(os.path.basename(p).split("HPC")[0]) for p in test_df["path"]]

# (pixels per mm, image width) as listed for each camera at FULL resolution
PPM_NATIVE = {
    "motorolaedge":         (11.492, 4000),
    "samsunga52":           (26.33,  9248),
    "motorolaedge60fusion": (12.465, 4096),
    "iphone14":             (13.942, 4032),
    "iphone16":             (19.525, 5712),
}

def real_ppm(path, camera):
    w, h = Image.open(path).size              # the ACTUAL size of this file
    ppm_listed, width_listed = PPM_NATIVE[camera]
    return ppm_listed * max(w, h) / width_listed

for df in (train_df, test_df):
    df["ppm_real"] = [real_ppm(p, c) for p, c in zip(df["path"], df["camera"])]
    df["long_side"] = [max(Image.open(p).size) for p in df["path"]]

print("Unmatched train:", train_df.sample_id.isna().sum(), "| Unmatched test:", test_df.sample_id.isna().sum())
print("Test IDs without photos:", [s for s in test_ids if s not in set(test_df.sample_id)])
print(pd.concat([train_df.assign(split="train"), test_df.assign(split="test")])
        .groupby(["split", "camera"])[["long_side", "ppm_real"]].mean().round(1))

# %% [code] {"execution":{"iopub.status.busy":"2026-11-20T03:22:31.122790Z","iopub.execute_input":"2026-11-20T03:22:31.123800Z","iopub.status.idle":"2026-11-20T03:22:37.297746Z","shell.execute_reply.started":"2026-11-20T03:22:31.123757Z","shell.execute_reply":"2026-11-20T03:22:37.296480Z"}}
# Cell 3: Crop the soil tray and look at the result
PPM_COMMON = 4.5     # every photo is brought to this resolution (px per mm)

def crop_tray(img):
    """Find the soil tray by colour (starting from the image centre) and cut off the frame."""
    H, Wd = img.shape[:2]
    s = 400 / max(H, Wd)
    small = cv2.GaussianBlur(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (0, 0), 3)
    lab_img = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab_img.shape[:2]
    center = lab_img[int(.3*h):int(.7*h), int(.3*w):int(.7*w)].reshape(-1, 3)
    med = np.median(center, axis=0)
    dist = np.linalg.norm(lab_img - med, axis=2)
    thr = max(np.percentile(np.linalg.norm(center - med, axis=1), 95) * 1.6, 12)
    soil = (dist < thr) & (lab_img[..., 0] > 0.5 * med[0])
    rows, cols = soil.mean(axis=1), soil.mean(axis=0)

    def expand(v, start):
        a = b = start
        while a > 0 and v[a-1] > 0.45: a -= 1
        while b < len(v)-1 and v[b+1] > 0.45: b += 1
        return a, b

    y0, y1 = expand(rows, h // 2)
    x0, x1 = expand(cols, w // 2)
    y0, y1, x0, x1 = int(y0/s), int((y1+1)/s), int(x0/s), int((x1+1)/s)
    hh, ww = y1 - y0, x1 - x0
    return img[y0+int(.03*hh):y1-int(.03*hh), x0+int(.03*ww):x1-int(.03*ww)]   # 3% safety margin

def load_crop(path, camera):
    """Open photo, cut to the tray, resize so that 1 mm = 4.5 pixels."""
    im = np.array(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
    ppm = real_ppm(path, camera)
    crop = crop_tray(im)
    if crop.shape[0] * crop.shape[1] < 0.25 * im.shape[0] * im.shape[1]:    # crop failed -> centre 80%
        h, w = im.shape[:2]
        crop = im[int(.1*h):int(.9*h), int(.1*w):int(.9*w)]
    scale = PPM_COMMON / ppm
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=interp)

# Visual check: one photo per camera (top = original, bottom = after cropping)
examples = pd.concat([train_df.groupby("camera").head(1), test_df.groupby("camera").head(1)])
fig, axes = plt.subplots(2, len(examples), figsize=(3.2 * len(examples), 6))
for i, (_, r) in enumerate(examples.iterrows()):
    im = np.array(ImageOps.exif_transpose(Image.open(r["path"])).convert("RGB"))
    axes[0, i].imshow(im);                         axes[0, i].set_title(r["camera"], fontsize=8)
    axes[1, i].imshow(load_crop(r["path"], r["camera"]))
    axes[0, i].axis("off"); axes[1, i].axis("off")
plt.tight_layout(); plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2026-11-20T03:23:01.338959Z","iopub.execute_input":"2026-11-20T03:23:01.339353Z","iopub.status.idle":"2026-11-20T03:25:01.963936Z","shell.execute_reply.started":"2026-11-20T03:23:01.339326Z","shell.execute_reply":"2026-11-20T03:25:01.962988Z"}}
# Cell 4: Texture features measured in millimetres (the slow cell)
BANDS_MM = [0.25, 0.5, 1, 2, 4, 8, 16, 32]    # texture "scales" in millimetres
WIN_MM   = [1, 2, 4, 8, 16]

def gray_norm(img):
    """Grey image with uneven lighting removed."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return g / (cv2.GaussianBlur(g, (0, 0), 60 * PPM_COMMON / 4) + 1e-3)

def band_features(gn):
    """How much texture there is at each size (a small grain gives fine texture, a big one coarse)."""
    out, prev = {}, cv2.GaussianBlur(gn, (0, 0), 0.125 * PPM_COMMON)
    for s in BANDS_MM:
        cur = cv2.GaussianBlur(gn, (0, 0), s * PPM_COMMON)
        out[f"dog_{s}"] = float(np.sqrt(np.mean((prev - cur) ** 2)))
        prev = cur
    total = sum(out.values()) + 1e-9
    for s in BANDS_MM:
        out[f"dogfrac_{s}"] = out[f"dog_{s}"] / total
    w = np.array([out[f"dog_{s}"] for s in BANDS_MM])
    out["dog_centroid_log2mm"] = float((w * np.log2(BANDS_MM)).sum() / w.sum())   # "typical texture size"
    return out

def local_std_features(gn):
    out = {}
    for wmm in WIN_MM:
        k = int(wmm * PPM_COMMON) | 1
        mu, mu2 = cv2.blur(gn, (k, k)), cv2.blur(gn * gn, (k, k))
        out[f"lstd_{wmm}"] = float(np.sqrt(np.maximum(mu2 - mu * mu, 0)).mean())
    return out

def gradient_features(gn):
    mag = np.hypot(cv2.Sobel(gn, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gn, cv2.CV_32F, 0, 1, ksize=3))
    return {"grad_mean": float(mag.mean()), "grad_p50": float(np.median(mag)),
            "grad_p90": float(np.percentile(mag, 90)), "grad_p99": float(np.percentile(mag, 99)),
            "grad_cv": float(mag.std() / (mag.mean() + 1e-9))}

def image_features(img):
    gn = gray_norm(img)
    f = {}
    f.update(band_features(gn)); f.update(local_std_features(gn)); f.update(gradient_features(gn))
    return f

def make_tiles(img, tile_mm=70, max_tiles=6):
    """Cut the crop into up to 6 square tiles of 70 mm (gives the model more training rows)."""
    t = int(tile_mm * PPM_COMMON)
    h, w = img.shape[:2]
    if h < t or w < t:
        return [img]
    ys = np.linspace(0, h - t, max(1, h // t)).astype(int)
    xs = np.linspace(0, w - t, max(1, w // t)).astype(int)
    out = [img[y:y+t, x:x+t] for y in ys for x in xs]
    if len(out) > max_tiles:
        out = [out[i] for i in np.linspace(0, len(out) - 1, max_tiles).astype(int)]
    return out

def process_photo(path, camera, sample_id, split):
    crop = load_crop(path, camera)
    rows = [dict(sample_id=sample_id, split=split, tile=-1, **image_features(crop))]   # whole crop
    for i, t in enumerate(make_tiles(crop)):
        rows.append(dict(sample_id=sample_id, split=split, tile=i, **image_features(t)))
    return rows

jobs = [(r.path, r.camera, r.sample_id, "train") for r in train_df.itertuples()] + \
       [(r.path, r.camera, r.sample_id, "test")  for r in test_df.itertuples()]

t0 = time.time()
results = Parallel(n_jobs=-1)(delayed(process_photo)(*j) for j in jobs)
F = pd.DataFrame([row for rows in results for row in rows])
feat_cols = [c for c in F.columns if c not in ("sample_id", "split", "tile")]
print(F.shape, len(feat_cols), "features |", round((time.time() - t0) / 60, 1), "min")

F.to_csv("/kaggle/working/features.csv", index=False)   # saved so we can reload it instead of recomputing

# %% [code] {"execution":{"iopub.status.busy":"2026-11-20T03:27:14.926120Z","iopub.execute_input":"2026-11-20T03:27:14.926981Z","iopub.status.idle":"2026-11-20T03:27:18.419714Z","shell.execute_reply.started":"2026-11-20T03:27:14.926950Z","shell.execute_reply":"2026-11-20T03:27:18.418928Z"}}
# Cell 5: Model and honest cross-validation
def to_logit(Yp):
    p = np.clip(Yp / 100, 0.005, 0.995)
    return np.log(p / (1 - p))

def from_logit(Z):
    return 100 / (1 + np.exp(-Z))

def fix_curves(P):
    """Make every curve valid: inside 0-100, never going down, exactly 100 at 200 mm."""
    P = np.clip(P, 0, 100)
    out = np.empty_like(P)
    for i, p in enumerate(P):
        out[i] = IsotonicRegression(y_min=0, y_max=100).fit_transform(np.arange(11), p)
    out[:, -1] = 100
    return out

class CurveModel:
    """Curves -> logit -> 3 main 'shape numbers' (PCA) -> one Ridge model per number."""
    def __init__(self, alpha=30, n_pc=3):
        self.alpha, self.n_pc = alpha, n_pc
    def fit(self, X, Yc):
        self.sc = StandardScaler().fit(X)
        Z = to_logit(Yc)
        self.pca = PCA(self.n_pc).fit(Z)
        T = self.pca.transform(Z)
        Xs = self.sc.transform(X)
        self.models = [Ridge(alpha=self.alpha).fit(Xs, T[:, j]) for j in range(self.n_pc)]
        return self
    def predict(self, X):
        Xs = self.sc.transform(X)
        T = np.column_stack([m.predict(Xs) for m in self.models])
        return from_logit(self.pca.inverse_transform(T))

tr = F[F.split == "train"].merge(lab, on="sample_id")

def loso(alpha):
    """Leave-one-sample-out: hold out ALL photos/tiles of one soil, train on the rest."""
    errs, truth_all, pred_all = {}, [], []
    for s in sorted(tr.sample_id.unique()):
        a, b = tr[tr.sample_id != s], tr[tr.sample_id == s]
        m = CurveModel(alpha=alpha).fit(a[feat_cols].values, a[COLS].values)
        p = fix_curves(m.predict(b[feat_cols].values)).mean(axis=0, keepdims=True)
        p = fix_curves(p)
        errs[s] = emd(b[COLS].values[:1], p)[0]
    return pd.Series(errs)

for alpha in [3, 10, 30, 100, 300]:
    e = loso(alpha)
    print(f"alpha={alpha:>4}  LOSO EMD = {e.mean():.1f}   (median {e.median():.1f})")
print("Mean-curve baseline: 89.9")

# %% [code] {"execution":{"iopub.status.busy":"2026-11-20T03:28:19.121806Z","iopub.execute_input":"2026-11-20T03:28:19.122149Z","iopub.status.idle":"2026-11-20T03:28:19.380658Z","shell.execute_reply.started":"2026-11-20T03:28:19.122121Z","shell.execute_reply":"2026-11-20T03:28:19.379752Z"}}
# Cell 6: Predict test, handle the coarse Münster soil, write the submission
BEST_ALPHA = 30      # change this to the best alpha from Cell 5

te = F[F.split == "test"]

# --- main model ---
main = CurveModel(alpha=BEST_ALPHA).fit(tr[feat_cols].values, tr[COLS].values)
P_main = pd.DataFrame(fix_curves(main.predict(te[feat_cols].values)),
                      index=te.sample_id.values, columns=COLS).groupby(level=0).mean()
P_main = pd.DataFrame(fix_curves(P_main.values), index=P_main.index, columns=COLS)

# --- "extrapolation" model for soils coarser than anything in training ---
LEVELS = np.array([2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98])
LOGD = np.log10(SUPPORT)
LO, HI = np.log10(0.0005), np.log10(200)

def curve_to_quantiles(y):      # curve -> "grain size at 2%, 5%, ... 98% passing" (log scale)
    yy = np.maximum.accumulate(np.concatenate([[0], y]) + np.arange(12) * 1e-6)
    return np.interp(LEVELS, yy, np.concatenate([[LO], LOGD]))

def quantiles_to_curve(q):      # and back
    xx = np.maximum.accumulate(np.concatenate([[LO], np.sort(q), [HI]]) + np.arange(len(q) + 2) * 1e-6)
    return np.interp(LOGD, xx, np.concatenate([[0], LEVELS, [100]]))

class QuantileModel:
    """Uses ONE feature (typical texture size) and extends the trend linearly beyond the training range."""
    def fit(self, X, Yc):
        self.sc = StandardScaler().fit(X)
        Q = np.array([curve_to_quantiles(y) for y in Yc])
        self.models = [Ridge(alpha=0.1).fit(self.sc.transform(X), Q[:, j]) for j in range(Q.shape[1])]
        return self
    def predict(self, X):
        Q = np.column_stack([m.predict(self.sc.transform(X)) for m in self.models])
        return np.array([quantiles_to_curve(q) for q in Q])

one = ["dog_centroid_log2mm"]
trf, tef = tr[tr.tile == -1], te[te.tile == -1]
anchor = QuantileModel().fit(trf[one].values, trf[COLS].values)
P_anc = pd.DataFrame(fix_curves(anchor.predict(tef[one].values)),
                     index=tef.sample_id.values, columns=COLS).groupby(level=0).mean()
P_anc = pd.DataFrame(fix_curves(P_anc.values), index=P_anc.index, columns=COLS)

# the extrapolation model only gets weight when a test soil is beyond the training range
c_max  = trf.groupby("sample_id")[one[0]].mean().max()
c_test = tef.groupby("sample_id")[one[0]].mean()
w_anchor = ((c_test - c_max) / 0.5).clip(0, 1)
print(pd.DataFrame({"texture_size": c_test.round(2), "train_max": round(c_max, 2), "extrapolation_weight": w_anchor.round(2)}).to_string())

P = P_main.mul(1 - w_anchor, axis=0) + P_anc.mul(w_anchor, axis=0)
P = pd.DataFrame(fix_curves(P.values), index=P.index, columns=COLS)

# --- write the submission in the same order as sample_submission ---
out = P.loc[sub.sample_id].reset_index().rename(columns={"index": "sample_id"})
v = out[COLS].values
assert (v >= 0).all() and (v <= 100).all() and (np.diff(v, axis=1) >= -1e-9).all() and (v[:, -1] == 100).all()
out.to_csv("submission.csv", index=False)
print(out.set_index("sample_id").round(0).astype(int).to_string())