# %% [code] {"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"execution":{"iopub.status.busy":"2026-11-19T03:56:13.134699Z","iopub.execute_input":"2026-11-19T03:56:13.135156Z","iopub.status.idle":"2026-11-19T03:56:13.306225Z","shell.execute_reply.started":"2026-11-19T03:56:13.135123Z","shell.execute_reply":"2026-11-19T03:56:13.305319Z"},"jupyter":{"outputs_hidden":false}}
# Cell 1: Imports and paths
import os, glob, random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision
from torchvision import transforms

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict

random.seed(0); np.random.seed(0); torch.manual_seed(0)

# Look at everything under /kaggle/input (no guessing the folder name)
BASE = "/kaggle/input"
print("Top-level items:", os.listdir(BASE))

for root, dirs, files in os.walk(BASE):
    depth = root.replace(BASE, "").count(os.sep)
    if depth <= 3:                       # keep the printout short
        print(root, "->", len(files), "files")

# Check that we can find the key files
print(glob.glob(f"{BASE}/**/Training_labels*.csv", recursive=True))
print(glob.glob(f"{BASE}/**/sample_submission.csv", recursive=True))
print(glob.glob(f"{BASE}/**/ppm*.csv", recursive=True))

# %% [code] {"execution":{"iopub.status.busy":"2026-11-19T03:57:25.672292Z","iopub.execute_input":"2026-11-19T03:57:25.672727Z","iopub.status.idle":"2026-11-19T03:57:25.853099Z","shell.execute_reply.started":"2026-11-19T03:57:25.672693Z","shell.execute_reply":"2026-11-19T03:57:25.851925Z"},"jupyter":{"outputs_hidden":false}}
# Cell 2: Load labels and look at the data Cell 2 (updated): Load labels and look at the data
BASE = "/kaggle/input/competitions/soil-grain-size-from-photos"

train = pd.read_csv(f"{BASE}/Training_labels_updated.csv")
sub   = pd.read_csv(f"{BASE}/sample_submission.csv")
ppm   = pd.read_csv(f"{BASE}/ppm_updated.csv")

print(train.shape, sub.shape, ppm.shape)
display(train.head(3))
display(sub.head(3))
display(ppm.head())

target_cols = [c for c in train.columns if c != "sample_id"]
print(target_cols)

# Find all photos in all subfolders (any case of .jpg / .jpeg)
all_imgs = [p for p in glob.glob(f"{BASE}/**/*", recursive=True)
            if p.lower().endswith((".jpg", ".jpeg"))]

train_imgs = sorted([p for p in all_imgs if "Training" in p])
test_imgs  = sorted([p for p in all_imgs if "Test" in p])

print("Train photos:", len(train_imgs), "| Test photos:", len(test_imgs))
print([p.replace(BASE, "") for p in train_imgs[:5]])
print([p.replace(BASE, "") for p in test_imgs[:5]])

# %% [code] {"execution":{"iopub.status.busy":"2026-11-19T03:58:38.354356Z","iopub.execute_input":"2026-11-19T03:58:38.354739Z","iopub.status.idle":"2026-11-19T03:58:38.397937Z","shell.execute_reply.started":"2026-11-19T03:58:38.354712Z","shell.execute_reply":"2026-11-19T03:58:38.396841Z"},"jupyter":{"outputs_hidden":false}}
# Cell 3: Metric and the “average curve” baseline
x = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
log_x = np.log10(x)

def emd(true, pred):
    """Area between true and predicted curves on a log axis (approximation of the competition metric)."""
    diff = np.abs(true - pred)                                # (n_samples, 11)
    widths = np.diff(log_x)                                   # 10 intervals
    area = 0.5 * (diff[:, :-1] + diff[:, 1:]) * widths
    return area.sum(axis=1).mean()

Y = train[target_cols].values            # (24, 11)

# Baseline 1: the competition's flat baseline
flat = np.tile(np.linspace(100/11, 100, 11), (len(Y), 1))
print("Flat baseline:", emd(Y, flat))

# Baseline 2: predict the mean training curve (leave-one-out, so it's fair)
mean_preds = np.array([np.delete(Y, i, axis=0).mean(axis=0) for i in range(len(Y))])
print("Mean-curve baseline (LOO):", emd(Y, mean_preds))

# %% [code] {"execution":{"iopub.status.busy":"2026-11-19T03:59:13.930489Z","iopub.execute_input":"2026-11-19T03:59:13.930898Z","iopub.status.idle":"2026-11-19T03:59:13.971619Z","shell.execute_reply.started":"2026-11-19T03:59:13.930865Z","shell.execute_reply":"2026-11-19T03:59:13.970377Z"},"jupyter":{"outputs_hidden":false}}
# Cell 4 (updated): Link photos to samples
def find_sample(path, ids):
    name = os.path.basename(path)
    # try longest IDs first so a short ID can't match inside a longer one
    for s in sorted(ids, key=len, reverse=True):
        if s in name:
            return s
    return None

def find_camera(path, sample_id):
    name = os.path.basename(path)
    if sample_id is None:
        return None
    return name.split(sample_id)[0].strip("_ ")   # text before the sample ID

train_ids = train["sample_id"].astype(str).tolist()
test_ids  = sub["sample_id"].astype(str).tolist()

train_df = pd.DataFrame({"path": train_imgs})
train_df["sample_id"] = train_df["path"].apply(lambda p: find_sample(p, train_ids))
train_df["camera"] = [find_camera(p, s) for p, s in zip(train_df["path"], train_df["sample_id"])]

test_df = pd.DataFrame({"path": test_imgs})
test_df["sample_id"] = test_df["path"].apply(lambda p: find_sample(p, test_ids))
test_df["camera"] = [find_camera(p, s) for p, s in zip(test_df["path"], test_df["sample_id"])]

print("Unmatched train photos:", train_df["sample_id"].isna().sum())
print("Unmatched test photos:", test_df["sample_id"].isna().sum())

print("\nTrain photos per sample:")
print(train_df.groupby("sample_id").size().to_string())
print("\nTest photos per sample:")
print(test_df.groupby("sample_id").size().to_string())

print("\nCameras (train):")
print(train_df["camera"].value_counts().to_string())
print("\nCameras (test):")
print(test_df["camera"].value_counts().to_string())

# %% [code] {"execution":{"iopub.status.busy":"2026-11-19T04:00:34.909793Z","iopub.execute_input":"2026-11-19T04:00:34.910169Z","iopub.status.idle":"2026-11-19T04:00:34.929836Z","shell.execute_reply.started":"2026-11-19T04:00:34.910141Z","shell.execute_reply":"2026-11-19T04:00:34.928894Z"},"jupyter":{"outputs_hidden":false}}
# Cell 4b: Fix test matching
import re

def clean(text):
    # lowercase and keep only letters and digits
    return re.sub(r"[^a-z0-9]", "", text.lower())

def find_sample_v2(path, ids):
    name = clean(os.path.basename(path))
    for s in sorted(ids, key=len, reverse=True):
        if clean(s) in name:
            return s
    return None

test_df["sample_id"] = test_df["path"].apply(lambda p: find_sample_v2(p, test_ids))
train_df["sample_id"] = train_df["path"].apply(lambda p: find_sample_v2(p, train_ids))

print("Unmatched train photos:", train_df["sample_id"].isna().sum())
print("Unmatched test photos:", test_df["sample_id"].isna().sum())

print("\nUnmatched test file names:")
for p in test_df[test_df["sample_id"].isna()]["path"]:
    print(os.path.basename(p))

print("\nTest IDs with NO photos:")
print([s for s in test_ids if s not in set(test_df["sample_id"].dropna())])

# %% [code] {"execution":{"iopub.status.busy":"2026-11-19T04:01:24.219077Z","iopub.execute_input":"2026-11-19T04:01:24.219449Z","iopub.status.idle":"2026-11-19T04:01:24.239709Z","shell.execute_reply.started":"2026-11-19T04:01:24.219419Z","shell.execute_reply":"2026-11-19T04:01:24.238613Z"},"jupyter":{"outputs_hidden":false}}
# Cell 4c: Fix the Münster photos and the camera names
def clean(text):
    text = text.lower()
    for a, b in [("ü", "ue"), ("ö", "oe"), ("ä", "ae"), ("ß", "ss")]:
        text = text.replace(a, b)
    return re.sub(r"[^a-z0-9]", "", text)

train_df["sample_id"] = train_df["path"].apply(lambda p: find_sample_v2(p, train_ids))
test_df["sample_id"]  = test_df["path"].apply(lambda p: find_sample_v2(p, test_ids))

# Camera = text before the sample ID (train) or before "HPC" (test), cleaned
def get_camera(path, sample_id, is_train):
    name = os.path.basename(path)
    if is_train:
        cam = name.split(sample_id)[0]
    else:
        cam = name.split("HPC")[0]
    return clean(cam)          # e.g. "iPhone_16" and "iPhone16" both become "iphone16"

train_df["camera"] = [get_camera(p, s, True)  for p, s in zip(train_df["path"], train_df["sample_id"])]
test_df["camera"]  = [get_camera(p, s, False) for p, s in zip(test_df["path"],  test_df["sample_id"])]

print("Unmatched train photos:", train_df["sample_id"].isna().sum())
print("Unmatched test photos:", test_df["sample_id"].isna().sum())
print("Test IDs with NO photos:", [s for s in test_ids if s not in set(test_df["sample_id"].dropna())])

print("\nCameras (train):")
print(train_df["camera"].value_counts().to_string())
print("\nCameras (test):")
print(test_df["camera"].value_counts().to_string())

# %% [code] {"execution":{"iopub.status.busy":"2026-11-19T04:02:13.339121Z","iopub.execute_input":"2026-11-19T04:02:13.339498Z","iopub.status.idle":"2026-11-19T04:04:40.405512Z","shell.execute_reply.started":"2026-11-19T04:02:13.339467Z","shell.execute_reply":"2026-11-19T04:04:40.404037Z"},"jupyter":{"outputs_hidden":false}}
# Cell 5: Turn photos into numbers (features)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
model.fc = torch.nn.Identity()          # we want features, not class labels
model = model.to(device).eval()

norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

CROP = 224
N_CROPS = 6   # crops per photo, taken at full resolution so grain detail is kept

def image_features(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    crops = []
    for _ in range(N_CROPS):
        left = random.randint(0, max(0, w - CROP))
        top  = random.randint(0, max(0, h - CROP))
        crops.append(norm(img.crop((left, top, left + CROP, top + CROP))))
    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        feats = model(batch)
    return feats.mean(dim=0).cpu().numpy()

def sample_features(df, ids):
    out = {}
    for s in ids:
        paths = df[df["sample_id"] == s]["path"].tolist()
        out[s] = np.mean([image_features(p) for p in paths], axis=0)
    return np.array([out[s] for s in ids])

X_train = sample_features(train_df, train_ids)
X_test  = sample_features(test_df, test_ids)
print(X_train.shape, X_test.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-19T04:05:55.481346Z","iopub.execute_input":"2026-11-19T04:05:55.481813Z","iopub.status.idle":"2026-11-19T04:05:56.020372Z","shell.execute_reply.started":"2026-11-19T04:05:55.481777Z","shell.execute_reply":"2026-11-19T04:05:56.019382Z"},"jupyter":{"outputs_hidden":false}}
# Cell 6: Model and honest cross-validation
def postprocess(p):
    p = np.clip(p, 0, 100)
    p = np.maximum.accumulate(p, axis=1)   # keep curve non-decreasing
    p[:, -1] = 100                         # 200 mm must be exactly 100
    return p

for alpha in [10, 100, 1000, 10000, 100000]:
    m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    preds = cross_val_predict(m, X_train, Y, cv=LeaveOneOut())
    print(f"alpha={alpha:>6}  CV EMD = {emd(Y, postprocess(preds)):.2f}")

print("Mean-curve baseline:", round(emd(Y, mean_preds), 2))

# %% [code] {"execution":{"iopub.status.busy":"2026-11-19T04:07:08.320260Z","iopub.execute_input":"2026-11-19T04:07:08.320928Z","iopub.status.idle":"2026-11-19T04:07:08.375711Z","shell.execute_reply.started":"2026-11-19T04:07:08.320886Z","shell.execute_reply":"2026-11-19T04:07:08.374414Z"},"jupyter":{"outputs_hidden":false}}
# Cell 7: Train on everything and write the submission
best_alpha = 1000   # change this to the best value from Cell 6

final = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
final.fit(X_train, Y)
test_pred = postprocess(final.predict(X_test))

submission = pd.DataFrame(test_pred, columns=sub.columns[1:])
submission.insert(0, "sample_id", test_ids)
submission.to_csv("submission.csv", index=False)

display(submission.head())
print("Monotonic:", bool((np.diff(test_pred, axis=1) >= 0).all()), "| last col all 100:", bool((test_pred[:, -1] == 100).all()))