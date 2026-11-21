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

# %% [code] {"execution":{"iopub.status.busy":"2026-11-21T04:58:02.691086Z","iopub.execute_input":"2026-11-21T04:58:02.691831Z","iopub.status.idle":"2026-11-21T04:58:02.701566Z","shell.execute_reply.started":"2026-11-21T04:58:02.691799Z","shell.execute_reply":"2026-11-21T04:58:02.700666Z"}}
# Cell 1: Imports and find the files
import glob
import numpy as np
import pandas as pd

BASE = "/kaggle/input/competitions/vr-calibration-free-emg-challenge-pieeg"
train_files = sorted(glob.glob(f"{BASE}/train/train_*.xlsx"))
test_files = sorted(glob.glob(f"{BASE}/test/test_*.xlsx"))
print(len(train_files), len(test_files))
print(train_files[:2])

# %% [code] {"execution":{"iopub.status.busy":"2026-11-21T04:58:25.170777Z","iopub.execute_input":"2026-11-21T04:58:25.171951Z","iopub.status.idle":"2026-11-21T04:58:40.818610Z","shell.execute_reply.started":"2026-11-21T04:58:25.171891Z","shell.execute_reply":"2026-11-21T04:58:40.817665Z"}}
# Cell 2: Look at one training file
df = pd.read_excel(train_files[0])
print(df.shape)
print(df.columns.tolist())
print(df["label"].value_counts())
print(df["label_source"].value_counts())

# %% [code] {"execution":{"iopub.status.busy":"2026-11-21T05:00:25.039802Z","iopub.execute_input":"2026-11-21T05:00:25.040769Z","iopub.status.idle":"2026-11-21T05:00:27.942403Z","shell.execute_reply.started":"2026-11-21T05:00:25.040735Z","shell.execute_reply":"2026-11-21T05:00:27.941516Z"}}
# Cell 3: Check sampling rate and guided run lengths
fs = len(df) / (df["t"].max() - df["t"].min())
print("approx sampling rate (Hz):", round(fs, 1))
print("approx window length (samples):", int(round(1.4 * fs)))

# rows per action in a test file
test0 = pd.read_excel(test_files[0])
print("test rows per action:")
print(test0.groupby("action_id").size().describe())

# guided run lengths in the training file
g = df[df["label_source"] == "guided"]
run_id = (g["label"] != g["label"].shift()).cumsum()
runs = g.groupby(run_id).agg(label=("label", "first"), rows=("label", "size"),
                             start=("t", "min"), end=("t", "max"))
runs["seconds"] = runs["end"] - runs["start"]
print(runs.head(15))
print(runs[["rows", "seconds"]].describe())

# %% [code] {"execution":{"iopub.status.busy":"2026-11-21T05:01:05.709320Z","iopub.execute_input":"2026-11-21T05:01:05.710157Z","iopub.status.idle":"2026-11-21T05:07:53.670495Z","shell.execute_reply.started":"2026-11-21T05:01:05.710123Z","shell.execute_reply":"2026-11-21T05:07:53.669520Z"}}
# Cell 4: Load all training sessions
train_dfs = []
for i, f in enumerate(train_files):
    d = pd.read_excel(f)
    d["session"] = i
    train_dfs.append(d)
print("loaded", len(train_dfs))

# %% [code] {"execution":{"iopub.status.busy":"2026-11-21T05:08:23.085851Z","iopub.execute_input":"2026-11-21T05:08:23.086889Z","iopub.status.idle":"2026-11-21T05:08:27.071761Z","shell.execute_reply.started":"2026-11-21T05:08:23.086857Z","shell.execute_reply":"2026-11-21T05:08:27.070904Z"}}
# Cell 5: Cut 1.4 s windows by time, mean-centred per window
SIGNALS = ["emg_new", "imu_new", "ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8",
           "ax","ay","az","gx","gy","gz"]

def make_windows(d, win_sec=1.4):
    g = d[d["label_source"] == "guided"]
    run_id = (g["label"] != g["label"].shift()).cumsum()
    X, y = [], []
    for _, run in g.groupby(run_id):
        t0, t1 = run["t"].min(), run["t"].max()
        start = t0
        while start + win_sec <= t1 + 0.05:
            w = run[(run["t"] >= start) & (run["t"] < start + win_sec)]
            if len(w) > 150:                      # skip near-empty windows
                v = w[SIGNALS].values
                X.append(v - v.mean(axis=0))      # mean-centre per window
                y.append(run["label"].iloc[0])
            start += win_sec
    return X, y

all_X, all_y, all_groups = [], [], []
for d in train_dfs:
    X, y = make_windows(d)
    all_X += X
    all_y += y
    all_groups += [d["session"].iloc[0]] * len(X)

print(len(all_X))
print(pd.Series(all_y).value_counts())
print("rows per window:", pd.Series([len(w) for w in all_X]).describe()[["mean", "min", "max"]])

# %% [code] {"execution":{"iopub.status.busy":"2026-11-21T05:08:53.397548Z","iopub.execute_input":"2026-11-21T05:08:53.398637Z","iopub.status.idle":"2026-11-21T05:09:11.284192Z","shell.execute_reply.started":"2026-11-21T05:08:53.398593Z","shell.execute_reply":"2026-11-21T05:09:11.283403Z"}}
# Cell 6: Check the test action IDs

# This is a separate check. 
# My later test cell concatenates all 5 test files, which would merge actions if IDs restart in each file.
for f in test_files:
    t = pd.read_excel(f)
    ids = t["action_id"].unique()
    print(f.split("/")[-1], len(ids), ids.min(), ids.max())

# %% [code] {"execution":{"iopub.status.busy":"2026-11-21T05:09:48.250464Z","iopub.execute_input":"2026-11-21T05:09:48.251543Z","iopub.status.idle":"2026-11-21T05:09:48.521494Z","shell.execute_reply.started":"2026-11-21T05:09:48.251505Z","shell.execute_reply":"2026-11-21T05:09:48.520568Z"}}
# Cell 7: Drop abnormal windows, build features
keep = [i for i, w in enumerate(all_X) if 200 <= len(w) <= 400]
print("kept", len(keep), "of", len(all_X))

def features(w):
    absw = np.abs(w)
    f = [
        absw.mean(axis=0),                       # mean absolute value
        w.std(axis=0),                           # std
        np.sqrt((w**2).mean(axis=0)),            # RMS
        np.abs(np.diff(w, axis=0)).sum(axis=0),  # waveform length
        absw.max(axis=0),                        # peak
    ]
    return np.concatenate(f)

X = np.array([features(all_X[i]) for i in keep])
y = np.array([all_y[i] for i in keep])
groups = np.array([all_groups[i] for i in keep])
print(X.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-21T05:10:09.591061Z","iopub.execute_input":"2026-11-21T05:10:09.592413Z","iopub.status.idle":"2026-11-21T05:10:16.167018Z","shell.execute_reply.started":"2026-11-21T05:10:09.592366Z","shell.execute_reply":"2026-11-21T05:10:16.165936Z"}}
# Cell 8: Validate on unseen sessions
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

scores = []
for tr, va in GroupKFold(n_splits=4).split(X, y, groups):
    model = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[va])
    scores.append(f1_score(y[va], pred, average="macro"))
print([round(s, 3) for s in scores], round(np.mean(scores), 3))

# %% [code] {"execution":{"iopub.status.busy":"2026-11-21T05:11:02.192787Z","iopub.execute_input":"2026-11-21T05:11:02.193297Z","iopub.status.idle":"2026-11-21T05:11:17.731751Z","shell.execute_reply.started":"2026-11-21T05:11:02.193270Z","shell.execute_reply":"2026-11-21T05:11:17.730599Z"}}
# Cell 9: Diagnose
from sklearn.metrics import confusion_matrix, classification_report

# 1. why were windows dropped?
lens = np.array([len(w) for w in all_X])
print("under 200 rows:", (lens < 200).sum(), "| over 400 rows:", (lens > 400).sum())

# 2. per-class results from grouped CV
pred_all = np.empty(len(y), dtype=object)
for tr, va in GroupKFold(n_splits=4).split(X, y, groups):
    m = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    m.fit(X[tr], y[tr])
    pred_all[va] = m.predict(X[va])

labels = sorted(set(y))
print(classification_report(y, pred_all, digits=3))
print(labels)
print(confusion_matrix(y, pred_all, labels=labels))

# 3. EMG-only vs IMU-only (features are 5 blocks x 16 signals)
emg_sig = [0] + list(range(2, 10))     # emg_new, ch1-ch8
imu_sig = [1] + list(range(10, 16))    # imu_new, ax-gz
def cols(sig):
    return [b * 16 + s for b in range(5) for s in sig]

for name, c in [("EMG only", cols(emg_sig)), ("IMU only", cols(imu_sig))]:
    sc = []
    for tr, va in GroupKFold(n_splits=4).split(X, y, groups):
        m = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        m.fit(X[tr][:, c], y[tr])
        sc.append(f1_score(y[va], m.predict(X[va][:, c]), average="macro"))
    print(name, round(np.mean(sc), 3))

# %% [code] {"execution":{"iopub.status.busy":"2026-11-21T05:11:56.373984Z","iopub.execute_input":"2026-11-21T05:11:56.375249Z","iopub.status.idle":"2026-11-21T05:12:13.123822Z","shell.execute_reply.started":"2026-11-21T05:11:56.375205Z","shell.execute_reply":"2026-11-21T05:12:13.122869Z"}}
# Cell 10: Scale-free EMG features
def emg_feats(w):
    e = w[:, 2:10]                                   # ch1-ch8
    rms = np.sqrt((e**2).mean(axis=0))
    share = rms / rms.sum()                          # energy share per channel
    spec = np.abs(np.fft.rfft(e, axis=0)) ** 2
    freqs = np.fft.rfftfreq(len(e), d=1/228.7)
    edges = [5, 20, 40, 70, 115]
    bands = np.array([spec[(freqs >= lo) & (freqs < hi)].sum(axis=0)
                      for lo, hi in zip(edges[:-1], edges[1:])])   # 4 x 8
    bands = bands / bands.sum(axis=0)                # band share per channel
    return np.concatenate([share, bands.ravel()])

X_emg = np.array([emg_feats(all_X[i]) for i in keep])
X_imu = X[:, cols(imu_sig)]
X_new = np.hstack([X_imu, X_emg])
print(X_emg.shape, X_new.shape)

def cv_score(Xm, name):
    sc = []
    for tr, va in GroupKFold(n_splits=4).split(Xm, y, groups):
        m = RandomForestClassifier(n_estimators=200, random_state=0,
                                   n_jobs=-1, class_weight="balanced")
        m.fit(Xm[tr], y[tr])
        sc.append(f1_score(y[va], m.predict(Xm[va]), average="macro"))
    print(name, round(np.mean(sc), 3))

cv_score(X_emg, "new EMG only")
cv_score(X_imu, "IMU only (balanced)")
cv_score(X_new, "IMU + new EMG")

# %% [code] {"execution":{"iopub.status.busy":"2026-11-21T05:14:01.354242Z","iopub.execute_input":"2026-11-21T05:14:01.354594Z","iopub.status.idle":"2026-11-21T05:14:22.795188Z","shell.execute_reply.started":"2026-11-21T05:14:01.354570Z","shell.execute_reply":"2026-11-21T05:14:22.794259Z"}}
# Cell 11: Per-session normalization
def session_norm(Xm, g):
    Xn = Xm.copy()
    for s in np.unique(g):
        i = g == s
        Xn[i] = (Xm[i] - Xm[i].mean(axis=0)) / (Xm[i].std(axis=0) + 1e-9)
    return Xn

X_emg_raw = X[:, cols(emg_sig)]          # original amplitude features
cv_score(session_norm(X_emg_raw, groups), "raw EMG feats, session-normed")
cv_score(session_norm(X_emg, groups), "scale-free EMG, session-normed")
cv_score(session_norm(X_imu, groups), "IMU, session-normed")
cv_score(session_norm(np.hstack([X_imu, X_emg_raw]), groups), "IMU + raw EMG, session-normed")

# %% [code] {"execution":{"iopub.status.busy":"2026-11-21T05:15:59.234402Z","iopub.execute_input":"2026-11-21T05:15:59.235523Z","iopub.status.idle":"2026-11-21T05:16:19.565511Z","shell.execute_reply.started":"2026-11-21T05:15:59.235489Z","shell.execute_reply":"2026-11-21T05:16:19.564439Z"}}
# Cell 12 (replacement): Train, predict, write submission
model = RandomForestClassifier(n_estimators=300, random_state=0,
                               n_jobs=-1, class_weight="balanced")
model.fit(X_imu, y)

test_df = pd.concat([pd.read_excel(f) for f in test_files], ignore_index=True)
ids, feats = [], []
for aid, g in test_df.groupby("action_id", sort=False):
    w = g.sort_values("t")[SIGNALS].values
    w = w - w.mean(axis=0)
    ids.append(aid)
    feats.append(features(w)[cols(imu_sig)])

sub = pd.DataFrame({"action_id": ids, "gesture": model.predict(np.array(feats))})
sample = pd.read_csv(f"{BASE}/sample_submission.csv")
print(sample.head(3))

sub = sub.set_index("action_id").loc[sample["action_id"]].reset_index()
sub.to_csv("submission.csv", index=False)
print(sub.shape, sub["gesture"].value_counts())