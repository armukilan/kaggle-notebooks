# %% [code] {"execution":{"iopub.status.busy":"2026-11-16T03:33:58.377919Z","iopub.execute_input":"2026-11-16T03:33:58.378281Z","iopub.status.idle":"2026-11-16T03:34:00.864202Z","shell.execute_reply.started":"2026-11-16T03:33:58.378244Z","shell.execute_reply":"2026-11-16T03:34:00.863258Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2026-11-16T03:34:00.866288Z","iopub.execute_input":"2026-11-16T03:34:00.867351Z","iopub.status.idle":"2026-11-16T03:34:09.908702Z","shell.execute_reply.started":"2026-11-16T03:34:00.867312Z","shell.execute_reply":"2026-11-16T03:34:09.907857Z"}}
# Cell 1: Load the data

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Find the competition folder automatically
DATA = next(Path('/kaggle/input').rglob('train.csv')).parent

train = pd.read_csv(DATA / 'train.csv')
test = pd.read_csv(DATA / 'test.csv')

print(train.shape, test.shape)
train.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-11-16T03:35:13.496135Z","iopub.execute_input":"2026-11-16T03:35:13.497348Z","iopub.status.idle":"2026-11-16T03:35:13.756961Z","shell.execute_reply.started":"2026-11-16T03:35:13.497305Z","shell.execute_reply":"2026-11-16T03:35:13.755922Z"}}
# Cell 2: Look at the target
print(train['is_fraud'].value_counts())
print("Fraud rate:", train['is_fraud'].mean())
print("Missing values:", train.isna().sum().sum())

# %% [code] {"execution":{"iopub.status.busy":"2026-11-16T03:36:51.734732Z","iopub.execute_input":"2026-11-16T03:36:51.735585Z","iopub.status.idle":"2026-11-16T03:36:52.737271Z","shell.execute_reply.started":"2026-11-16T03:36:51.735554Z","shell.execute_reply":"2026-11-16T03:36:52.736230Z"}}
# Cell 3: Prepare the features
TARGET = 'is_fraud'
features = [c for c in train.columns if c not in ['id', TARGET]]

# Find column types
text_cols = train[features].select_dtypes('object').columns.tolist()
bool_cols = train[features].select_dtypes('bool').columns.tolist()

X = train[features].copy()
X_test = test[features].copy()

# True/False -> 1/0
X[bool_cols] = X[bool_cols].astype(int)
X_test[bool_cols] = X_test[bool_cols].astype(int)

# Text columns -> category type (LightGBM understands these)
for c in text_cols:
    cats = sorted(train[c].unique())
    X[c] = pd.Categorical(X[c], categories=cats)
    X_test[c] = pd.Categorical(X_test[c], categories=cats)

y = train[TARGET].astype(int)
print("Text columns:", text_cols)
print("Boolean columns:", bool_cols)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-16T03:37:13.437309Z","iopub.execute_input":"2026-11-16T03:37:13.438066Z","iopub.status.idle":"2026-11-16T03:38:55.208975Z","shell.execute_reply.started":"2026-11-16T03:37:13.438037Z","shell.execute_reply":"2026-11-16T03:38:55.208069Z"}}
# Cell 4: Quick test on one split

# This is a fast check that everything works.

from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = lgb.LGBMClassifier(
    n_estimators=3000,
    learning_rate=0.05,
    num_leaves=4,          # small trees work best on this data
    min_child_samples=100,
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.5,
    verbose=-1,
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(200, verbose=False)],
)

val_pred = model.predict_proba(X_val)[:, 1]
print("Validation AUC:", roc_auc_score(y_val, val_pred))

# %% [code] {"execution":{"iopub.status.busy":"2026-11-16T03:39:26.260406Z","iopub.execute_input":"2026-11-16T03:39:26.260725Z","iopub.status.idle":"2026-11-16T03:47:05.327685Z","shell.execute_reply.started":"2026-11-16T03:39:26.260699Z","shell.execute_reply":"2026-11-16T03:47:05.326873Z"}}
# Cell 5: Train with 5-fold cross-validation and predict the test set

# This takes a few minutes.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(X))           # predictions on train rows we didn't train on
test_pred = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.05,
        num_leaves=4,
        min_child_samples=100,
        subsample=0.8, subsample_freq=1,
        colsample_bytree=0.5,
        verbose=-1,
    )
    model.fit(
        X.iloc[tr_idx], y.iloc[tr_idx],
        eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / 5
    print(f"Fold {fold+1} AUC:", roc_auc_score(y.iloc[val_idx], oof[val_idx]))

print("Overall CV AUC:", roc_auc_score(y, oof))

# %% [code] {"execution":{"iopub.status.busy":"2026-11-16T03:49:42.644446Z","iopub.execute_input":"2026-11-16T03:49:42.645433Z","iopub.status.idle":"2026-11-16T03:49:43.570194Z","shell.execute_reply.started":"2026-11-16T03:49:42.645401Z","shell.execute_reply":"2026-11-16T03:49:43.569205Z"}}
# Cell 6: Create the submission file
submission = pd.DataFrame({'id': test['id'], 'is_fraud': test_pred})
submission.to_csv('/kaggle/working/submission.csv', index=False)

print(submission.shape)   # should be (299984, 2)
submission.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-11-16T03:50:03.398130Z","iopub.execute_input":"2026-11-16T03:50:03.398526Z","iopub.status.idle":"2026-11-16T03:50:03.553079Z","shell.execute_reply.started":"2026-11-16T03:50:03.398497Z","shell.execute_reply":"2026-11-16T03:50:03.552097Z"}}
# Cell 7: Check the file
sample = pd.read_csv(DATA / 'sample_submission.csv')
print("Same columns:", list(sample.columns) == list(submission.columns))
print("Same ids:", (sample['id'].values == submission['id'].values).all())
print(submission['is_fraud'].describe())