# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-08-29T03:17:01.183933Z","iopub.execute_input":"2026-08-29T03:17:01.184483Z","iopub.status.idle":"2026-08-29T03:17:03.848233Z","shell.execute_reply.started":"2026-08-29T03:17:01.184445Z","shell.execute_reply":"2026-08-29T03:17:03.847183Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2026-08-29T03:17:03.850672Z","iopub.execute_input":"2026-08-29T03:17:03.851201Z","iopub.status.idle":"2026-08-29T03:17:12.229758Z","shell.execute_reply.started":"2026-08-29T03:17:03.851168Z","shell.execute_reply":"2026-08-29T03:17:12.228713Z"}}
# Cell 1 — Imports and load data

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

train = pd.read_csv('/kaggle/input/competitions/playground-series-s6e8/train.csv')
test = pd.read_csv('/kaggle/input/competitions/playground-series-s6e8/test.csv')
sample_sub = pd.read_csv('/kaggle/input/competitions/playground-series-s6e8/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-08-29T03:17:12.230990Z","iopub.execute_input":"2026-08-29T03:17:12.231379Z","iopub.status.idle":"2026-08-29T03:17:12.456382Z","shell.execute_reply.started":"2026-08-29T03:17:12.231336Z","shell.execute_reply":"2026-08-29T03:17:12.455355Z"}}
# Cell 2 — Quick exploration (optional, but good sanity check)

train.info()
print(train['addicted_label'].value_counts(normalize=True))

categorical_cols = ['gender', 'stress_level', 'academic_work_impact']
for col in categorical_cols:
    print(col, train[col].unique())

# %% [code] {"execution":{"iopub.status.busy":"2026-08-29T03:17:12.457536Z","iopub.execute_input":"2026-08-29T03:17:12.457951Z","iopub.status.idle":"2026-08-29T03:17:12.981188Z","shell.execute_reply.started":"2026-08-29T03:17:12.457922Z","shell.execute_reply":"2026-08-29T03:17:12.980091Z"}}
# Cell 3 — Combine + encode categoricals

target = 'addicted_label'

train['is_train'] = 1
test['is_train'] = 0
test[target] = -1  # placeholder

full = pd.concat([train, test], axis=0, ignore_index=True)

# Encode gender (one-hot)
full = pd.get_dummies(full, columns=['gender'], dummy_na=True, prefix='gender')

# Encode stress_level (ordinal)
stress_map = {'Low': 0, 'Medium': 1, 'High': 2}
full['stress_level'] = full['stress_level'].map(stress_map)

# Encode academic_work_impact (binary)
impact_map = {'No': 0, 'Yes': 1}
full['academic_work_impact'] = full['academic_work_impact'].map(impact_map)

print(full.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-08-29T03:17:12.982606Z","iopub.execute_input":"2026-08-29T03:17:12.983059Z","iopub.status.idle":"2026-08-29T03:17:13.550201Z","shell.execute_reply.started":"2026-08-29T03:17:12.982983Z","shell.execute_reply":"2026-08-29T03:17:13.549113Z"}}
# Cell 4 — Feature engineering

def add_features(df):
    df = df.copy()
    
    orig_numeric = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
                     'work_study_hours', 'sleep_hours', 'notifications_per_day',
                     'app_opens_per_day', 'weekend_screen_time']
    df['n_missing'] = df[orig_numeric].isnull().sum(axis=1)
    
    eps = 1e-3
    df['social_to_screen'] = df['social_media_hours'] / (df['daily_screen_time_hours'] + eps)
    df['gaming_to_screen'] = df['gaming_hours'] / (df['daily_screen_time_hours'] + eps)
    df['screen_to_sleep'] = df['daily_screen_time_hours'] / (df['sleep_hours'] + eps)
    df['opens_to_notifs'] = df['app_opens_per_day'] / (df['notifications_per_day'] + eps)
    
    df['total_activity_hours'] = (df['social_media_hours'] + df['gaming_hours'] +
                                   df['work_study_hours'] + df['sleep_hours'])
    df['screen_time_diff'] = df['daily_screen_time_hours'] - df['total_activity_hours']
    
    df['weekend_vs_daily'] = df['weekend_screen_time'] - df['daily_screen_time_hours']
    
    return df

full = add_features(full)

# Split back into train/test
train_processed = full[full['is_train'] == 1].drop(columns=['is_train'])
test_processed = full[full['is_train'] == 0].drop(columns=['is_train', target])

feature_cols = [c for c in train_processed.columns if c not in ['id', target]]
X = train_processed[feature_cols]
y = train_processed[target]
X_test = test_processed[feature_cols]

print("Feature count:", len(feature_cols))
print("X shape:", X.shape, "X_test shape:", X_test.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-08-29T03:30:07.078085Z","iopub.execute_input":"2026-08-29T03:30:07.078487Z","iopub.status.idle":"2026-08-29T03:45:03.667872Z","shell.execute_reply.started":"2026-08-29T03:30:07.078453Z","shell.execute_reply":"2026-08-29T03:45:03.666381Z"}}
# Cell 5 — 5-fold CV training

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 63,
    'verbose': -1,
    'seed': 42
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params, train_data,
        valid_sets=[val_data],
        num_boost_round=3000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
    
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold+1} AUC: {fold_auc:.5f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall OOF AUC: {overall_auc:.5f}")

# %% [code] {"execution":{"iopub.status.busy":"2026-08-29T03:47:29.100776Z","iopub.execute_input":"2026-08-29T03:47:29.101177Z","iopub.status.idle":"2026-08-29T03:47:29.824367Z","shell.execute_reply.started":"2026-08-29T03:47:29.101147Z","shell.execute_reply":"2026-08-29T03:47:29.823079Z"}}
# Cell 6 — Build submission

submission = pd.DataFrame({
    'id': test_processed['id'].astype(int),
    'addicted_label': test_preds
})
submission.to_csv('submission.csv', index=False)

print(submission.shape)
print(sample_sub.shape)
submission.head()