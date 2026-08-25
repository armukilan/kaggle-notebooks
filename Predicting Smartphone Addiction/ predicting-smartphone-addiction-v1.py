# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-08-25T03:03:34.047329Z","iopub.execute_input":"2026-08-25T03:03:34.047586Z","iopub.status.idle":"2026-08-25T03:03:36.597411Z","shell.execute_reply.started":"2026-08-25T03:03:34.047550Z","shell.execute_reply":"2026-08-25T03:03:36.596508Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2026-08-25T03:06:36.722266Z","iopub.execute_input":"2026-08-25T03:06:36.722617Z","iopub.status.idle":"2026-08-25T03:06:38.753261Z","shell.execute_reply.started":"2026-08-25T03:06:36.722587Z","shell.execute_reply":"2026-08-25T03:06:38.752433Z"}}
import numpy as np
import pandas as pd

# Load train and test sets
train = pd.read_csv('/kaggle/input/competitions/playground-series-s6e8/train.csv')
test = pd.read_csv('/kaggle/input/competitions/playground-series-s6e8/test.csv')
sample_sub = pd.read_csv('/kaggle/input/competitions/playground-series-s6e8/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)

train.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-08-25T03:07:06.396332Z","iopub.execute_input":"2026-08-25T03:07:06.396685Z","iopub.status.idle":"2026-08-25T03:07:06.541002Z","shell.execute_reply.started":"2026-08-25T03:07:06.396654Z","shell.execute_reply":"2026-08-25T03:07:06.540126Z"}}
train.info()
train['addicted_label'].value_counts(normalize=True)

# %% [code] {"execution":{"iopub.status.busy":"2026-08-25T03:09:05.963401Z","iopub.execute_input":"2026-08-25T03:09:05.963709Z","iopub.status.idle":"2026-08-25T03:09:06.212812Z","shell.execute_reply.started":"2026-08-25T03:09:05.963683Z","shell.execute_reply":"2026-08-25T03:09:06.211876Z"}}
# Step 2: Handle missing values and encode categorical columns.

# Separate features and target
target = 'addicted_label'
features = [c for c in train.columns if c not in ['id', target]]

numeric_cols = train[features].select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = train[features].select_dtypes(include=['object']).columns.tolist()

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# Check what values the categorical columns take
for col in categorical_cols:
    print(col, train[col].unique())

# %% [code] {"execution":{"iopub.status.busy":"2026-08-25T03:11:03.632580Z","iopub.execute_input":"2026-08-25T03:11:03.632942Z","iopub.status.idle":"2026-08-25T03:11:09.301877Z","shell.execute_reply.started":"2026-08-25T03:11:03.632910Z","shell.execute_reply":"2026-08-25T03:11:09.300730Z"}}
# Step 3: Preprocessing + baseline model.

# I'll use LightGBM here — it handles missing values natively (no need to impute), 
# handles categorical splits well, and is fast to iterate with. This gets you a solid baseline fast.

# gender: one-hot encode (Male/Female/Other, plus a missing indicator)
# stress_level: ordinal encode (Low=0, Medium=1, High=2) since it has a natural order
# academic_work_impact: binary encode (Yes=1, No=0)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Combine train and test for consistent encoding
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

# Split back into train/test
train_processed = full[full['is_train'] == 1].drop(columns=['is_train'])
test_processed = full[full['is_train'] == 0].drop(columns=['is_train', target])

# Define feature columns (everything except id and target)
feature_cols = [c for c in train_processed.columns if c not in ['id', target]]

X = train_processed[feature_cols]
y = train_processed[target]
X_test = test_processed[feature_cols]

print("Feature columns:", feature_cols)
print("X shape:", X.shape, "X_test shape:", X_test.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-08-25T03:12:03.617700Z","iopub.execute_input":"2026-08-25T03:12:03.618059Z","iopub.status.idle":"2026-08-25T03:13:25.953187Z","shell.execute_reply.started":"2026-08-25T03:12:03.618030Z","shell.execute_reply":"2026-08-25T03:13:25.952276Z"}}
# Step 4: Train/validation split, train a LightGBM baseline, and check ROC-AUC.

# Stratified train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Set up LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Basic parameters for binary classification with AUC as the metric
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
    'seed': 42
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

# Check validation AUC
val_preds = model.predict(X_val, num_iteration=model.best_iteration)
val_auc = roc_auc_score(y_val, val_preds)
print(f"Validation AUC: {val_auc:.5f}")

# %% [code] {"execution":{"iopub.status.busy":"2026-08-25T03:14:59.358207Z","iopub.execute_input":"2026-08-25T03:14:59.359055Z","iopub.status.idle":"2026-08-25T03:15:36.169095Z","shell.execute_reply.started":"2026-08-25T03:14:59.359019Z","shell.execute_reply":"2026-08-25T03:15:36.168239Z"}}
# Step 5: Retrain on full data and generate your first submission.

# Right now the model only saw 80% of the training data. For the actual submission, 
# retrain on 100% of it using the same number of rounds we just found (996), then predict on the test set.

# Retrain on full training data using best_iteration from validation run
final_train_data = lgb.Dataset(X, label=y)

final_model = lgb.train(
    params,
    final_train_data,
    num_boost_round=model.best_iteration  # 996 from previous step
)

# Predict on test set
test_preds = final_model.predict(X_test)

# Build submission file
submission = pd.DataFrame({
    'id': test_processed['id'].astype(int),
    'addicted_label': test_preds
})

submission.to_csv('submission.csv', index=False)
submission.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-08-25T03:16:55.286778Z","iopub.execute_input":"2026-08-25T03:16:55.287625Z","iopub.status.idle":"2026-08-25T03:16:55.292374Z","shell.execute_reply.started":"2026-08-25T03:16:55.287582Z","shell.execute_reply":"2026-08-25T03:16:55.291602Z"}}
print(submission.shape)
print(sample_sub.shape)