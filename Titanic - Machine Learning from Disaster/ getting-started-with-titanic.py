# %% [code] {"execution":{"iopub.status.busy":"2025-09-22T03:47:20.259365Z","iopub.execute_input":"2025-09-22T03:47:20.259682Z","iopub.status.idle":"2025-09-22T03:47:20.621065Z","shell.execute_reply.started":"2025-09-22T03:47:20.259658Z","shell.execute_reply":"2025-09-22T03:47:20.620241Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2025-09-22T04:02:30.005495Z","iopub.execute_input":"2025-09-22T04:02:30.005883Z","iopub.status.idle":"2025-09-22T04:02:30.025481Z","shell.execute_reply.started":"2025-09-22T04:02:30.005854Z","shell.execute_reply":"2025-09-22T04:02:30.024600Z"}}
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()

# %% [code] {"execution":{"iopub.status.busy":"2025-09-22T03:47:28.271555Z","iopub.execute_input":"2025-09-22T03:47:28.271902Z","iopub.status.idle":"2025-09-22T03:47:28.284591Z","shell.execute_reply.started":"2025-09-22T03:47:28.271873Z","shell.execute_reply":"2025-09-22T03:47:28.283542Z"}}
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

# %% [code] {"execution":{"iopub.status.busy":"2025-09-22T03:47:57.288182Z","iopub.execute_input":"2025-09-22T03:47:57.289053Z","iopub.status.idle":"2025-09-22T03:47:57.295907Z","shell.execute_reply.started":"2025-09-22T03:47:57.289022Z","shell.execute_reply":"2025-09-22T03:47:57.294980Z"}}
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

# %% [code] {"execution":{"iopub.status.busy":"2025-09-22T04:03:10.227228Z","iopub.execute_input":"2025-09-22T04:03:10.227500Z","iopub.status.idle":"2025-09-22T04:03:10.441969Z","shell.execute_reply.started":"2025-09-22T04:03:10.227482Z","shell.execute_reply":"2025-09-22T04:03:10.440602Z"}}
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")