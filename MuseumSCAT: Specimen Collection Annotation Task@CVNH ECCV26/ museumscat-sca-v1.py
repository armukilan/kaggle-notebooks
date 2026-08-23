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

# %% [code] {"execution":{"iopub.status.busy":"2026-08-23T04:09:01.759381Z","iopub.execute_input":"2026-08-23T04:09:01.759639Z","iopub.status.idle":"2026-08-23T04:09:01.764096Z","shell.execute_reply.started":"2026-08-23T04:09:01.759619Z","shell.execute_reply":"2026-08-23T04:09:01.763437Z"}}
# All inputs

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

pd.set_option("display.max_colwidth", 80)
np.random.seed(42)
# any random sampling we do later (like picking validation images) is reproducible

print("Setup complete.")

# %% [code] {"execution":{"iopub.status.busy":"2026-08-23T04:09:52.595438Z","iopub.execute_input":"2026-08-23T04:09:52.595707Z","iopub.status.idle":"2026-08-23T04:09:52.616282Z","shell.execute_reply.started":"2026-08-23T04:09:52.595687Z","shell.execute_reply":"2026-08-23T04:09:52.615494Z"}}
DATA_DIR = "/kaggle/input/competitions/museumscat-specimen-collection-annotation-task"
IMAGE_DIR = os.path.join(DATA_DIR, "images")

if not os.path.isdir(DATA_DIR):
    DATA_DIR = "."
    IMAGE_DIR = "."

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

print("train.csv exists:", os.path.isfile(TRAIN_CSV))
print("test.csv exists :", os.path.isfile(TEST_CSV))

# %% [code] {"execution":{"iopub.status.busy":"2026-08-23T04:14:01.268058Z","iopub.execute_input":"2026-08-23T04:14:01.268349Z","iopub.status.idle":"2026-08-23T04:14:01.284925Z","shell.execute_reply.started":"2026-08-23T04:14:01.268279Z","shell.execute_reply":"2026-08-23T04:14:01.284243Z"}}
# Load train.csv and peek at it

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

print("train shape:", train.shape)
print("test shape :", test.shape)
train.head()
# test.head()