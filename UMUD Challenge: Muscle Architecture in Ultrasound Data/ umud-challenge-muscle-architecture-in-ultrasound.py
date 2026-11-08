# %% [code] {"execution":{"iopub.status.busy":"2026-11-08T04:07:01.001177Z","iopub.execute_input":"2026-11-08T04:07:01.001512Z","iopub.status.idle":"2026-11-08T04:07:10.363048Z","shell.execute_reply.started":"2026-11-08T04:07:01.001481Z","shell.execute_reply":"2026-11-08T04:07:10.362174Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2026-11-08T04:07:10.364768Z","iopub.execute_input":"2026-11-08T04:07:10.365270Z","iopub.status.idle":"2026-11-08T04:07:17.418085Z","shell.execute_reply.started":"2026-11-08T04:07:10.365246Z","shell.execute_reply":"2026-11-08T04:07:17.417410Z"}}
# Cell 1: Imports and settings

# Loads the tools we need and fixes the working image size. All images get shrunk to 256×384 so training is fast.
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from glob import glob
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

competition_dir = "/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data"
dev = "cuda" if torch.cuda.is_available() else "cpu"
H, W = 256, 384   # size every image is shrunk to

# %% [code] {"execution":{"iopub.status.busy":"2026-11-08T04:07:17.419259Z","iopub.execute_input":"2026-11-08T04:07:17.419764Z","iopub.status.idle":"2026-11-08T04:07:26.012237Z","shell.execute_reply.started":"2026-11-08T04:07:17.419722Z","shell.execute_reply":"2026-11-08T04:07:26.011192Z"}}
# Cell 2: Scale (pixels per cm) for the test images

# Our answers must be in millimetres, but images are in pixels. This table tells us how many pixels equal 1 cm for each test image.

def collect_metadata(pattern):
    """Determine px_per_cm and the bounding box for every image whose path matches pattern."""
    old_a = None
    nt = []
    test_df = []
    prev_dist_list = []
    for path in sorted(glob(f'{competition_dir}/{pattern}')):
        px_per_cm = -1
        l, t, r, b = -1, -1, -1, -1
        id = int(path[-8:-4])
        filetype = path[-3:]
        with Image.open(path) as img:
            a = np.asarray(img)
        if len(a.shape) == 3:
            assert a.shape[2] == 3
        height, width = a.shape[:2]

        # Determine the scale (pixels per cm)
        if filetype == 'png':
            # only in test dataset
            # variable scale (3..6 cm high)
            # images are 2.85 cm wide and have square pixels
            first_tick = np.argmax(a[:, 6].mean(axis=-1) > 50) # 117
            second_minor_tick = 150 + np.argmax(a[150:, 6].mean(axis=-1) > 50)
            second_major_tick = 150 + np.argmax(a[150:, 9].mean(axis=-1) > 50)
            last_tick = len(a) - 1 - np.argmax(a[:, 6].mean(axis=-1)[::-1] > 50)
            if (second_major_tick - first_tick) < 3 * (second_minor_tick - first_tick):
                px_per_cm = second_major_tick - first_tick # major ticks are 10 mm, minor ticks are 5 mm
            else:
                px_per_cm = second_minor_tick - first_tick # major ticks are 50 mm, minor ticks are 10 mm
            hw = a.shape[1] // 2
            w2 = np.argmin(a[:, hw:].sum(axis=(0, 2)))
            l, t, r, b = hw - w2, first_tick, hw + w2, last_tick
            assert 2.8 < (r - l) / px_per_cm < 2.9 # image width in cm
        elif a.shape == (800, 1200, 3):
            # tick marks can be left, right, and at the bottom; train images can be hflipped; pixels can be non-square
            if (a[87, 1147:1157] == 175).all():
                # variable scale (3..7 cm high), tick marks at right border
                # images in test dataset are 5.75 cm wide and have square pixels
                assert (a[87, 1147:1157] == 175).all() # first tick
                first_tick = np.argmax(a[:, 1150].mean(axis=-1) > 50) # 87
                # assert first_tick == 87, first_tick
                second_major_tick = first_tick + 20 + np.argmax(a[first_tick + 20:, 1150].mean(axis=-1) > 50)
                last_tick = len(a) - 1 - np.argmax(a[:, 1150].mean(axis=-1)[::-1] > 50)
                n_ticks = (last_tick - first_tick) / (second_major_tick - first_tick)
                nt.append(n_ticks)
                n_ticks = round(n_ticks)
                if n_ticks <= 14:
                    px_per_cm = (last_tick - first_tick) / n_ticks * 2 # <= 14 ticks -> 2 ticks per cm
                    if n_ticks == 7:
                        l, t, r, b = 142, 91, 1058, last_tick # 3.5 cm
                    elif n_ticks == 8:
                        l, t, r, b = 162, 91, 1038, last_tick # 4 cm
                    elif n_ticks == 9:
                        l, t, r, b = 210, 91, 990, last_tick # 4.5 cm
                    elif n_ticks == 10:
                        l, t, r, b = 249, 91, 951, last_tick # 5 cm
                    elif n_ticks == 11:
                        l, t, r, b = 281, 91, 919, last_tick # 5.5 cm
                    elif n_ticks == 12:
                        l, t, r, b = 307, 91, 893, last_tick # 6 cm
                    elif n_ticks == 13:
                        l, t, r, b = 330, 91, 870, last_tick # 6.5 cm
                    elif n_ticks == 14:
                        l, t, r, b = 349, 91, 851, last_tick # 7 cm
                    else:
                        assert False
                    assert 5.7 < (r - l) / px_per_cm < 5.8 # image width in cm
                elif n_ticks == 15:
                    px_per_cm = (last_tick - first_tick) / 3 # 15 ticks = 3 cm -> 5 ticks per cm
                    l, t, r, b = 142, 91, 1058, last_tick # 3 cm
                    assert 5.7 < (r - l) / px_per_cm < 5.8 # image width in cm
                else:
                    assert False
            elif  (a[42, 67:74, 0] > 115).all():
                # 42px = 0 cm, 783px = 5 cm, tick marks at left border
                px_per_cm = (783 - 42) / 5
                l, t, r, b = 170, 42, 1030, 798 # ticks left 5 cm
            else:
                assert False
        elif a.shape == (644, 1088, 3):
            # 0px = 0cm, 630.5px = 5 cm
            px_per_cm = 630.5 / 5
            l, t, r, b = 140, 0, 947, 643 # ticks left 5 cm
        elif a.shape[0] in [512, 513]:
            # fixed scale
            if a[-5, 49, 0] == 168 and a[-5, 438, 0] == 168:
                px_per_cm = (442 - 53) / 5
            elif a[-5, 52, 0] == 168 and a[-5, 441, 0] == 168:
                px_per_cm = (442 - 53) / 5
            elif a[-5, 53, 0] == 168 and a[-5, 442, 0] == 168:
                px_per_cm = (442 - 53) / 5
            else:
                assert False
            l, t, r, b = 0, 0, width, height - 10
        elif a.shape[0] == 853:
            assert len(a.shape) == 2
            if a[-5, 100] == 170 and a[-5, 934] == 170:
                px_per_cm = (934 - 100) / 5
            elif a[-5, 44] == 170 and a[-5, 879] == 170:
                px_per_cm = (879 - 44) / 5
            else:
                assert False
            l, t, r, b = 0, 0, width, height
        else:
            assert False, "Unknown format"
            
        test_df.append((id, 
                        path.split('/')[-1],
                        filetype,
                        a.shape[0], a.shape[1],
                        len(a.shape),
                        px_per_cm,
                        l, t, r, b))
        # print(f"{id:4}.{filetype}: {a.shape[0]:5} {a.shape[1]:5} {len(a.shape)} {px_per_cm:7.2f}")
    
    test_df = pd.DataFrame(test_df, columns=['id', 'image_id', 'filetype', 'height', 'width', 'dim', 'px_per_cm', 'l', 't', 'r', 'b'])
    print(f'good: {((test_df.px_per_cm > 0) & (test_df.l >= 0)).mean():.0%}')
    return test_df

test_df = collect_metadata('test_images_v2/test_set_v2/IMG_*.*')
meta = test_df.set_index('image_id')

# %% [code] {"execution":{"iopub.status.busy":"2026-11-08T04:07:26.013332Z","iopub.execute_input":"2026-11-08T04:07:26.013678Z","iopub.status.idle":"2026-11-08T04:10:23.914819Z","shell.execute_reply.started":"2026-11-08T04:07:26.013650Z","shell.execute_reply":"2026-11-08T04:10:23.914125Z"}}
# Cell 3: Load the training data

# Reads each image with its mask. 
# Masks are stored inconsistently (the foreground is sometimes 0, sometimes 255), so we use a simple rule: 
# the foreground is whichever value covers fewer pixels. 
# The last few lines draw one example so you can check that the mask sits on the bright bands.

def load(img_dir, mask_dir):
    masks = {p.stem: p for p in Path(mask_dir).iterdir()}
    X, Y = [], []
    for p in tqdm(sorted(Path(img_dir).iterdir())):
        if p.stem not in masks: continue
        x = np.array(Image.open(p).convert('L'))
        m = np.array(Image.open(masks[p.stem]))
        fg = (m == 0)
        if fg.mean() > 0.5: fg = ~fg          # foreground = the rarer value
        X.append(cv2.resize(x, (W, H), interpolation=cv2.INTER_AREA))
        Y.append(cv2.resize(fg.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) > 0)
    return np.array(X), np.array(Y, dtype=np.uint8)

R = competition_dir
Xa, Ya = load(f"{R}/apo_imgs_v1/apo_images_new_model_v1", f"{R}/apo_masks_v1/apo_masks_new_model_v1")
Xf, Yf = load(f"{R}/fasc_imgs_v1/fasc_images_new_model_v1", f"{R}/fasc_masks_v1/fasc_masks_new_model_v1")
print(Xa.shape, Xf.shape)

fig, ax = plt.subplots(1, 2, figsize=(10, 3))   # sanity check: image with mask overlaid
for a, (X, Y) in zip(ax, [(Xa, Ya), (Xf, Yf)]):
    a.imshow(X[0], cmap='gray'); a.imshow(Y[0], alpha=0.4, cmap='autumn'); a.axis('off')

# %% [code] {"execution":{"iopub.status.busy":"2026-11-08T04:10:23.915998Z","iopub.execute_input":"2026-11-08T04:10:23.916420Z","iopub.status.idle":"2026-11-08T04:10:23.924652Z","shell.execute_reply.started":"2026-11-08T04:10:23.916389Z","shell.execute_reply":"2026-11-08T04:10:23.923884Z"}}
# # Cell 4: The model (a small U-Net)

# A U-Net takes an image and outputs a same-size map saying “is this pixel part of the structure?”. 
# The same model is used twice, once per structure.

def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

class UNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.e1, s.e2, s.e3, s.b = block(1, 16), block(16, 32), block(32, 64), block(64, 128)
        s.u3, s.d3 = nn.ConvTranspose2d(128, 64, 2, 2), block(128, 64)
        s.u2, s.d2 = nn.ConvTranspose2d(64, 32, 2, 2), block(64, 32)
        s.u1, s.d1 = nn.ConvTranspose2d(32, 16, 2, 2), block(32, 16)
        s.out, s.p = nn.Conv2d(16, 1, 1), nn.MaxPool2d(2)
    def forward(s, x):
        e1 = s.e1(x); e2 = s.e2(s.p(e1)); e3 = s.e3(s.p(e2)); b = s.b(s.p(e3))
        d = s.d3(torch.cat([s.u3(b), e3], 1))
        d = s.d2(torch.cat([s.u2(d), e2], 1))
        d = s.d1(torch.cat([s.u1(d), e1], 1))
        return s.out(d)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-08T04:10:23.925763Z","iopub.execute_input":"2026-11-08T04:10:23.926203Z","iopub.status.idle":"2026-11-08T04:24:19.927928Z","shell.execute_reply.started":"2026-11-08T04:10:23.926176Z","shell.execute_reply":"2026-11-08T04:24:19.927258Z"}}
# Cell 5: Training function and training

# The structures are thin, so most pixels are background. 
# The loss function combines two parts, BCE (weighted so the rare foreground pixels matter more) and 
# Dice (which measures overlap), to stop the model from lazily predicting “nothing”. 
# We hold out 10% of the images and print the Dice score on them (1.0 is perfect).

def dice_loss(p, y):
    p = torch.sigmoid(p)
    return 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)

def train(X, Y, epochs=15):
    nv = len(X) // 10                                  # last 10% = validation
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    model = UNet().to(dev); opt = torch.optim.Adam(model.parameters(), 1e-3)
    prep = lambda i: (Xt[i].float().div(255).unsqueeze(1).to(dev), Yt[i].float().unsqueeze(1).to(dev))
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(X) - nv)
        for i in range(0, len(perm), 16):
            x, y = prep(perm[i:i + 16]); p = model(x)
            loss = F.binary_cross_entropy_with_logits(p, y, pos_weight=torch.tensor(10., device=dev)) + dice_loss(p, y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            x, y = prep(torch.arange(len(X) - nv, len(X))[:64])
            pr = (torch.sigmoid(model(x)) > 0.5).float()
            print(f"epoch {ep+1}: val dice {(2*(pr*y).sum()/(pr.sum()+y.sum()+1)).item():.3f}")
    return model

apo_model = train(Xa, Ya)
fasc_model = train(Xf, Yf)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-08T04:25:32.087395Z","iopub.execute_input":"2026-11-08T04:25:32.087808Z","iopub.status.idle":"2026-11-08T04:25:32.098349Z","shell.execute_reply.started":"2026-11-08T04:25:32.087781Z","shell.execute_reply":"2026-11-08T04:25:32.097410Z"}}
# Cell 6: Turning masks into measurements

# This is the geometry step, the same thing an expert does by hand:

# MT: in each image column, the mask has an upper band and a lower band. The gap between them is the thickness. We take the median over all columns.
# PA: fit a straight line through each fascicle, then take its angle relative to the deep aponeurosis line.
# FL: a simple first version, MT / sin(PA). We’ll improve this later.

def predict(model, gray):
    x = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
    x = torch.tensor(x).float().div(255)[None, None].to(dev)
    with torch.no_grad():
        return (torch.sigmoid(model(x))[0, 0] > 0.5).cpu().numpy()

def apo_lines(mask):                       # find upper and lower band in every column
    sup, deep = [], []
    for x in range(W):
        ys = np.where(mask[:, x])[0]
        if len(ys) < 2: continue
        gap = np.argmax(np.diff(ys))
        if np.diff(ys)[gap] < 10: continue   # only one band here
        sup.append((x, ys[:gap + 1].mean())); deep.append((x, ys[gap + 1:].mean()))
    return np.array(sup), np.array(deep)

def measure(apo, fasc, ow, oh, px_per_cm):
    sx, sy = ow / W, oh / H                # convert small-image pixels back to original pixels
    sup, deep = apo_lines(apo)
    if len(sup) < 10: return 20, 100, 25   # fallback if nothing was found
    mt = np.median(deep[:, 1] - sup[:, 1]) * sy * 10 / px_per_cm      # in mm
    apo_ang = np.degrees(np.arctan(np.polyfit(deep[:, 0] * sx, deep[:, 1] * sy, 1)[0]))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fasc.astype(np.uint8))
    pas = []
    for k in range(1, n):                  # each connected blob = one fascicle piece
        if stats[k, cv2.CC_STAT_AREA] < 30: continue
        ys, xs = np.where(lab == k)
        pts = np.column_stack([xs * sx, ys * sy]).astype(np.float32)
        vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        a = abs(np.degrees(np.arctan2(vy, vx)) - apo_ang) % 180
        pas.append(min(a, 180 - a))
    pas = [a for a in pas if 5 <= a <= 45]
    pa = np.median(pas) if pas else 20
    return pa, mt / np.sin(np.radians(pa)), mt

# %% [code] {"execution":{"iopub.status.busy":"2026-11-08T04:25:55.125862Z","iopub.execute_input":"2026-11-08T04:25:55.126287Z","iopub.status.idle":"2026-11-08T04:26:10.345324Z","shell.execute_reply.started":"2026-11-08T04:25:55.126260Z","shell.execute_reply":"2026-11-08T04:26:10.344385Z"}}
# Cell 7: Predict the test set and write the submission

# Loops over all 309 test images, runs both models, measures, clips to the allowed ranges, and saves the file.

rows = []
for name in tqdm(meta.index):
    g = np.array(Image.open(f"{R}/test_images_v2/test_set_v2/{name}").convert('L')); oh, ow = g.shape
    pa, fl, mt = measure(predict(apo_model, g), predict(fasc_model, g), ow, oh, meta.px_per_cm[name])
    rows.append((name, np.clip(pa, 5, 45), np.clip(fl, 30, 200), np.clip(mt, 10, 50)))
sub = pd.DataFrame(rows, columns=['image_id', 'pa_deg', 'fl_mm', 'mt_mm'])
sub.to_csv('submission.csv', index=False)
sub.describe()