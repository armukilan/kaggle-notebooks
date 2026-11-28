# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code] {"jupyter":{"outputs_hidden":false}}
# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# # Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# # Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

# import kagglehub
# # kagglehub.dataset_download('<owner>/<dataset-slug>')

# %% [code] {"execution":{"iopub.status.busy":"2026-11-22T04:51:43.060808Z","iopub.execute_input":"2026-11-22T04:51:43.061071Z","iopub.status.idle":"2026-11-22T04:51:49.933966Z","shell.execute_reply.started":"2026-11-22T04:51:43.061032Z","shell.execute_reply":"2026-11-22T04:51:49.933313Z"}}
# Cell 1: Setup and find the data
import os, glob, gc, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
warnings.filterwarnings("ignore")

CANDIDATES = [
    "/kaggle/input/competitions/memecoins-pump-or-dump",
    "/kaggle/input/memecoins-pump-or-dump",
    "data",
]
DATA = next(p for p in CANDIDATES if os.path.exists(p))
print("DATA folder:", DATA)
print(sorted(os.listdir(DATA))[:15])

# %% [code] {"execution":{"iopub.status.busy":"2026-11-22T04:53:15.221531Z","iopub.execute_input":"2026-11-22T04:53:15.221817Z","iopub.status.idle":"2026-11-22T04:53:15.323380Z","shell.execute_reply.started":"2026-11-22T04:53:15.221798Z","shell.execute_reply":"2026-11-22T04:53:15.322730Z"}}
# Cell 2: Load the token lists
train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
sample = pd.read_csv(f"{DATA}/sample_submission.csv")

for df in (train, test):
    df["day"] = pd.to_datetime(df["mt"], unit="s").dt.strftime("%Y-%m-%d")

print("train tokens:", len(train), "| test tokens:", len(test))
print("train days:", train.day.min(), "->", train.day.max(), "| test day:", test.day.unique())
print("share reaching 5x:", (train.max_x >= 5).mean().round(3))
train.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-11-22T04:53:58.500863Z","iopub.execute_input":"2026-11-22T04:53:58.501117Z","iopub.status.idle":"2026-11-22T04:53:59.345116Z","shell.execute_reply.started":"2026-11-22T04:53:58.501101Z","shell.execute_reply":"2026-11-22T04:53:59.344344Z"}}
# Cell 3: Feature builder (the main work)
SOL_COLS_CURVE = ["mint", "ts_rel", "direction", "sol", "wallet", "failed"]
SOL_COLS_AMM = ["mint", "ts_rel", "direction", "sol", "new_price", "signing_wallet",
                "fee_payer", "pool_quote_token_reserves_after"]

creation = pd.read_csv(f"{DATA}/pumpfun_token_creation.csv")
CRE_COLS = [c for c in ["bundle_size", "bundled_buys", "bundled_buys_count", "dev_balance", "created_ts"]
            if c in creation.columns]
creation = creation[["mint"] + CRE_COLS].drop_duplicates("mint")
for c in CRE_COLS:
    creation[c] = pd.to_numeric(creation[c], errors="coerce")


def hhi_and_top(df, wcol):
    """concentration of SOL volume across wallets, per mint"""
    w = df.groupby(["mint", wcol])["sol"].sum()
    tot = w.groupby(level=0).transform("sum").replace(0, np.nan)
    share = w / tot
    return (share ** 2).groupby(level=0).sum(), share.groupby(level=0).max()


def day_features(day, mints):
    mints = set(mints)
    out = pd.DataFrame(index=sorted(mints))

    # ---------- bonding curve ----------
    c = pd.read_parquet(f"{DATA}/day_{day}_pumpfun_all_swaps.parquet", columns=SOL_COLS_CURVE)
    c = c[c.mint.isin(mints)]
    out["c_failed"] = c[c.failed != 0].groupby("mint").size()
    ex = c[c.failed == 0]
    buy = ex[ex.direction.str.lower().str.startswith("b")]
    sell = ex[~ex.direction.str.lower().str.startswith("b")]
    g = ex.groupby("mint")
    out["c_n"] = g.size()
    out["c_wallets"] = g["wallet"].nunique()
    out["c_sol"] = g["sol"].sum()
    out["c_age_s"] = -g["ts_rel"].min()
    out["c_sell_sol"] = sell.groupby("mint")["sol"].sum()
    out["c_buy_sol"] = buy.groupby("mint")["sol"].sum()
    out["c_last10m_share"] = ex[ex.ts_rel > -600].groupby("mint")["sol"].sum() / out["c_sol"]
    out["c_last60s_share"] = ex[ex.ts_rel > -60].groupby("mint")["sol"].sum() / out["c_sol"]
    hh, top = hhi_and_top(buy, "wallet")
    out["c_hhi"], out["c_top_share"] = hh, top
    out["c_fail_ratio"] = out["c_failed"] / (out["c_failed"] + out["c_n"])
    out["c_trades_per_wallet"] = out["c_n"] / out["c_wallets"]
    del c, ex, buy, sell

    # ---------- first 60 seconds on PumpSwap ----------
    a = pd.read_parquet(f"{DATA}/day_{day}_pumpswap_all_swaps.parquet", columns=SOL_COLS_AMM)
    a = a[a.mint.isin(mints)]
    a["is_buy"] = a.direction.str.upper().str.startswith("B")
    ab, asell = a[a.is_buy], a[~a.is_buy]
    g = a.groupby("mint")
    out["a_n"] = g.size()
    out["a_wallets"] = g["signing_wallet"].nunique()
    out["a_buy_sol"] = ab.groupby("mint")["sol"].sum()
    out["a_sell_sol"] = asell.groupby("mint")["sol"].sum()
    out["a_buy_share"] = out["a_buy_sol"] / (out["a_buy_sol"] + out["a_sell_sol"])
    out["a_max_buy"] = ab.groupby("mint")["sol"].max()
    for name, lo, hi in [("w1", 0, 15), ("w2", 15, 30), ("w3", 30, 61)]:
        part = a[(a.ts_rel >= lo) & (a.ts_rel < hi)]
        pb = part[part.is_buy].groupby("mint")["sol"].sum()
        ps = part[~part.is_buy].groupby("mint")["sol"].sum()
        out[f"a_{name}_vol"] = part.groupby("mint")["sol"].sum()
        out[f"a_{name}_net"] = pb.sub(ps, fill_value=0)
    out["a_momentum"] = out["a_w3_vol"] / (out["a_w1_vol"] + 1e-9)
    p_first, p_last = g["new_price"].first(), g["new_price"].last()
    out["a_p_last_over_first"] = p_last / p_first
    out["a_p_max_over_first"] = g["new_price"].max() / p_first
    out["a_p_last_over_max"] = p_last / g["new_price"].max()
    out["a_p_last_over_min"] = p_last / g["new_price"].min()
    out["a_depth_last"] = g["pool_quote_token_reserves_after"].last()
    out["a_depth_growth"] = out["a_depth_last"] / g["pool_quote_token_reserves_after"].first()
    diff = (a.fee_payer != a.signing_wallet)
    out["a_payer_diff_share"] = diff.groupby(a.mint).mean()
    hh, top = hhi_and_top(a, "signing_wallet")
    out["a_hhi"], out["a_top_share"] = hh, top
    del a, ab, asell

    out.index.name = "mint"
    return out.reset_index()


def build_features(tokens):
    parts = []
    for day, grp in tokens.groupby("day"):
        parts.append(day_features(day, grp.mint.tolist()))
        gc.collect()
    feats = pd.concat(parts, ignore_index=True)
    df = tokens.merge(feats, on="mint", how="left").merge(creation, on="mint", how="left")
    if "created_ts" in df.columns:
        df["token_age_s"] = df["mt"] - df["created_ts"]
        df = df.drop(columns="created_ts")
    return df

# %% [code] {"execution":{"iopub.status.busy":"2026-11-22T04:55:14.137647Z","iopub.execute_input":"2026-11-22T04:55:14.138208Z","iopub.status.idle":"2026-11-22T04:56:11.523667Z","shell.execute_reply.started":"2026-11-22T04:55:14.138189Z","shell.execute_reply":"2026-11-22T04:56:11.522460Z"}}
# Cell 4: Build the feature tables (takes a few minutes)
train_f = build_features(train)
test_f = build_features(test)
print(train_f.shape, test_f.shape)

FEATURES = [c for c in train_f.columns if c not in ["mint", "mt", "max_x", "final_x", "day"]]
print(len(FEATURES), "features")
train_f["win"] = (train_f.max_x >= 5).astype(int)
train_f[FEATURES].describe().T.head(10)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-22T04:56:29.897895Z","iopub.execute_input":"2026-11-22T04:56:29.898158Z","iopub.status.idle":"2026-11-22T04:56:38.418397Z","shell.execute_reply.started":"2026-11-22T04:56:29.898142Z","shell.execute_reply":"2026-11-22T04:56:38.417930Z"}}
# Cell 5: Model, PnL rule, and honest walk-forward test
def pnl_for(tp, max_x, final_x):
    """exact payoff rule from the Evaluation page"""
    tp = np.asarray(tp, dtype=float)
    hit = max_x >= tp
    p = np.where(hit, tp - 1 - 0.3, np.maximum(final_x - 1 - 0.3, -1))
    return np.where(tp > 0, p, 0.0)


def fit_predict(tr, te, seeds=(0, 1, 2)):
    preds = np.zeros(len(te))
    for s in seeds:
        m = lgb.LGBMClassifier(
            n_estimators=250, learning_rate=0.03, num_leaves=8,
            min_child_samples=60, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=5.0,
            random_state=s, n_jobs=4, verbose=-1)
        m.fit(tr[FEATURES], tr["win"])
        preds += m.predict_proba(te[FEATURES])[:, 1] / len(seeds)
    return preds


# walk-forward: for each of the last N days, train only on tokens whose hour was already over
VAL_DAYS = sorted(train_f.day.unique())[-10:]
oof = []
for d in VAL_DAYS:
    day_start = pd.Timestamp(d).timestamp()
    tr = train_f[train_f.mt <= day_start - 3700]
    te = train_f[train_f.day == d].copy()
    te["p"] = fit_predict(tr, te)
    oof.append(te)
    print(d, "train rows:", len(tr), "| test rows:", len(te))
oof = pd.concat(oof)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-22T04:57:47.419933Z","iopub.execute_input":"2026-11-22T04:57:47.420255Z","iopub.status.idle":"2026-11-22T04:57:47.475277Z","shell.execute_reply.started":"2026-11-22T04:57:47.420232Z","shell.execute_reply":"2026-11-22T04:57:47.474272Z"}}
# Cell 6: Pick the trading threshold and see the day-by-day table
TP = 5.0
rows = []
for thr in np.arange(0.10, 0.70, 0.02):
    tp = np.where(oof.p >= thr, TP, 0.0)
    pn = pnl_for(tp, oof.max_x.values, oof.final_x.values)
    daily = pd.Series(pn).groupby(oof.day.values).sum()
    rows.append(dict(thr=round(thr, 2), trades=int((tp > 0).sum()), total_pnl=pn.sum(),
                     days_positive=int((daily > 0).sum()), days_negative=int((daily < 0).sum())))
sweep = pd.DataFrame(rows)
print(sweep.to_string(index=False))

ok = sweep[sweep.trades >= 15]
if len(ok) and ok.total_pnl.max() > 0:
    THR = float(ok.loc[ok.total_pnl.idxmax(), "thr"])
else:
    THR = 1.1   # nothing beats zero -> skip everything (safe)
print("\nChosen threshold:", THR)

tp = np.where(oof.p >= THR, TP, 0.0)
oof["pnl"] = pnl_for(tp, oof.max_x.values, oof.final_x.values)
oof["bought"] = (tp > 0)
print(oof.groupby("day").agg(tokens=("mint", "size"), trades=("bought", "sum"), pnl=("pnl", "sum")).round(2))
print("TOTAL walk-forward PnL:", round(oof.pnl.sum(), 2))

# %% [code]
# # Cell 7: Train on everything, predict test, write the file
# test_f["p"] = fit_predict(train_f, test_f, seeds=(0, 1, 2, 3, 4))
# test_f["tp"] = np.where(test_f.p >= THR, TP, 0.0)

# sub = sample[["mint"]].merge(test_f[["mint", "tp"]], on="mint", how="left").fillna({"tp": 0.0})
# assert len(sub) == len(sample) and sub.tp.between(0, 100).all()
# sub.to_csv("/kaggle/working/submission.csv", index=False)
# print("tokens bought:", int((sub.tp > 0).sum()), "of", len(sub))
# sub.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-11-22T04:59:50.774510Z","iopub.execute_input":"2026-11-22T04:59:50.774795Z","iopub.status.idle":"2026-11-22T05:02:28.911941Z","shell.execute_reply.started":"2026-11-22T04:59:50.774777Z","shell.execute_reply":"2026-11-22T05:02:28.911184Z"}}
# Cell 8: Add the archive (takes several minutes)
archive = pd.read_csv(f"{DATA}/train_archive.csv")
archive["day"] = pd.to_datetime(archive["mt"], unit="s").dt.strftime("%Y-%m-%d")
print("archive tokens:", len(archive), "| days:", archive.day.min(), "->", archive.day.max())

archive_f = build_features(archive)
all_f = pd.concat([archive_f, train_f.drop(columns=["win"])], ignore_index=True)
all_f = all_f.sort_values("mt").reset_index(drop=True)

LEVELS = [3, 5, 8]
for k in LEVELS:
    all_f[f"win{k}"] = (all_f.max_x >= k).astype(int)
print(all_f.shape)
print({f"win{k}": round(all_f[f"win{k}"].mean(), 3) for k in LEVELS})

# %% [code] {"execution":{"iopub.status.busy":"2026-11-22T05:03:14.741605Z","iopub.execute_input":"2026-11-22T05:03:14.741904Z","iopub.status.idle":"2026-11-22T05:04:55.824973Z","shell.execute_reply.started":"2026-11-22T05:03:14.741888Z","shell.execute_reply":"2026-11-22T05:04:55.823860Z"}}
# Cell 9: Walk-forward with expected-profit rule
def fit_predict_k(tr, te, k, seeds=(0, 1, 2)):
    p = np.zeros(len(te))
    for s in seeds:
        m = lgb.LGBMClassifier(
            n_estimators=250, learning_rate=0.03, num_leaves=8,
            min_child_samples=80, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=5.0,
            random_state=s, n_jobs=4, verbose=-1)
        m.fit(tr[FEATURES], tr[f"win{k}"])
        p += m.predict_proba(te[FEATURES])[:, 1] / len(seeds)
    return p


def add_ev(tr, te):
    """one model per level -> expected profit per level for each token"""
    for k in LEVELS:
        te[f"p{k}"] = fit_predict_k(tr, te, k)
        miss = tr[tr.max_x < k]
        loss = np.maximum(miss.final_x - 1.3, -1).mean()      # typical loss when it never hits
        te[f"ev{k}"] = te[f"p{k}"] * (k - 1.3) + (1 - te[f"p{k}"]) * loss
    return te


VAL_DAYS2 = sorted(train_f.day.unique())[-14:]
oof2 = []
for d in VAL_DAYS2:
    day_start = pd.Timestamp(d).timestamp()
    tr = all_f[all_f.mt <= day_start - 3700]
    te = all_f[all_f.day == d].copy()
    oof2.append(add_ev(tr, te))
    print(d, "train rows:", len(tr), "| test rows:", len(te))
oof2 = pd.concat(oof2)

EVCOLS = [f"ev{k}" for k in LEVELS]
oof2["best_ev"] = oof2[EVCOLS].max(axis=1)
oof2["best_k"] = np.array(LEVELS)[oof2[EVCOLS].values.argmax(axis=1)]

rows = []
for margin in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5]:
    tp = np.where(oof2.best_ev >= margin, oof2.best_k, 0.0)
    pn = pnl_for(tp, oof2.max_x.values, oof2.final_x.values)
    daily = pd.Series(pn).groupby(oof2.day.values).sum()
    rows.append(dict(margin=margin, trades=int((tp > 0).sum()), total_pnl=round(pn.sum(), 2),
                     days_pos=int((daily > 0).sum()), days_neg=int((daily < 0).sum())))
print(pd.DataFrame(rows).to_string(index=False))

# %% [code] {"execution":{"iopub.status.busy":"2026-11-22T05:07:01.561383Z","iopub.execute_input":"2026-11-22T05:07:01.562386Z","iopub.status.idle":"2026-11-22T05:07:01.611957Z","shell.execute_reply.started":"2026-11-22T05:07:01.562341Z","shell.execute_reply":"2026-11-22T05:07:01.611271Z"}}
# Cell 10: Does the model rank tokens better than chance?
# from sklearn.metrics import roc_auc_score

# for k in LEVELS:
#     auc = roc_auc_score(oof2[f"win{k}"], oof2[f"p{k}"])
#     top = oof2.nlargest(int(len(oof2) * 0.05), f"p{k}")      # top 5% of tokens by confidence
#     print(f"{k}x  AUC={auc:.3f} | base rate={oof2[f'win{k}'].mean():.3f} | "
#           f"hit rate in top 5%={top[f'win{k}'].mean():.3f} | lift={top[f'win{k}'].mean()/oof2[f'win{k}'].mean():.1f}x")

# %% [code] {"execution":{"iopub.status.busy":"2026-11-22T05:07:45.905094Z","iopub.execute_input":"2026-11-22T05:07:45.905349Z","iopub.status.idle":"2026-11-22T05:08:07.953483Z","shell.execute_reply.started":"2026-11-22T05:07:45.905330Z","shell.execute_reply":"2026-11-22T05:08:07.952820Z"}}
# Cell 11: Add history features (market mood, creator track record, time of day)
creators = pd.read_csv(f"{DATA}/pumpfun_token_creation.csv", usecols=["mint", "creator"]).drop_duplicates("mint")

def add_history_features(df, hist):
    df = df.copy()
    h = hist.sort_values("mt")
    hmt = h["mt"].values
    cum = {k: np.concatenate([[0], np.cumsum(h[f"win{k}"].values)]) for k in (3, 5)}
    t = df["mt"].values - 3700                      # only tokens already resolved
    hi = np.searchsorted(hmt, t, side="right")

    # market mood: share of recently resolved tokens that hit 3x / 5x
    for W, name in [(3 * 3600, "3h"), (12 * 3600, "12h"), (48 * 3600, "48h")]:
        lo = np.searchsorted(hmt, t - W, side="left")
        n = hi - lo
        df[f"reg_n_{name}"] = n
        for k in (3, 5):
            df[f"reg_win{k}_{name}"] = np.where(n > 0, (cum[k][hi] - cum[k][lo]) / np.maximum(n, 1), np.nan)

    # time of day / weekday
    df["hour"] = (df["mt"] % 86400) / 3600
    df["dow"] = ((df["mt"] // 86400) + 4) % 7

    # creator track record (only resolved earlier tokens)
    h2 = h[["mint", "mt", "win3", "win5"]].merge(creators, on="mint", how="left")
    d2 = df[["mint", "mt"]].merge(creators, on="mint", how="left")
    cnt = np.zeros(len(d2)); r3 = np.full(len(d2), np.nan); r5 = np.full(len(d2), np.nan)
    hgroups = {c: g.sort_values("mt") for c, g in h2.groupby("creator")}
    for c, idx in d2.groupby("creator").indices.items():
        g = hgroups.get(c)
        if g is None:
            continue
        gmt = g["mt"].values
        c3 = np.concatenate([[0], np.cumsum(g["win3"].values)])
        c5 = np.concatenate([[0], np.cumsum(g["win5"].values)])
        hh = np.searchsorted(gmt, d2["mt"].values[idx] - 3700, side="right")
        cnt[idx] = hh
        r3[idx] = np.where(hh > 0, c3[hh] / np.maximum(hh, 1), np.nan)
        r5[idx] = np.where(hh > 0, c5[hh] / np.maximum(hh, 1), np.nan)
    df["cr_prev_n"], df["cr_win3_rate"], df["cr_win5_rate"] = cnt, r3, r5
    return df

BASE_FEATURES = [c for c in FEATURES]
all_f = add_history_features(all_f, all_f)
test_f = add_history_features(test_f, all_f)

NEW = [c for c in all_f.columns if c.startswith(("reg_", "cr_")) or c in ("hour", "dow")]
FEATURES = BASE_FEATURES + NEW
print(len(FEATURES), "features now | new:", NEW)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-22T05:09:21.407373Z","iopub.execute_input":"2026-11-22T05:09:21.407651Z","iopub.status.idle":"2026-11-22T05:11:33.682557Z","shell.execute_reply.started":"2026-11-22T05:09:21.407627Z","shell.execute_reply":"2026-11-22T05:11:33.681986Z"}}
# Cell 12: Re-run the walk-forward with the new features
from sklearn.metrics import roc_auc_score
def run_walkforward(days):
    out = []
    for d in days:
        day_start = pd.Timestamp(d).timestamp()
        tr = all_f[all_f.mt <= day_start - 3700]
        te = all_f[all_f.day == d].copy()
        out.append(add_ev(tr, te))
        print(d, "done")
    o = pd.concat(out)
    o["best_ev"] = o[EVCOLS].max(axis=1)
    o["best_k"] = np.array(LEVELS)[o[EVCOLS].values.argmax(axis=1)]
    return o

oof3 = run_walkforward(VAL_DAYS2)

for k in LEVELS:
    print(f"{k}x AUC={roc_auc_score(oof3[f'win{k}'], oof3[f'p{k}']):.3f}")

rows = []
for margin in [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
    tp = np.where(oof3.best_ev >= margin, oof3.best_k, 0.0)
    pn = pnl_for(tp, oof3.max_x.values, oof3.final_x.values)
    daily = pd.Series(pn).groupby(oof3.day.values).sum()
    rows.append(dict(margin=margin, trades=int((tp > 0).sum()), total_pnl=round(pn.sum(), 2),
                     days_pos=int((daily > 0).sum()), days_neg=int((daily < 0).sum())))
print(pd.DataFrame(rows).to_string(index=False))

# %% [code] {"execution":{"iopub.status.busy":"2026-11-22T05:12:30.900369Z","iopub.execute_input":"2026-11-22T05:12:30.900609Z","iopub.status.idle":"2026-11-22T05:12:31.239449Z","shell.execute_reply.started":"2026-11-22T05:12:30.900593Z","shell.execute_reply":"2026-11-22T05:12:31.238533Z"}}
# Cell 13: Which features separate winners from losers?
# d = all_f[all_f.mt > all_f.mt.max() - 40 * 86400].copy()
# y = np.log(d.max_x.clip(lower=0.05))
# rows = []
# for f in FEATURES:
#     x = d[f]
#     if x.notna().sum() < 1000:
#         continue
#     q = x.rank(pct=True)
#     rows.append(dict(
#         feature=f,
#         spearman=round(x.rank().corr(y.rank()), 3),
#         win5_bottom10=round(d.loc[q <= 0.1, "win5"].mean(), 3),
#         win5_top10=round(d.loc[q >= 0.9, "win5"].mean(), 3),
#         win3_bottom10=round(d.loc[q <= 0.1, "win3"].mean(), 3),
#         win3_top10=round(d.loc[q >= 0.9, "win3"].mean(), 3),
#     ))
# res = pd.DataFrame(rows)
# res = res.reindex(res.spearman.abs().sort_values(ascending=False).index)
# print("base rates: win3 =", round(d.win3.mean(), 3), "| win5 =", round(d.win5.mean(), 3))
# print(res.head(25).to_string(index=False))

# %% [code] {"execution":{"iopub.status.busy":"2026-11-22T05:12:56.398480Z","iopub.execute_input":"2026-11-22T05:12:56.398814Z","iopub.status.idle":"2026-11-22T05:13:31.363093Z","shell.execute_reply.started":"2026-11-22T05:12:56.398794Z","shell.execute_reply":"2026-11-22T05:13:31.362155Z"}}
# Cell 14: Predict the size of the run instead of yes/no
# def wf_reg(days, feats):
#     out = []
#     for dd in days:
#         ds = pd.Timestamp(dd).timestamp()
#         tr = all_f[all_f.mt <= ds - 3700]
#         te = all_f[all_f.day == dd].copy()
#         ytr = np.log(tr.max_x.clip(0.05, 30))
#         s = np.zeros(len(te))
#         for seed in (0, 1):
#             m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=16,
#                                   min_child_samples=40, subsample=0.8, subsample_freq=1,
#                                   colsample_bytree=0.7, reg_lambda=5.0,
#                                   random_state=seed, n_jobs=4, verbose=-1)
#             m.fit(tr[feats], ytr)
#             s += m.predict(te[feats]) / 2
#         te["score"] = s
#         out.append(te)
#         print(dd, "done")
#     return pd.concat(out)

# oofr = wf_reg(VAL_DAYS2, FEATURES)
# for k in LEVELS:
#     auc = roc_auc_score(oofr[f"win{k}"], oofr["score"])
#     top = oofr.nlargest(int(len(oofr) * 0.05), "score")
#     print(f"{k}x  AUC={auc:.3f} | hit rate in top 5%={top[f'win{k}'].mean():.3f} "
#           f"| base={oofr[f'win{k}'].mean():.3f} | lift={top[f'win{k}'].mean()/oofr[f'win{k}'].mean():.1f}x")

# %% [code]
# Cell 15: Final submission
# pick the margin from the walk-forward (oof3 from Cell 12); fall back to "skip everything"
best = None
for margin in [-0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
    tp = np.where(oof3.best_ev >= margin, oof3.best_k, 0.0)
    pn = pnl_for(tp, oof3.max_x.values, oof3.final_x.values)
    daily = pd.Series(pn).groupby(oof3.day.values).sum()
    trades, total = int((tp > 0).sum()), pn.sum()
    if trades >= 15 and total > 0 and (daily > 0).sum() > (daily < 0).sum():
        if best is None or total > best[1]:
            best = (margin, total, trades)

MARGIN = best[0] if best else 99.0
print("walk-forward choice:", best, "-> margin =", MARGIN, "(99 = skip everything)")

# train on all labelled history, score the test tokens
test_final = add_ev(all_f, test_f.copy())
test_final["best_ev"] = test_final[EVCOLS].max(axis=1)
test_final["best_k"] = np.array(LEVELS)[test_final[EVCOLS].values.argmax(axis=1)]
test_final["tp"] = np.where(test_final.best_ev >= MARGIN, test_final.best_k, 0.0)

sub = sample[["mint"]].merge(test_final[["mint", "tp"]], on="mint", how="left").fillna({"tp": 0.0})
assert len(sub) == len(sample)
assert sub.tp.apply(lambda v: v == 0 or 3 <= v <= 100).all()
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("tokens bought:", int((sub.tp > 0).sum()), "of", len(sub))
sub.head()