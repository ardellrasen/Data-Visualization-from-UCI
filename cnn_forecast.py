"""Peramalan konsumsi listrik 1 jam ke depan dengan CNN 1D, lengkap dengan visualisasi.

Pemakaian: python cnn_forecast.py <data.zip> <folder_output>
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["KERAS_BACKEND"] = "torch"   # harus sebelum import keras

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from keras import layers

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
FIG = OUT / "figures_cnn"
FIG.mkdir(parents=True, exist_ok=True)
WINDOWS = [24, 48, 168]
SEED = 42
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": .3})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 1. DATA
raw = pd.read_csv(RAW, sep=";", low_memory=False, na_values=["?"], dtype={"Date": str, "Time": str})
raw["Datetime"] = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
minute = raw.set_index("Datetime")["Global_active_power"].astype(float).asfreq("min")
grp = minute.resample("h")
hourly = grp.mean().where(grp.count() >= 30)      # jam dengan < 30 menit valid dianggap kosong

isna = hourly.isna()
block = (isna != isna.shift()).cumsum()
gap_len = isna.groupby(block).transform("sum")
series = hourly.interpolate(limit_area="inside").where(~isna | (gap_len <= 3))  # hanya gap <= 3 jam

# ------------------------------------------------------------------ 2. SPLIT (validasi & uji masing-masing 1 tahun penuh)
end = series.index[-1]
test_start = end - pd.DateOffset(years=1)
val_start = test_start - pd.DateOffset(years=1)
train_mask = series.index < val_start
mu, sigma = series[train_mask].mean(), series[train_mask].std()   # hanya dari data latih

idx = series.index
features = np.column_stack([
    ((series - mu) / sigma).to_numpy(),
    np.sin(2 * np.pi * idx.hour / 24), np.cos(2 * np.pi * idx.hour / 24),
    np.sin(2 * np.pi * idx.day_of_week / 7), np.cos(2 * np.pi * idx.day_of_week / 7),
    np.sin(2 * np.pi * idx.day_of_year / 365.25), np.cos(2 * np.pi * idx.day_of_year / 365.25),
]).astype("float32")
FEATURE_NAMES = ["daya", "jam_sin", "jam_cos", "hari_sin", "hari_cos", "tahun_sin", "tahun_cos"]
target = features[:, 0]
valid = np.isfinite(target)


def make_windows(window, start, stop):
    """Window input yang seluruhnya valid; target = jam berikutnya dan berada di [start, stop)."""
    ok = np.convolve(valid.astype(int), np.ones(window + 1, dtype=int), "valid") == window + 1
    ends = np.where(ok)[0] + window                       # posisi target
    ends = ends[(idx[ends] >= start) & (idx[ends] < stop)]
    X = np.stack([features[e - window:e] for e in ends])
    return X, target[ends][:, None], ends


def to_kw(z):
    return np.asarray(z).ravel() * sigma + mu


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def build_cnn(window, n_features):
    return keras.Sequential([
        layers.Input((window, n_features)),
        layers.Conv1D(32, 3, padding="causal", activation="relu"),
        layers.Conv1D(32, 3, padding="causal", dilation_rate=2, activation="relu"),
        layers.Conv1D(32, 3, padding="causal", dilation_rate=4, activation="relu"),
        layers.Flatten(),   # mempertahankan posisi jam di dalam window
        layers.Dense(32, activation="relu"),
        layers.Dropout(.1),
        layers.Dense(1),
    ])


# ------------------------------------------------------------------ 3. LATIH & BANDINGKAN WINDOW
keras.utils.set_random_seed(SEED)
res = {"window": {}, "n_hours": int(len(series)), "val_start": str(val_start), "test_start": str(test_start)}
best = None
for w in WINDOWS:
    Xtr, ytr, _ = make_windows(w, idx[0], val_start)
    Xva, yva, _ = make_windows(w, val_start, test_start)
    model = build_cnn(w, features.shape[1])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
    t0 = time.time()
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=30, batch_size=128, verbose=2,
                     callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    val_mae = mae(to_kw(yva), to_kw(model.predict(Xva, verbose=0)))
    res["window"][w] = {"val_mae_kw": val_mae, "epochs": len(hist.history["loss"]), "seconds": round(time.time() - t0, 1),
                        "n_train": int(len(Xtr)), "n_val": int(len(Xva))}
    print(w, res["window"][w], flush=True)
    if best is None or val_mae < best["val_mae"]:
        best = {"w": w, "val_mae": val_mae, "model": model, "hist": hist.history}

# ------------------------------------------------------------------ 4. UJI (sekali, dengan window terbaik dari validasi)
w, model = best["w"], best["model"]
res["best_window"] = w
Xte, yte, ends = make_windows(w, test_start, end + pd.Timedelta(hours=1))
y_true = to_kw(yte)
y_cnn = to_kw(model.predict(Xte, verbose=0))
t = idx[ends]
raw_kw = to_kw(target)
preds = {"Persistence (t-1)": raw_kw[ends - 1], "Seasonal naive (t-24)": raw_kw[ends - 24],
         "Seasonal naive (t-168)": raw_kw[ends - 168], "CNN 1D": y_cnn}
res["test"] = {}
for name, p in preds.items():
    m = np.isfinite(p)
    res["test"][name] = {"mae": mae(y_true[m], p[m]), "rmse": rmse(y_true[m], p[m]), "n": int(m.sum())}
df = pd.DataFrame({"actual": y_true, "cnn": y_cnn, "persist": raw_kw[ends - 1]}, index=t)
df["err"] = (df.cnn - df.actual).abs()
df["err_p"] = (df.persist - df.actual).abs()
df.to_csv(OUT / "cnn_test_predictions.csv")
res["mae_by_month"] = {int(k): float(v) for k, v in df.groupby(df.index.month)["err"].mean().items()}
res["mae_by_hour"] = {int(k): float(v) for k, v in df.groupby(df.index.hour)["err"].mean().items()}
res["bias_kw"] = float((df.cnn - df.actual).mean())
res["top_decile_actual"] = {"threshold": float(df.actual.quantile(.9))}
top = df[df.actual >= df.actual.quantile(.9)]
res["top_decile_actual"].update({"mae": float(top.err.mean()), "bias": float((top.cnn - top.actual).mean())})
res["weekend_mae"] = {"weekday": float(df[df.index.day_of_week < 5].err.mean()), "weekend": float(df[df.index.day_of_week >= 5].err.mean())}

# Pentingnya tiap lag: seberapa besar MAE naik jika satu posisi input daya diacak
rng = np.random.default_rng(SEED)
sample = rng.choice(len(Xte), min(2000, len(Xte)), replace=False)
Xs, ys = Xte[sample], y_true[sample]
base = mae(ys, to_kw(model.predict(Xs, verbose=0, batch_size=512)))
imp = {}
for lag in range(1, w + 1):
    Xp = Xs.copy()
    Xp[:, -lag, 0] = rng.permutation(Xp[:, -lag, 0])
    imp[lag] = mae(ys, to_kw(model.predict(Xp, verbose=0, batch_size=512))) - base
res["lag_importance_top10"] = dict(sorted(((int(k), float(v)) for k, v in imp.items()), key=lambda kv: -kv[1])[:10])

# ------------------------------------------------------------------ 5. GAMBAR
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(best["hist"]["loss"], label="Latih"); ax.plot(best["hist"]["val_loss"], label="Validasi")
ax.set(title=f"Fig C1. Kurva pelatihan (window {w} jam)", xlabel="Epoch", ylabel="MAE (skala standar)"); ax.legend()
save(fig, "c1_training_curve.png")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ws = list(res["window"]); ax[0].bar([f"{x} jam" for x in ws], [res["window"][x]["val_mae_kw"] for x in ws], color="#4c72b0")
ax[0].set(title="Fig C2a. MAE validasi per panjang window", ylabel="MAE (kW)")
names = list(res["test"]); ypos = np.arange(len(names))
ax[1].barh(ypos, [res["test"][n]["mae"] for n in names], color=["#b0b0b0"] * 3 + ["#d95f02"])
ax[1].set_yticks(ypos, names); ax[1].invert_yaxis()
ax[1].set(title="Fig C2b. MAE data uji: CNN vs baseline", xlabel="MAE (kW)")
save(fig, "c2_comparison.png")

fig, ax = plt.subplots(2, 1, figsize=(13, 7))
for a, (lab, s0) in zip(ax, [("musim dingin", "2010-01-04"), ("musim panas", "2010-07-05")]):
    z = df[s0:str((pd.Timestamp(s0) + pd.Timedelta(days=6)).date())]
    a.plot(z.index, z.actual, color="k", lw=1.3, label="Aktual"); a.plot(z.index, z.cnn, color="#d95f02", lw=1.3, label="Prediksi CNN")
    a.set(title=f"Fig C3. Aktual vs prediksi, satu minggu {lab} ({s0})", ylabel="Daya aktif (kW)"); a.legend()
save(fig, "c3_actual_vs_pred.png")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
hb = ax[0].hexbin(df.actual, df.cnn, gridsize=50, cmap="YlOrRd", mincnt=1)
lim = [0, max(df.actual.max(), df.cnn.max())]; ax[0].plot(lim, lim, "k--", lw=1, label="Prediksi sempurna"); ax[0].legend()
ax[0].set(title="Fig C4a. Prediksi vs aktual", xlabel="Aktual (kW)", ylabel="Prediksi (kW)"); fig.colorbar(hb, ax=ax[0], label="Jumlah jam")
ax[1].hist(df.cnn - df.actual, bins=80, color="#4c72b0"); ax[1].axvline(0, color="k", ls="--", lw=1)
ax[1].set(title="Fig C4b. Distribusi error (prediksi - aktual)", xlabel="Error (kW)", ylabel="Jumlah jam")
save(fig, "c4_error_distribution.png")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(df.groupby(df.index.hour).err.mean(), marker="o", color="#d95f02", label="CNN")
ax[0].plot(df.groupby(df.index.hour).err_p.mean(), marker="o", color="gray", label="Persistence")
ax[0].set(title="Fig C5a. MAE per jam dalam sehari", xlabel="Jam", ylabel="MAE (kW)", xticks=range(0, 24, 2)); ax[0].legend()
mon = df.groupby(df.index.month).err.mean()
ax[1].bar(mon.index, mon.values, color="#55a868"); ax[1].set(title="Fig C5b. MAE per bulan", xlabel="Bulan", ylabel="MAE (kW)", xticks=range(1, 13))
save(fig, "c5_error_by_time.png")

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(list(imp), list(imp.values()), color="#4c72b0")
for k in range(24, w + 1, 24):
    ax.axvline(k, color="r", ls=":", lw=.7)
ax.set(title="Fig C6. Lag yang dipakai CNN: kenaikan MAE saat satu lag diacak", xlabel="Lag (jam ke belakang)", ylabel="Kenaikan MAE (kW)")
save(fig, "c6_lag_importance.png")

model.save(OUT / "cnn_best.keras")
with open(OUT / "cnn_metrics.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
print("DONE")
