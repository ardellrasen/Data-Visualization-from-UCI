"""Fase 1 - EDA: Electricity consumption (UCI Individual Household Electric Power Consumption).

Menjalankan analisis nyata dan menyimpan angka ke outputs/metrics/eda_stats.json
serta gambar ke outputs/figures/. Tidak ada angka yang ditulis manual.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss

RAW = Path(sys.argv[1])
OUT = Path(sys.argv[2])
FIG = OUT / "figures"
MET = OUT / "metrics"
FIG.mkdir(parents=True, exist_ok=True)
MET.mkdir(parents=True, exist_ok=True)

TARGET = "Global_active_power"
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.titleweight": "bold"})
stats = {}


def fmt_dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelsize=8)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 1. LOAD
raw = pd.read_csv(RAW, sep=";", low_memory=False, na_values=["?"],
                  dtype={"Date": str, "Time": str})
stats["n_rows"], stats["n_cols"] = map(int, raw.shape)
stats["dtypes"] = {k: str(v) for k, v in raw.dtypes.items()}
print("shape:", raw.shape)
print(raw.dtypes)

# ---------------------------------------------------------------- 2. DATETIME INDEX
raw["Datetime"] = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
df = raw.drop(columns=["Date", "Time"]).set_index("Datetime")
num_cols = [c for c in df.columns]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
stats["start"], stats["end"] = str(df.index.min()), str(df.index.max())
print("range:", stats["start"], "->", stats["end"])

# ---------------------------------------------------------------- 3. MISSING / DUPLICATE / CONTINUITY
stats["duplicated_timestamps"] = int(df.index.duplicated().sum())
stats["index_monotonic_increasing"] = bool(df.index.is_monotonic_increasing)
full_idx = pd.date_range(df.index.min(), df.index.max(), freq="min")
stats["expected_minutes"] = int(len(full_idx))
stats["missing_timestamps_rows"] = int(len(full_idx) - df.index.nunique())

miss_per_col = df.isna().sum()
stats["nan_per_column"] = {k: int(v) for k, v in miss_per_col.items()}
stats["nan_pct_target"] = float(df[TARGET].isna().mean() * 100)
stats["nan_rows_all_columns_equal"] = bool((df.isna().sum(axis=1).isin([0, 7])).all())

df = df[~df.index.duplicated(keep="first")].reindex(full_idx)
df.index.name = "Datetime"
s_min = df[TARGET]
stats["nan_minutes_after_reindex"] = int(s_min.isna().sum())

# run-length of NaN gaps (minutes)
isn = s_min.isna().to_numpy()
edges = np.diff(np.r_[0, isn.astype(int), 0])
starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
runs = ends - starts
gaps = pd.DataFrame({"start": s_min.index[starts], "minutes": runs}).sort_values("minutes", ascending=False)
stats["n_gaps"] = int(len(gaps))
stats["gap_minutes_describe"] = {k: float(v) for k, v in pd.Series(runs).describe().items()}
stats["gaps_ge_60min"] = int((runs >= 60).sum())
stats["gaps_ge_180min"] = int((runs >= 180).sum())
stats["gaps_ge_1440min"] = int((runs >= 1440).sum())
stats["top10_gaps"] = [{"start": str(r.start), "minutes": int(r.minutes)} for r in gaps.head(10).itertuples()]
stats["share_nan_minutes_in_gaps_ge_60"] = float(runs[runs >= 60].sum() / runs.sum())
gaps.to_csv(MET / "missing_gaps.csv", index=False)
print("NaN minutes:", stats["nan_minutes_after_reindex"], "gaps:", len(gaps), "longest:", runs.max())

# ---------------------------------------------------------------- 4. STATISTICS (minute-level, valid values only)
d = s_min.dropna()
stats["minute_stats"] = {k: float(v) for k, v in d.describe().items()}
stats["minute_skew"], stats["minute_kurtosis"] = float(d.skew()), float(d.kurt())
stats["minute_zero_values"] = int((d == 0).sum())

# ---------------------------------------------------------------- 5. HOURLY RESAMPLE
grp = s_min.resample("h")
hourly = grp.mean()                      # rata-rata dari menit valid saja
valid_cnt = grp.count()                  # jumlah menit valid per jam
stats["n_hours"] = int(len(hourly))
stats["hours_all_nan"] = int(hourly.isna().sum())
stats["hours_partial"] = int(((valid_cnt > 0) & (valid_cnt < 60)).sum())
stats["hours_full"] = int((valid_cnt == 60).sum())
stats["hourly_stats"] = {k: float(v) for k, v in hourly.dropna().describe().items()}
stats["hourly_skew"], stats["hourly_kurtosis"] = float(hourly.skew()), float(hourly.kurt())
stats["hourly_min"] = float(hourly.min())
stats["hourly_zero_or_near_zero_lt_0.1"] = int((hourly < 0.1).sum())

# gap panjang pada level jam
ish = hourly.isna().to_numpy()
e2 = np.diff(np.r_[0, ish.astype(int), 0])
s2, en2 = np.where(e2 == 1)[0], np.where(e2 == -1)[0]
hr = en2 - s2
stats["hourly_nan_runs"] = int(len(hr))
stats["hourly_nan_runs_gt_3h"] = int((hr > 3).sum())
stats["hourly_nan_hours_in_runs_gt_3h"] = int(hr[hr > 3].sum()) if len(hr) else 0
stats["hourly_longest_nan_run_h"] = int(hr.max()) if len(hr) else 0
hourly.to_csv(MET / "hourly_global_active_power_raw.csv", header=True)

# ---------------------------------------------------------------- 6. STATIONARITY
h = hourly.dropna()
adf = adfuller(h.values, autolag="AIC")
stats["adf"] = {"stat": float(adf[0]), "pvalue": float(adf[1]), "lags": int(adf[2]), "crit": {k: float(v) for k, v in adf[4].items()}}
kp = kpss(h.values, regression="c", nlags="auto")
stats["kpss_level"] = {"stat": float(kp[0]), "pvalue": float(kp[1]), "lags": int(kp[2])}
kpt = kpss(h.values, regression="ct", nlags="auto")
stats["kpss_trend"] = {"stat": float(kpt[0]), "pvalue": float(kpt[1])}

# tren linear sederhana + perbandingan antar tahun/bulan
hh = hourly.copy()
yearly = hh.groupby(hh.index.year).agg(["mean", "count"])
stats["yearly_mean"] = {int(k): float(v) for k, v in yearly["mean"].items()}
monthly = hh.resample("MS").mean()
stats["monthly_mean"] = {str(k.date()): (None if np.isnan(v) else float(v)) for k, v in monthly.items()}
# pergeseran rata-rata antar 3 segmen kronologis 70/15/15 (informasi, bukan keputusan modelling)
n = len(hh)
i1, i2 = int(n * .70), int(n * .85)
for nm, seg in [("train", hh.iloc[:i1]), ("val", hh.iloc[i1:i2]), ("test", hh.iloc[i2:])]:
    stats[f"segment_{nm}"] = {"start": str(seg.index.min()), "end": str(seg.index.max()),
                               "mean": float(seg.mean()), "std": float(seg.std()),
                               "months_covered": sorted({int(m) for m in seg.index.month.unique()})}

# ACF numerik pada lag kunci (nilai jam terisi interpolasi pendek hanya untuk ACF)
hi = hourly.interpolate(limit=3).dropna()
acf_vals = {lag: float(hi.autocorr(lag)) for lag in [1, 2, 3, 6, 12, 24, 48, 168]}
stats["acf_hourly"] = acf_vals

# ---------------------------------------------------------------- 7. POLA
hh_df = hourly.to_frame("p")
hh_df["hour"], hh_df["dow"], hh_df["month"] = hh_df.index.hour, hh_df.index.dayofweek, hh_df.index.month
hh_df["weekend"] = hh_df["dow"] >= 5
hod = hh_df.groupby("hour")["p"].mean()
dow = hh_df.groupby("dow")["p"].mean()
stats["hour_of_day_mean"] = {int(k): float(v) for k, v in hod.items()}
stats["peak_hour"], stats["trough_hour"] = int(hod.idxmax()), int(hod.idxmin())
stats["dow_mean"] = {int(k): float(v) for k, v in dow.items()}
stats["weekday_vs_weekend_mean"] = {"weekday": float(hh_df.loc[~hh_df.weekend, "p"].mean()),
                                    "weekend": float(hh_df.loc[hh_df.weekend, "p"].mean())}
mm = hh_df.groupby("month")["p"].mean()
stats["month_of_year_mean"] = {int(k): float(v) for k, v in mm.items()}

# outlier IQR (jam)
q1, q3 = h.quantile([.25, .75])
iqr = q3 - q1
ub = q3 + 1.5 * iqr
stats["iqr_upper_fence"] = float(ub)
stats["n_hourly_above_fence"] = int((h > ub).sum())
stats["pct_hourly_above_fence"] = float((h > ub).mean() * 100)
stats["top5_hours"] = {str(k): float(v) for k, v in h.nlargest(5).items()}

# ---------------------------------------------------------------- 8. FIGURES
# F1 original minute series (dengan 3 zoom)
fig, ax = plt.subplots(2, 1, figsize=(13, 7))
ax[0].plot(s_min.index, s_min.values, lw=.3, color="#1f77b4")
ax[0].set(title="Fig 1a. Konsumsi daya aktif asli (resolusi 1 menit), seluruh periode",
          ylabel="Global_active_power (kW)", xlabel="Waktu")
zoom = s_min["2007-01-08":"2007-01-14"]
ax[1].plot(zoom.index, zoom.values, lw=.6, color="#1f77b4")
ax[1].set(title="Fig 1b. Zoom satu minggu (8-14 Jan 2007), resolusi 1 menit",
          ylabel="Global_active_power (kW)", xlabel="Waktu")
save(fig, "01_original_timeseries.png")

# F2 hourly
fig, ax = plt.subplots(2, 1, figsize=(13, 7))
ax[0].plot(hourly.index, hourly.values, lw=.4, color="#d95f02")
ax[0].plot(hourly.resample("W").mean().index, hourly.resample("W").mean().values, color="k", lw=1.3, label="Rata-rata mingguan")
ax[0].set(title="Fig 2a. Konsumsi rata-rata per jam, seluruh periode", ylabel="Daya aktif (kW)", xlabel="Waktu")
ax[0].legend()
z = hourly["2007-01-08":"2007-01-14"]
zm = s_min["2007-01-08":"2007-01-14"]
ax[1].plot(zm.index, zm.values, lw=.4, color="#1f77b4", alpha=.45, label="1 menit")
ax[1].plot(z.index, z.values, lw=1.6, color="#d95f02", label="Rata-rata per jam")
ax[1].set(title="Fig 2b. Efek resampling: 1 menit vs rata-rata jam (8-14 Jan 2007)", ylabel="Daya aktif (kW)", xlabel="Waktu")
ax[1].legend()
save(fig, "02_hourly_timeseries.png")

# F3 daily pattern
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
sns.lineplot(data=hh_df.dropna(), x="hour", y="p", errorbar=("sd", 1), ax=ax[0], color="#d95f02")
ax[0].set(title="Fig 3a. Profil harian (rata-rata ± 1 SD)", xlabel="Jam dalam sehari (waktu lokal pencatatan)", ylabel="Daya aktif (kW)", xticks=range(0, 24, 2))
sns.lineplot(data=hh_df.dropna(), x="hour", y="p", hue="weekend", errorbar=None, ax=ax[1], palette=["#1b9e77", "#7570b3"])
ax[1].set(title="Fig 3b. Profil harian: hari kerja vs akhir pekan", xlabel="Jam dalam sehari", ylabel="Daya aktif (kW)", xticks=range(0, 24, 2))
h_, l_ = ax[1].get_legend_handles_labels(); ax[1].legend(h_, ["Hari kerja", "Akhir pekan"])
save(fig, "03_daily_pattern.png")

# F4 weekly
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
names = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
sns.barplot(x=names, y=dow.values, ax=ax[0], color="#4c72b0")
ax[0].set(title="Fig 4a. Rata-rata konsumsi per hari dalam seminggu", ylabel="Daya aktif (kW)", xlabel="Hari")
heat = hh_df.dropna().pivot_table(index="dow", columns="hour", values="p", aggfunc="mean")
heat.index = names
sns.heatmap(heat, cmap="YlOrRd", ax=ax[1], cbar_kws={"label": "kW"})
ax[1].set(title="Fig 4b. Heatmap rata-rata: hari × jam", xlabel="Jam", ylabel="Hari")
save(fig, "04_weekly_pattern.png")

# F5 monthly / seasonal
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
mnames = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
sns.barplot(x=[mnames[i - 1] for i in mm.index], y=mm.values, ax=ax[0], color="#55a868")
ax[0].set(title="Fig 5a. Rata-rata konsumsi per bulan (gabungan semua tahun)", ylabel="Daya aktif (kW)", xlabel="Bulan")
ax[1].plot(monthly.index, monthly.values, marker="o", ms=3, color="#c44e52")
ax[1].axvline(hh.index[i1], color="k", ls="--", lw=1, label="Batas train | val")
ax[1].axvline(hh.index[i2], color="gray", ls="--", lw=1, label="Batas val | test")
ax[1].set(title="Fig 5b. Rata-rata bulanan sepanjang waktu + batas split 70/15/15", ylabel="Daya aktif (kW)", xlabel="Bulan")
ax[1].legend()
fmt_dates(ax[1])
save(fig, "05_monthly_seasonal.png")

# F6 histogram
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
sns.histplot(d.values, bins=100, ax=ax[0], color="#1f77b4")
ax[0].set(title="Fig 6a. Distribusi data 1 menit", xlabel="Daya aktif (kW)", ylabel="Frekuensi")
sns.histplot(h.values, bins=80, kde=True, ax=ax[1], color="#d95f02")
ax[1].axvline(h.mean(), color="k", ls="--", label=f"Mean = {h.mean():.2f} kW")
ax[1].axvline(h.median(), color="g", ls=":", label=f"Median = {h.median():.2f} kW")
ax[1].set(title="Fig 6b. Distribusi rata-rata per jam", xlabel="Daya aktif (kW)", ylabel="Frekuensi")
ax[1].legend()
save(fig, "06_histogram.png")

# F7 boxplot
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
sns.boxplot(y=h.values, ax=ax[0], color="#fdb462", fliersize=2)
ax[0].set(title="Fig 7a. Boxplot data per jam", ylabel="Daya aktif (kW)")
sns.boxplot(data=hh_df.dropna(), x="hour", y="p", ax=ax[1], color="#80b1d3", fliersize=1)
ax[1].set(title="Fig 7b. Boxplot per jam dalam sehari", xlabel="Jam", ylabel="Daya aktif (kW)")
ax[1].tick_params(axis="x", labelsize=7)
sns.boxplot(data=hh_df.dropna(), x="month", y="p", ax=ax[2], color="#b3de69", fliersize=1)
ax[2].set(title="Fig 7c. Boxplot per bulan", xlabel="Bulan (1-12)", ylabel="Daya aktif (kW)")
save(fig, "07_boxplot.png")

# F8 missing values
fig, ax = plt.subplots(2, 1, figsize=(13, 7))
daily_nan = s_min.isna().astype(float).resample("D").mean() * 100
stats["days_with_any_nan"] = int((daily_nan > 0).sum())
stats["days_total"] = int(len(daily_nan))
stats["days_with_nan_gt_50pct"] = int((daily_nan > 50).sum())
ax[0].fill_between(daily_nan.index, daily_nan.values, step="mid", color="#e41a1c")
ax[0].vlines(daily_nan.index[daily_nan > 0], 0, daily_nan[daily_nan > 0], color="#e41a1c", lw=1.2)
ax[0].set(title="Fig 8a. Persentase menit hilang per hari (hanya hari bermasalah tampak)", ylabel="Menit hilang (%)",
          xlabel="Tanggal", ylim=(0, 105))
fmt_dates(ax[0])
top = gaps[gaps["minutes"] >= 60].sort_values("start")
ax[1].bar([str(t.date()) for t in top["start"]], top["minutes"] / 60, color="#377eb8")
ax[1].set(title="Fig 8b. Gap berdurasi >= 1 jam (data hilang berurutan)", ylabel="Durasi gap (jam)", xlabel="Awal gap")
ax[1].tick_params(axis="x", rotation=45)
save(fig, "08_missing_values.png")

# F9 ACF + decomposition-like (stasioneritas & periodisitas)
from statsmodels.graphics.tsaplots import plot_acf
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
plot_acf(hi.values, lags=24 * 8, ax=ax[0], zero=False, alpha=.05)
for k in range(1, 9):
    ax[0].axvline(24 * k, color="r", ls=":", lw=.6)
ax[0].set(title="Fig 9a. Autokorelasi (ACF) data per jam, lag hingga 8 hari", xlabel="Lag (jam)", ylabel="Autokorelasi")
roll = hourly.rolling(24 * 30, min_periods=24 * 15)
ax[1].plot(hourly.index, roll.mean(), label="Rolling mean 30 hari", color="#1f77b4")
ax[1].plot(hourly.index, roll.std(), label="Rolling std 30 hari", color="#ff7f0e")
ax[1].set(title="Fig 9b. Rolling mean & std (cek stasioneritas)", xlabel="Waktu", ylabel="Daya aktif (kW)")
ax[1].legend()
fmt_dates(ax[1])
save(fig, "09_acf_stationarity.png")

with open(MET / "eda_stats.json", "w") as f:
    json.dump(stats, f, indent=2, default=str)
print("DONE")
