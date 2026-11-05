"""
VortX: A Machine-Learning Framework for Subseasonal Prediction of Sudden Stratospheric Warmings

By Allison Zhou
"""
# ------------------------- import all the libraries -------------------------

import os
import glob
import urllib.request
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, roc_auc_score
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.metrics import roc_curve

# ------------------------- configure the pre settings -------------------------

# make the paths
RAW = "data/raw"
PROC = "data/processed"
RESULTS = "results"

# create output folders if missing
os.makedirs(PROC, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
os.makedirs("figures", exist_ok=True)

# target and forecast schedule
TARGET_COL = "u60_10" # zonal-mean u at 60N, 10 hPa
WINDOW = 15 # target averaging window (days)
LEADS = [5, 10, 15, 20, 25, 30] # days from forecast day to start of target window
WINTER_MONTHS = [11, 12, 1, 2, 3] # nov-mar
EMBARGO = 25 # training days removed on each side of winter
PRE, POST = 3, 10 # ssw event window: -3 to +10 days from onset

# configure the pre-set XGBoost hyperparameters
XGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    early_stopping_rounds=50, 
)

# feature groups used in each ablation run, remove 1 physical factor per round
ABLATIONS = {
    "full": ["F", "W", "Q"],
    "no_F": ["W", "Q"], # remove wave forcing
    "no_W": ["F", "Q"], # remove vortex state
    "no_Q": ["F", "W"], # remove QBO
}

# ------------------------- step 1: daily table -------------------------
# qbo = quasi-biennial oscillation. the wind shifts between westerly vs easterly phase. ssw events are most likely to occur in EASTERLY
def load_qbo(dates):
    # NOAA monthly QBO index labeled westerly (>= 0) or easterly (< 0), 1 value per day 
    lines = urllib.request.urlopen("https://psl.noaa.gov/data/correlation/qbo.data").read().decode().splitlines()
    q = pd.DataFrame([l.split() for l in lines[1:] if len(l.split()) == 13], dtype=float)

    # reshape to 1 value per (year, month) and mark missing values (-999 or -99.99) as NaN
    q = q.set_index(0).stack()
    q.index = pd.PeriodIndex([f"{int(y)}-{int(m)}" for y, m in q.index], freq="M")
    q[q < -90] = np.nan

    # match each day to its month and fill missing recent months with the last available value
    q = q.reindex(dates.to_period("M")).ffill().values
    return np.where(q >= 0, "westerly", "easterly")


def build_full_dataset():
    # create a full dataset that has 1 row per day: EP flux, wave-1/2 of 10 hPa height, u60 at 10 hPa, QBO phase
    from aostools import climate as ao
    rows = []
    for f in sorted(glob.glob(f"{RAW}/era5_*.nc")):
        ds = xr.open_dataset(f).rename(valid_time="time") # dimension for aostools

        # calculate the EP flux from u, v, T; vertical component at 100 hPa averaged over 45-75N, then flip sign to indicate positive = upward
        _, ep2, _, _ = ao.ComputeEPfluxDivXr(ds["u"], ds["v"], ds["t"], lon="longitude", lat="latitude", pres="pressure_level", time="time")
        ep = -ep2.sel(pressure_level=100, latitude=slice(75, 45)).mean("latitude")

        # find fft around each latitude circle with index 1 = w-1 coefficient, index 2 = w-2 coefficient
        z10 = ds["z"].sel(pressure_level=10)
        w1 = np.fft.rfft(z10.sel(latitude=slice(75, 45)).mean("latitude").values, axis=1)[:, 1]
        w2 = np.fft.rfft(z10.sel(latitude=60).values, axis=1)[:, 2]

        # zonal-mean u at 60N, 10 hPa
        u60 = ds["u"].sel(pressure_level=10, latitude=60).mean("longitude")
        
        # daily row, wave amplitude = |coefficient|, also the w-2 direction is not used
        rows.append(pd.DataFrame({
            "ep_flux": ep.values,
            "wave1_real": w1.real,
            "wave1_imag": w1.imag,
            "wave1_amp": np.abs(w1),
            "wave2_amp": np.abs(w2),
            TARGET_COL: u60.values,
        }, index=pd.to_datetime(ds["time"].values)))
        ds.close()
        
    # join this into 1 table and then add qbo phase
    out = pd.concat(rows).sort_index()
    out.index.name = "date"
    out["qbo_phase"] = load_qbo(out.index)
    out.to_csv(f"{PROC}/full_dataset.csv")
    return out

# build the model table: full dataset with QBO as 0/1 and include a winter label
def build_base():
    base = pd.read_csv(f"{PROC}/full_dataset.csv", index_col="date", parse_dates=True)
    base["qbo"] = (base.pop("qbo_phase") == "westerly").astype(int)  # 1 = westerly, 0 = easterly
    base["winter"] = base.index.year + (base.index.month >= 7).astype(int) # jul-dec belong to following winter
    base.to_csv(f"{PROC}/base.csv")
    return base


def load_base():
    return pd.read_csv(f"{PROC}/base.csv", index_col="date", parse_dates=True)


# ------------------------- step 2: find ssw onset events -------------------------

# find SSW onset dates given by the charlton-polvani (2007) criteria:
# - reversal of ū60 (60°N, 10 hPa) from westerly to easterly
# - onset = first easterly day, from nov–mar, after ≥ 20 consecutive westerly days
# - wind must return westerly for ≥ 10 days before april 30
# - no new event until winds have been westerly for 20 consecutive days (prevents double counting)

def find_ssw_onsets(base):
    u = base[TARGET_COL]
    onsets = []
    i = 20
    while i < len(u):
        d = u.index[i]
        
        # onset = first easterly day in nov-mar after at least 20 consecutive westerly days
        if d.month in WINTER_MONTHS and u.iloc[i] < 0 and (u.iloc[i-20:i] >= 0).all():
            
            # event must return westerly for 10+ days before apr 30
            apr30 = pd.Timestamp(d.year if d.month < 7 else d.year + 1, 4, 30)
            if (u.loc[d:apr30] >= 0).sum() >= 10:
                onsets.append(d)
               
                # keep searching and skip ahead until the wind has been westerly for 20 days in a row
                streak = 0
                while i < len(u) and streak < 20:
                    streak = streak + 1 if u.iloc[i] >= 0 else 0
                    i += 1
                continue
        i += 1
    onsets = pd.DatetimeIndex(onsets)
    pd.DataFrame({"onset": onsets}).to_csv(f"{PROC}/ssw_onsets.csv", index=False)
    return onsets

# ------------------------- step 3: define anomalies, features, and target -------------------------

def smooth_annual(s):
    # find the mean of each calendar day (wrap dec into jan if needed)
    s = s.reindex(range(1, 367)).interpolate(limit_direction="both")
    
    # get rid of noise, make each day the average of the surrounding 31 days
    padded = pd.concat([s.iloc[-15:], s, s.iloc[:15]])
    return padded.rolling(31, center=True).mean().iloc[15:-15]


def anomalize(raw, train_mask):
    # use a z score: (value - climatology) / sd, with climatology from training days only
    train = raw[train_mask]
    doy_train = train.index.dayofyear # assign number to each date
    clim = smooth_annual(train.groupby(doy_train).mean()) # mean
    sd = smooth_annual(train.groupby(doy_train).std()) # std
    doy = raw.index.dayofyear
    return (raw.values - clim.loc[doy].values) / sd.loc[doy].values # returned z score


def build_features(a, base):
    # make the imput columns
    X = {}
    ep = a["ep_flux"]
    u = a[TARGET_COL]
    
# define inputs as 3 feature groups: F, W, Q
    # F: wave source
    for lag in range(0, 21, 2): # lag every 2 days
        X[f"F_ep_l{lag}"] = ep.shift(lag) # EP flux L days ago
    X["F_ep_burst10"] = (ep > 1).rolling(10).sum() # days above +1 sd in the last 10
    X["F_ep_burst20"] = (ep > 1).rolling(20).sum() # days above +1 sd in the last 20
    X["F_ep_max10"] = ep.rolling(10).max() # highest EP flux in last 10 days 

    # W: vortex state at 10 hPa
    for lag in range(0, 21, 2):
        X[f"W_u_l{lag}"] = u.shift(lag) # u60 lag days ago
    X["W_dudt5"] = u - u.shift(5) # change in u60 in 5 days
    
    for v in ["wave1_real", "wave1_imag", "wave1_amp", "wave2_amp"]:
        for lag in range(0, 21, 4): # lag every 4 days
            X[f"W_{v}_l{lag}"] = a[v].shift(lag) # wave coefficient L days ago
            
    # create table
    X = pd.DataFrame(X, index=a.index)

    # Q: quasi-biennial oscillation phase
    X["Q_qbo"] = base["qbo"].values # add qbo index
    return X

def build_target(u, lead):
    # target: mean of u over days t + lead ... t + lead + 14 (future date); persistence: mean over days t-14 ... t (past date)
    forward_mean = u[::-1].rolling(WINDOW).mean()[::-1] # mean of current day and next 14 days
    return forward_mean.shift(-lead), u.rolling(WINDOW).mean() # reverse it --> persistence

# ------------------------- step 4: leave one winter out -------------------------

# make a leave-one-out-cross validation. leave one of the winters out and test the rest 47, repeat x48
def run_lowo(lead, groups, base):
    # train on every winter except one, forecast that one, repeat for all x48
    raw = base.drop(columns=["winter", "qbo"])
    target_date = base.index + pd.Timedelta(days=lead)
    in_winter = target_date.month.isin(WINTER_MONTHS) # use only nov-mar
    target_winter = target_date.year + (target_date.month >= 7).astype(int)

    results = []
    for w in np.sort(base["winter"].unique()):
        # set a held out winter and have a 25 day embargo on each side to be removed from training
        wdays = base.index[base["winter"] == w]
        lo, hi = wdays.min() - pd.Timedelta(days=EMBARGO), wdays.max() + pd.Timedelta(days=EMBARGO)
        train_mask = ~((base.index >= lo) & (base.index <= hi))

        # z score using training days only to prevent data leaks from the test winter
        a = pd.DataFrame({c: anomalize(raw[c], train_mask) for c in raw.columns}, index=base.index)
        X = build_features(a, base)
        y, persist = build_target(a[TARGET_COL], lead)
        cols = [c for c in X.columns if c[0] in groups] # pick feature groups (F, W, Q)

        D = pd.concat([X, y.rename("y"), persist.rename("persist")], axis=1)[in_winter].dropna()
        train = D[D.index.isin(base.index[train_mask])]
        test = D[D.index.isin(base.index[target_winter == w])]
        if len(test) == 0:
            continue

        # use last 5 training winters as validation set for early stopping
        train_winter = base.loc[train.index, "winter"]
        is_val = train_winter.isin(np.sort(train_winter.unique())[-5:]).values
        model = xgb.XGBRegressor(random_state=0, **XGB_PARAMS)
        model.fit(train[~is_val][cols], train[~is_val]["y"],
                  eval_set=[(train[is_val][cols], train[is_val]["y"])], verbose=False)

        # save forecasts for the held out winter
        results.append(pd.DataFrame({
            "winter": w,
            "y": test["y"].values,
            "pred": model.predict(test[cols]),
            "persist": test["persist"].values,
        }, index=test.index))
        print(f"  lead {lead:2d} {'+'.join(groups):8s} winter {w}", end="\r")

    R = pd.concat(results)
    R["lead"] = lead
    R["groups"] = "+".join(groups)
    return R


# ------------------------- step 5: score -------------------------
# score model based on AUC of model, percent warned, and EP flux statistics

def ssw_events(R, onsets):
    # decide which forecasts count as forecasts of an SSW
    start = R.index + pd.Timedelta(days=int(R["lead"].iloc[0])) # date each target window starts
    d = (start.values[:, None] - onsets.values[None, :]) / np.timedelta64(1, "D") # window start minus onset date, in days (rows = forecasts, cols = SSWs)
    win = (d >= -PRE) & (d <= POST)  # is true if the window starts 3 days before to 10 days after that SSW
    match = win & (R.index.values[:, None] < onsets.values[None, :])  # is true if also issued before the onset
    keep = match.any(1) | ~win.any(1)  # score forecast if it is an SSW forecast or not near any SSW; drop the ones issued after onset
    return match, keep, d

def score(R, onsets):
    # auc for the model and persistence, SSW events vs everything else
    match, keep, _ = ssw_events(R, onsets)
    ev, K = match.any(1)[keep], R[keep]
    # negate this so a lower predicted wind = higher event score
    return {"lead": R["lead"].iloc[0], "groups": R["groups"].iloc[0],
            "auc_model": roc_auc_score(ev, -K["pred"]), "auc_persist": roc_auc_score(ev, -K["persist"])}


def warnings(R, onsets, col="pred", halo=15):
    # define warning = forecast in the lowest q% at this lead, q = SSW event rate
    match, keep, d = ssw_events(R, onsets)
    q = 100 * match.any(1)[keep].mean()
    warn = (R[col] < np.percentile(R[col][keep], q)).values
    hit = (match & warn[:, None]).any(0) 
    # SSW counts as warned if at least 1 of its forecasts had a warning
    quiet = keep & (np.abs(d).min(1) > halo)
    return hit, warn[quiet].mean() 


def ep_composite(base, warned, missed, lags=np.arange(-40, 11)):
    # mean EP flux anomaly from 40 days before to 10 days after onset, warned vs missed
    ep = base["ep_flux"]
    doy = ep.index.dayofyear
    clim = smooth_annual(ep.groupby(doy).mean())
    sd = smooth_annual(ep.groupby(doy).std())
    epa = pd.Series((ep.values - clim.loc[doy].values) / sd.loc[doy].values, index=ep.index)

    def stack(dates): # rows = events, columns = lags
        return np.array([[epa.get(on + pd.Timedelta(days=int(l)), np.nan) for l in lags] for on in dates])

    W, M = stack(warned), stack(missed)
    return pd.DataFrame({"lag": lags,
                         "warned_mean": np.nanmean(W, axis=0), "warned_se": np.nanstd(W, axis=0) / np.sqrt(len(warned)),
                         "missed_mean": np.nanmean(M, axis=0), "missed_se": np.nanstd(M, axis=0) / np.sqrt(len(missed))})


# ------------------------- step 6: make tables -------------------------
# tables will be inputted into docs

def table_skill():
    # AUC vs lead times, full model vs persistence, also it saves every forecast for later steps
    base = load_base()
    onsets = find_ssw_onsets(base)
    P = pd.concat([run_lowo(lead, ABLATIONS["full"], base) for lead in LEADS])
    P.to_csv(f"{RESULTS}/pred_full.csv")
    T = pd.DataFrame([score(P[P.lead == lead], onsets) for lead in LEADS])
    T.to_csv(f"{RESULTS}/table1_skill.csv", index=False)
    print(T.round(3).to_string(index=False))
    return T


def table_onset():
    # share of SSWs warned + false alarm rate, aka model vs persistence
    onsets = find_ssw_onsets(load_base())
    P = pd.read_csv(f"{RESULTS}/pred_full.csv", index_col=0, parse_dates=True)
    rows = []
    for lead in LEADS:
        R = P[P.lead == lead]
        hit, fa = warnings(R, onsets)
        hit_p, fa_p = warnings(R, onsets, "persist")
        rows.append({"lead": lead, "hit_rate": hit.mean(), "false_alarm": fa,
                     "hit_rate_persist": hit_p.mean(), "false_alarm_persist": fa_p})
    T = pd.DataFrame(rows)
    T.to_csv(f"{RESULTS}/table2_onset.csv", index=False)
    print(T.round(3).to_string(index=False))
    return T


def table_ablation():
    # retrain model with one feature group removed, and then find AUC for each (separate each feature group by impact)
    base = load_base()
    onsets = find_ssw_onsets(base)
    rows = []
    for name, groups in ABLATIONS.items():
        for lead in LEADS:
            rows.append({**score(run_lowo(lead, groups, base), onsets), "ablation": name})
    T = pd.DataFrame(rows)
    T.to_csv(f"{RESULTS}/table3_ablation.csv", index=False)
    print(T.pivot(index="ablation", columns="lead", values="auc_model").round(3).to_string())
    return T


def table_composite(lead=20):
    # the ep flux before warned vs missed ssw's at the 20 day lead 
    base = load_base()
    onsets = find_ssw_onsets(base)
    P = pd.read_csv(f"{RESULTS}/pred_full.csv", index_col=0, parse_dates=True)
    hit, _ = warnings(P[P.lead == lead], onsets)
    C = ep_composite(base, onsets[hit], onsets[~hit])
    C.to_csv(f"{RESULTS}/composite.csv", index=False)
    pd.DataFrame({"onset": onsets, "warned": hit.astype(int)}).to_csv(f"{RESULTS}/event_classification.csv", index=False)
    # set bounds to filter dates
    early = (C.lag >= -25) & (C.lag <= -13)
    late = (C.lag >= -10) & (C.lag <= -1)
    print(f"{hit.sum()} warned, {(~hit).sum()} missed at {lead} d")
    print(f"days -25 to -13: warned {C.warned_mean[early].mean():+.2f}, missed {C.missed_mean[early].mean():+.2f}")
    print(f"days -10 to  -1: warned {C.warned_mean[late].mean():+.2f}, missed {C.missed_mean[late].mean():+.2f}")
    return C
    
def table_shap():
    # SHAP by feature group, model trained on all winters with no early stopping (table 4)
    import shap
    base = load_base()
    raw = base.drop(columns=["winter", "qbo"])
    all_days = np.ones(len(base), bool)
    a = pd.DataFrame({c: anomalize(raw[c], all_days) for c in raw.columns}, index=base.index)
    X = build_features(a, base)
    params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
    rows = []
    for lead in LEADS:
        y, _ = build_target(a[TARGET_COL], lead)
        in_winter = (base.index + pd.Timedelta(days=lead)).month.isin(WINTER_MONTHS)
        D = pd.concat([X, y.rename("y")], axis=1)[in_winter].dropna()
        model = xgb.XGBRegressor(random_state=0, **params).fit(D[X.columns], D["y"])
        # mean absolute SHAP per feature, summed within each group
        s = pd.Series(np.abs(shap.TreeExplainer(model).shap_values(D[X.columns])).mean(0), index=X.columns)
        rows.append({"lead": lead, **{g: s[s.index.str[0] == g].sum() for g in "FWQ"}, "top_feature": s.idxmax()})
    T = pd.DataFrame(rows)
    T.to_csv(f"{RESULTS}/table4_shap.csv", index=False)
    print(T.round(2).to_string(index=False))
    return T

# ------------------------- step 7: make figures -------------------------
# set the style up first, make style same for every figure
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "legend.fontsize": 6.5,
})
GRAY, LIGHT = "0.55", "#8da3b9" # reference lines, light bars/shading
WARNED, MISSED = "#2a7f62", "#7b3fa0" # green, purple
PC = ccrs.PlateCarree() # lat/lon coordinates for cartopy

def save(fig, name, dpi=300):
    fig.savefig(f"figures/{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# make the polar map used by figures 2 and 7

LEVELS = np.arange(-1200, 1201, 100) # anomaly shading levels (m)
# red-blue without the darkest end of spectrum
CMAP = LinearSegmentedColormap.from_list("rdbu_trim", plt.get_cmap("RdBu_r")(np.linspace(0.05, 0.95, 256))) 
_t = np.linspace(0, 2 * np.pi, 200)
CIRCLE = mpath.Path(np.column_stack([0.5 + 0.5 * np.cos(_t), 0.5 + 0.5 * np.sin(_t)])) # round map boundary


def z10(date):
    # 10 hPa geopotential height (m) for every day in the month containing date
    return xr.open_dataset(f"{RAW}/era5_{date.year}_{date.month:02d}.nc")["z"].sel(pressure_level=10) / 9.80665 # gravitational g

# should generate a circle POV of north pole with outline of continents
def polar_map(ax, date):
    # 1 polar map: height contours + anomaly shading for a single date, use cartopy
    z = z10(date).sel(valid_time=date)
    # climatology = mean height on this calendar day over 1979-2025
    # day=1 to avoid feb 29 errors
    allz = xr.concat([z10(date.replace(year=y, day=1)) for y in range(1979, 2026)], dim="valid_time")
    anom = z - allz.sel(valid_time=allz.valid_time.dt.day == date.day).mean("valid_time")
    
    # repeat the first longitude at 360 to prevent a seam in the map
    close = lambda a: xr.concat([a, a.isel(longitude=0).assign_coords(longitude=a.longitude[0] + 360)], dim="longitude")
    z, anom = close(z.sortby("longitude")), close(anom.sortby("longitude"))
    c = ax.contourf(z.longitude, z.latitude, anom, levels=LEVELS, cmap=CMAP, extend="both", transform=PC, zorder=1) # shading
    ax.gridlines(color="0.75", lw=0.4, ylocs=[30, 60], xlocs=[]).zorder = 2 # 30N and 60N circles
    
    ax.contour(z.longitude, z.latitude, z, levels=np.arange(28000, 32000, 400), colors="k", linewidths=0.5, transform=PC, zorder=3) # height contours
    ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="0.3", zorder=4)
    ax.set_extent([-180, 180, 20, 90], PC) # 20N up till the pole
    ax.set_boundary(CIRCLE, transform=ax.transAxes)
    ax.spines["geo"].set(edgecolor="0.3", linewidth=0.4)
    return c # return this so colorbar can use it

# red = high geopotential height bc hot weather, blue =  low vice versa
def polar_figure(n, name, draw_panels):
    # row of n polar maps with one shared colorbar underneath
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.4), subplot_kw={"projection": ccrs.NorthPolarStereo()})
    c = draw_panels(axes)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.2, wspace=0.05)
    fig.colorbar(c, cax=fig.add_axes([0.3, 0.08, 0.4, 0.03]), orientation="horizontal",
                 label="10 hPa height anomaly (m)", ticks=np.arange(-1200, 1201, 600))
    save(fig, name, dpi=150)


# set up figures in paper order -------->

# figure 1: mean EP flux anomaly for all 32 onsets
# appearance: as days approach onset, ep flux increases dramatically, then dies down after. plot should have some shading to indicate error +-
def fig_ep_intro():
    base, onsets = load_base(), pd.read_csv(f"{PROC}/ssw_onsets.csv", parse_dates=["onset"]).onset
    C = ep_composite(base, onsets, onsets)
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.fill_between(C.lag, C.warned_mean - C.warned_se, C.warned_mean + C.warned_se, color=LIGHT, alpha=0.4, lw=0) # +-1 se
    ax.plot(C.lag, C.warned_mean, color=WARNED, lw=1.2)
    ax.axhline(0, color=GRAY, lw=0.7, ls=":"); ax.axvline(0, color="k", lw=0.7) # zero anomaly, onset day
    ax.set(xlabel="Days relative to SSW onset", ylabel="100 hPa EP-flux anomaly (sd)", xlim=(-40, 10))
    fig.tight_layout(); save(fig, "fig1_epflux")

# figure 2: vortex at onset for one displacement and one split
# appearance: displacement fig with blue going South, split fig with blue on west and east (?) of red in the north pole
def fig_types(events={"2006-01-21": "Displacement", "2009-01-25": "Split"}): # most obvious/distinguishable dates
    def panels(axes):
        for ax, lab, (date, status) in zip(axes, "ab", events.items()):
            d = pd.Timestamp(date)
            c = polar_map(ax, d)
            ax.set_title(f"({lab}) {status}: {d.strftime('%b %-d, %Y')}", fontsize=7)
        return c
    polar_figure(len(events), "fig2_types", panels)


def fig_qbo():
    # figure 3: NOAA 30 hPa QBO index, westerly vs easterly months timeline
    # appearance: timeline should have alternating blue vs purple around every 28 months to indicate changing wind speed
    lines = urllib.request.urlopen("https://psl.noaa.gov/data/correlation/qbo.data").read().decode().splitlines()
    q = pd.DataFrame([l.split() for l in lines[1:] if len(l.split()) == 13], dtype=float).set_index(0).stack() # set one value per month
    q.index = pd.to_datetime([f"{int(y)}-{int(m)}" for y, m in q.index]); q[q < -90] = np.nan # make -999 = missing
    q = q["1979":"2025"]
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.fill_between(q.index, q, 0, where=q >= 0, color=LIGHT, lw=0, label="Westerly")
    ax.fill_between(q.index, q, 0, where=q < 0, color=MISSED, alpha=0.6, lw=0, label="Easterly")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="Year", ylabel="30 hPa equatorial zonal wind (m s$^{-1}$)", xlim=(q.index.min(), q.index.max()))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2); fig.tight_layout(); save(fig, "fig3_qbo") # place the legend above the plot


def fig_skill():
    # figure 4: (a) AUC vs lead for VortX and persistence, (b) ROC curve for each lead time
    # appearance: black line (VortX), grey line (persistence), VortX should be higher than persistence every lead time. ROC also above diagonal line
    onsets = find_ssw_onsets(load_base())
    T = pd.read_csv(f"{RESULTS}/table1_skill.csv")
    P = pd.read_csv(f"{RESULTS}/pred_full.csv", index_col=0, parse_dates=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, len(LEADS))) # use darker color
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})

    # (a) 
    a.axhline(0.5, color="0.6", lw=0.7, ls=":", label="Climatology") # no skill line
    a.plot(T.lead, T.auc_persist, color="0.45", lw=1, marker="s", ms=3, label="Persistence")
    a.plot(T.lead, T.auc_model, color="k", lw=1.2, marker="o", ms=3, label="VortX")
    a.set(xlabel="Lead time (days)", ylabel="AUC, SSW event", xticks=T.lead, ylim=(0.4, 1), yticks=np.arange(0.4, 1.01, 0.1))
    a.yaxis.grid(True, color="0.92", lw=0.5); a.set_axisbelow(True)
    h, l = a.get_legend_handles_labels(); a.legend(h[::-1], l[::-1], loc="upper right") # list vortx first on the legend
    a.text(-0.18, 1.02, "a)", transform=a.transAxes, fontsize=8, fontweight="bold")

    # (b)
    for lead, col in zip(LEADS, colors):
        R = P[P.lead == lead]
        match, keep, _ = ssw_events(R, onsets)
        fpr, tpr, _ = roc_curve(match.any(1)[keep], -R.pred[keep]) # negate it, lower wind = higher event score
        auc = T.loc[T.lead == lead, "auc_model"].iloc[0]
        b.plot(fpr, tpr, color=col, lw=1, label=f"{lead} d ({auc:.2f})")
    b.plot([0, 1], [0, 1], color="0.6", lw=0.7, ls="--") # no skill diagonal line
    b.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1), aspect="equal")
    b.legend(loc="lower right", title="Lead (AUC)", title_fontsize=7)
    b.text(-0.18, 1.02, "b)", transform=b.transAxes, fontsize=8, fontweight="bold")
    fig.tight_layout(); save(fig, "fig5_skill")


def fig_onset():
    # figure 6: % of SSWs warned (bars) and false alarm rate (lines); VortX vs persistence
    # appearance: blue bar (VortX) should be higher than grey (persistence). false alarm rates should be low, with persistence being a bit higher
    T = pd.read_csv(f"{RESULTS}/table2_onset.csv")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(T.lead - 0.8, T.hit_rate, width=1.6, color=LIGHT, label="SSWs warned, VortX") # bars side by side
    ax.bar(T.lead + 0.8, T.hit_rate_persist, width=1.6, color="#d3dde8", label="SSWs warned, persistence")
    ax.plot(T.lead, T.false_alarm, color=WARNED, lw=1.4, marker="o", ms=3, label="False-alarm rate, VortX")
    ax.plot(T.lead, T.false_alarm_persist, color=MISSED, lw=1.4, marker="s", ms=3, ls="--", label="False-alarm rate, persistence")
    for x, y in zip(T.lead, T.hit_rate): # % label above each VortX bar
        ax.text(x - 0.8, y + 0.02, f"{100*y:.0f}%", ha="center", fontsize=6.5)
    ax.set(xlabel="Lead time (days)", ylabel="Fraction of SSWs warned / false-alarm rate", xticks=T.lead, ylim=(0, 1))
    ax.yaxis.grid(True, color="0.92", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    fig.tight_layout(); save(fig, "fig6_onset")


def fig_composite(lead=20):
    # figure 7: EP flux before onset, SSWs warned vs missed at the 20 day lead
    # appearance: missed SSW should not spike until extremely close to onset, warned should be high prior to SSW
    C = pd.read_csv(f"{RESULTS}/composite.csv")
    E = pd.read_csv(f"{RESULTS}/event_classification.csv")
    n = {"warned": E.warned.sum(), "missed": (E.warned == 0).sum()} # counts for the legend
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.axvspan(-lead - 3, -lead + 3, color="0.9", lw=0) # gray band = when the 20 day forecast was issued
    for col, color in [("warned", WARNED), ("missed", MISSED)]:
        m, se = C[f"{col}_mean"], C[f"{col}_se"]
        ax.fill_between(C.lag, m - se, m + se, color=color, alpha=0.15, lw=0)  # +-1 se
        ax.plot(C.lag, m, color=color, lw=1.2, label=f"{col.capitalize()} at {lead} d (n = {n[col]})")
    ax.axhline(0, color=GRAY, lw=0.7, ls=":") # zero anomaly
    ax.axvline(0, color="k", lw=0.7) # onset day
    ax.set(xlabel="Days relative to SSW onset", ylabel="100 hPa EP-flux anomaly (sd)", xlim=(C.lag.min(), C.lag.max()), ylim=(-1, 2))
    ax.legend(loc="upper left")
    fig.tight_layout(); save(fig, "fig7_composite")


def fig_maps(events={"1988-03-14": "Warned", "2009-01-25": "Missed"}, lead=20):
    # figure 8: vortex on the forecast day and at onset, one warned and one missed SSW
    # appearance: compare 2 geopotential maps before & after 2 different onsets. the distinction should be obvious
    def panels(axes):
        labels = iter("abcdefgh")
        for i, (onset, status) in enumerate(events.items()):
            on = pd.Timestamp(onset)
            issue = on - pd.Timedelta(days=lead) # day the 20 day forecast was made
            polar_map(axes[2 * i], issue)
            c = polar_map(axes[2 * i + 1], on)
            axes[2 * i].set_title(f"({next(labels)}) {status}: issued {issue.strftime('%b %-d, %Y')}", fontsize=7)
            axes[2 * i + 1].set_title(f"({next(labels)}) {status}: onset {on.strftime('%b %-d, %Y')}", fontsize=7)
        return c
    polar_figure(2 * len(events), "fig8_maps", panels)


def make_figures():
    # all paper figs except for the one generated in draw.io
    fig_ep_intro()
    fig_types()
    fig_qbo()
    fig_skill()
    fig_onset()
    fig_composite()
    fig_maps()