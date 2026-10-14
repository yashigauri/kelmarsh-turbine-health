from pathlib import Path
import pandas as pd

RAW = Path("data/raw/scada_2019")
OUT = Path("reports")
OUT.mkdir(exist_ok=True)

COLS = ["Date and time", "Power (kW)", "Energy Export (kWh)", "Wind speed (m/s)",
        "Wind speed Sensor 1 (m/s)", "Wind speed Sensor 2 (m/s)",
        "Turbine Power setpoint (kW)"]


def clean(c):
    return c.lstrip("# ").strip()


def header_row(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if "," in line:
                return i


def read_cols(path, wanted=None):
    use = None if wanted is None else (lambda c: clean(c) in wanted)
    df = pd.read_csv(path, skiprows=header_row(path), usecols=use, low_memory=False)
    df.columns = [clean(c) for c in df.columns]
    df["Date and time"] = pd.to_datetime(df["Date and time"])
    return df


frames = []
for n in range(1, 7):
    f = next(RAW.glob(f"Turbine_Data_Kelmarsh_{n}_*.csv"))
    d = read_cols(f, set(COLS)).reindex(columns=COLS)
    d["turbine"] = n
    frames.append(d)
allt = pd.concat(frames, ignore_index=True)
allt["kwh_from_power"] = allt["Power (kW)"] / 6

# ---------- Q1: Energy Export vs power-implied energy ----------
print("=== Q1: Energy Export vs Power/6 ===")
hi = allt[allt["Power (kW)"] > 1500].copy()
hi["minute"] = hi["Date and time"].dt.minute
by_slot = hi.groupby("minute")[["Energy Export (kWh)", "kwh_from_power"]].mean()
by_slot["ratio"] = by_slot.iloc[:, 0] / by_slot.iloc[:, 1]
print("By minute-of-hour (rows with Power > 1500 kW):")
print(by_slot.round(3).to_string())


def agg_err(d, rule):
    x = d.set_index("Date and time")[["Energy Export (kWh)", "kwh_from_power"]].dropna()
    r = x.resample(rule).sum()
    return round((r.iloc[:, 0] - r.iloc[:, 1]).abs().mean() / r.iloc[:, 1].mean() * 100, 2)


err = pd.DataFrame({
    rule: {n: agg_err(allt[allt.turbine == n], rule) for n in range(1, 7)}
    for rule in ["10min", "30min", "1h", "1D"]
})
print("\nMean absolute difference (% of mean energy) by aggregation level:")
print(err.to_string())

# ---------- Q2: does power track the setpoint? ----------
print("\n=== Q2: power vs setpoint, by wind speed ===")
w = allt.dropna(subset=["Wind speed (m/s)", "Power (kW)", "Turbine Power setpoint (kW)"]).copy()
w["wind_bin"] = w["Wind speed (m/s)"].round().clip(upper=16)
w["sp_below_rated"] = w["Turbine Power setpoint (kW)"] < 2050
w["near_sp"] = w["sp_below_rated"] & (w["Power (kW)"] >= 0.95 * w["Turbine Power setpoint (kW)"])
tab = w.groupby("wind_bin").agg(
    rows=("near_sp", "size"),
    mean_power=("Power (kW)", "mean"),
    mean_setpoint=("Turbine Power setpoint (kW)", "mean"),
    pct_sp_below_rated=("sp_below_rated", "mean"),
    pct_power_near_sp=("near_sp", "mean"),
)
tab[["pct_sp_below_rated", "pct_power_near_sp"]] *= 100
print(tab.round(1).to_string())

# ---------- Q3: wind sensors ----------
print("\n=== Q3: Sensor 1 / Sensor 2 ratio by month ===")
allt["month"] = allt["Date and time"].dt.month
m = allt.groupby(["turbine", "month"])[["Wind speed Sensor 1 (m/s)", "Wind speed Sensor 2 (m/s)"]].mean()
print((m.iloc[:, 0] / m.iloc[:, 1]).unstack().round(3).to_string())
for s in ["Wind speed Sensor 1 (m/s)", "Wind speed Sensor 2 (m/s)"]:
    diff = (allt["Wind speed (m/s)"] - allt[s]).abs().groupby(allt["turbine"]).mean().round(3)
    print(f"Mean |main wind - {s}|:", diff.to_dict())

# ---------- Q4: timestamp alignment (Turbine 1, long manual stop on 4 Nov) ----------
print("\n=== Q4: timestamp alignment around the 4 Nov stop (11:15:48 to 16:54:34) ===")
t1 = allt[allt.turbine == 1].set_index("Date and time")
for a, b in [("2019-11-04 10:40", "2019-11-04 11:50"), ("2019-11-04 16:30", "2019-11-04 17:20")]:
    print(t1.loc[a:b, ["Wind speed (m/s)", "Power (kW)"]].round(1).to_string())
    print()

# ---------- Q5: the 16-day block (Turbine 1, all columns) ----------
print("=== Q5: the 5 to 21 May block ===")
f1 = next(RAW.glob("Turbine_Data_Kelmarsh_1_*.csv"))
full = read_cols(f1)
t = full["Date and time"]
blk = (t >= "2019-05-05 03:20") & (t <= "2019-05-21 09:50")
nn = full.drop(columns="Date and time").notna()
pres = pd.DataFrame({
    "present_in_block_pct": (nn[blk].mean() * 100).round(1),
    "present_outside_pct": (nn[~blk].mean() * 100).round(1),
})
pres.to_csv(OUT / "turbine1_block_presence.csv")
lost = pres[(pres.present_in_block_pct < 5) & (pres.present_outside_pct > 90)]
print(f"{len(lost)} normally-populated columns vanish during the block:")
for c in lost.index:
    print("  -", c)
keep = ["Power (kW)", "Wind speed (m/s)", "Gear oil temperature (°C)",
        "Front bearing temperature (°C)", "Stator temperature 1 (°C)", "Rotor speed (RPM)"]
print("\nKey signals during vs outside the block:")
print(pres.loc[[c for c in keep if c in pres.index]].to_string())