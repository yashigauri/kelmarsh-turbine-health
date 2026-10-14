from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw/scada_2019")
OUT = Path("reports")
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 220)

NEED = {
    "Date and time": "timestamp",
    "Power (kW)": "power",
    "Gear oil temperature (°C)": "oil_temp",
    "Gear oil inlet temperature (°C)": "oil_inlet_temp",
    "Gear oil inlet pressure (bar)": "oil_in_press",
    "Gear oil pump pressure (bar)": "oil_pump_press",
}
NAMES = {"oil_temp": "d_oil_temp", "oil_inlet_temp": "d_inlet_temp",
         "oil_in_press": "d_inlet_press", "oil_pump_press": "d_pump_press"}

WARN = [("2019-02-24 16:46", "2019-04-04 12:36"),
        ("2019-04-09 10:09", "2019-05-21 10:07"),
        ("2019-05-28 20:55", "2019-05-30 07:35")]
WARN = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in WARN]


def clean(c):
    return c.lstrip("# ").strip()


def header_row(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if "," in line:
                return i


frames = []
for n in range(1, 7):
    f = next(RAW.glob(f"Turbine_Data_Kelmarsh_{n}_*.csv"))
    d = pd.read_csv(f, skiprows=header_row(f),
                    usecols=lambda c: clean(c) in NEED, low_memory=False)
    d.columns = [clean(c) for c in d.columns]
    d = d.rename(columns=NEED)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d["turbine"] = n
    frames.append(d)
df = pd.concat(frames, ignore_index=True)


def wide(col):
    return df.pivot(index="timestamp", columns="turbine", values=col)


P = wide("power")
inband = (P > 800) & (P < 1800)


def peer_diff(col, target=1, min_peers=3):
    W = wide(col).where(inband)
    peers = W.drop(columns=target)
    ok = (peers.notna().sum(axis=1) >= min_peers) & W[target].notna()
    return (W[target] - peers.median(axis=1))[ok]


def in_warn(ts):
    m = np.zeros(len(ts), dtype=bool)
    for a, b in WARN:
        m |= (ts >= a) & (ts < b)
    return m


def warn_fraction(a, b):
    tot = (b - a).total_seconds()
    ov = sum(max((min(b, y) - max(a, x)).total_seconds(), 0) for x, y in WARN)
    return round(ov / tot, 2)


# ---------- Levels: do the pressure signals look sensible? ----------
print("=== 1. In-band (800-1800 kW) median level per turbine ===")
lev = pd.DataFrame({col: wide(col).where(inband).median() for col in NAMES})
print(lev.round(2).to_string())
print("\nIn-band percentiles, all turbines pooled:")
pool = pd.DataFrame({col: wide(col).where(inband).stack() for col in ["oil_in_press", "oil_pump_press"]})
print(pool.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).round(1).to_string())

# ---------- Warning vs non-warning, T1 minus peers ----------
print("\n=== 2. Turbine 1 minus peer median: inside vs outside the oil-filter warnings ===")
rows = []
for col in NAMES:
    d = peer_diff(col)
    w = in_warn(d.index)
    for label, sel in [("inside warnings", d[w]), ("outside warnings", d[~w])]:
        rows.append({"signal": col, "period": label, "rows": len(sel),
                     "median": sel.median(), "p25": sel.quantile(0.25),
                     "p75": sel.quantile(0.75)})
print(pd.DataFrame(rows).round(2).to_string(index=False))

# ---------- Weekly timeline ----------
parts = {}
for col, nm in NAMES.items():
    d = peer_diff(col)
    parts[nm] = d.resample("W").median()
    if col == "oil_temp":
        parts["n_rows"] = d.resample("W").count()
tab = pd.concat(parts, axis=1)
tab["warn_frac"] = [warn_fraction(i - pd.Timedelta(days=6), i + pd.Timedelta(days=1))
                    for i in tab.index]
tab.index = tab.index.date
print("\n=== 3. Weekly median of (T1 minus peers) ===")
print(tab.round(2).to_string())
tab.to_csv(OUT / "t1_oil_weekly_vs_peers.csv")