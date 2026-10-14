from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw/scada_2019")
INTERIM = Path("data/interim/scada_2019.parquet")
OUT = Path("reports")
OUT.mkdir(exist_ok=True)
YEAR_START = pd.Timestamp("2019-01-01")
WIN = pd.Timedelta("10min")
N_WIN = 52560
pd.set_option("display.width", 220)


def clean(c):
    return c.lstrip("# ").strip()


def header_row(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if "," in line:
                return i


def read_status(n):
    f = next(RAW.glob(f"Status_Kelmarsh_{n}_*.csv"))
    s = pd.read_csv(f, skiprows=header_row(f))
    s.columns = [clean(c) for c in s.columns]
    s["start"] = pd.to_datetime(s["Timestamp start"])
    end = s["Timestamp end"].astype(str).str.strip()
    s["end"] = pd.to_datetime(end.where(end != "-"), errors="coerce")
    s["turbine"] = n
    return s


st = pd.concat([read_status(n) for n in range(1, 7)], ignore_index=True)
st["Message"] = st["Message"].fillna("(none)")
iv = st[st["end"].notna() & (st["end"] > st["start"])][
    ["turbine", "start", "end", "Status", "Code", "Message"]]

sc = pd.read_parquet(INTERIM, columns=["timestamp", "turbine", "power_kw",
                                       "wind_speed_ms", "gear_oil_temp_c"])
sc["idx"] = ((sc["timestamp"] - YEAR_START) // WIN).astype(int)

# ---------- Reference power curve from windows untouched by any explicit interval ----------
touched = {n: np.zeros(N_WIN, dtype=bool) for n in range(1, 7)}
for r in iv.itertuples(index=False):
    i0 = max(int((r.start - YEAR_START) // WIN), 0)
    i1 = min(int((r.end - YEAR_START) // WIN), N_WIN - 1)
    touched[r.turbine][i0:i1 + 1] = True
sc["touched"] = [touched[t][i] for t, i in zip(sc["turbine"], sc["idx"])]
sc["wbin"] = np.floor(sc["wind_speed_ms"] * 2) / 2
ok = (~sc["touched"]) & sc["power_kw"].notna() & sc["wind_speed_ms"].notna()
ref = sc[ok].groupby("wbin")["power_kw"].median()
sc["ref_kw"] = sc["wbin"].map(ref)
sc["ratio"] = sc["power_kw"] / sc["ref_kw"]

# ---------- Part 1: behaviour per status code ----------
recs = []
for r in iv.itertuples(index=False):
    first = max(int(np.ceil((r.start - YEAR_START) / WIN)), 0)
    last = min(int(np.floor((r.end - YEAR_START) / WIN)) - 1, N_WIN - 1)
    for i in range(first, last + 1):
        recs.append((r.turbine, i, r.Status, r.Code, r.Message))
cov = pd.DataFrame(recs, columns=["turbine", "idx", "Status", "Code", "Message"])
print(f"Fully covered (entry, window) pairs: {len(cov)}")

hw = cov.merge(sc[["turbine", "idx", "ratio", "ref_kw"]], on=["turbine", "idx"], how="left")
hw = hw[(hw["ref_kw"] >= 300) & hw["ratio"].notna()].copy()
stop_keys = set(zip(cov.loc[cov["Status"] == "Stop", "turbine"],
                    cov.loc[cov["Status"] == "Stop", "idx"]))
hw["in_stop"] = [(t, i) in stop_keys for t, i in zip(hw["turbine"], hw["idx"])]
hw["nonprod"] = hw["ratio"] < 0.1
hw["np_no_stop"] = hw["nonprod"].astype(float).where(~hw["in_stop"])

tab = hw.groupby(["Status", "Code", "Message"]).agg(
    windows=("ratio", "size"),
    turbines=("turbine", "nunique"),
    mean_ratio=("ratio", "mean"),
    pct_not_producing=("nonprod", "mean"),
    windows_without_stop=("in_stop", lambda s: int((~s).sum())),
    pct_not_producing_excl_stop=("np_no_stop", "mean"),
).reset_index()
tab["pct_not_producing"] *= 100
tab["pct_not_producing_excl_stop"] *= 100
tab.to_csv(OUT / "code_behaviour_2019.csv", index=False)

show = tab[tab["windows"] >= 5].copy()
show["Message"] = show["Message"].astype(str).str.slice(0, 34)
print("\n=== 1. Behaviour in high wind, by status code (windows fully inside the entry) ===")
print("Overlapping entries each count the same window; 'excl_stop' removes windows also inside a Stop.")
print(show.sort_values("windows", ascending=False).head(45).round(2).to_string(index=False))

# ---------- Part 2: Turbine 1 oil-filter episodes vs gear-oil temperature ----------
p = sc.pivot(index="timestamp", columns="turbine", values="power_kw")
g = sc.pivot(index="timestamp", columns="turbine", values="gear_oil_temp_c")
similar = ((p > 800) & (p < 1800)).all(axis=1)
diffs = {}
for t in range(1, 7):
    peer = g.drop(columns=t).median(axis=1)
    diffs[t] = (g[t] - peer)[similar]

periods = [
    ("before warning (to 24 Feb)", "2019-01-01", "2019-02-24 16:46"),
    ("warning 1 (24 Feb-4 Apr)", "2019-02-24 16:46", "2019-04-04 12:36"),
    ("gap (4-9 Apr)", "2019-04-04 12:36", "2019-04-09 10:09"),
    ("warning 2 (9 Apr-21 May)", "2019-04-09 10:09", "2019-05-21 10:07"),
    ("21-28 May", "2019-05-21 10:07", "2019-05-28 20:55"),
    ("warning 3 (28-30 May)", "2019-05-28 20:55", "2019-05-30 07:35"),
    ("after 30 May", "2019-05-30 07:35", "2020-01-01"),
]
rows = []
for label, a, b in periods:
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    row = {"period": label}
    for t in range(1, 7):
        s = diffs[t]
        s = s[(s.index >= a) & (s.index < b)].dropna()
        row[f"T{t}"] = round(s.median(), 2) if len(s) else np.nan
        if t == 1:
            row["n_rows_T1"] = len(s)
    rows.append(row)
print("\n=== 2. Median gear-oil temperature difference vs median of the other five (deg C) ===")
print("Rows used: all six turbines between 800 and 1800 kW at the same timestamp.")
print(pd.DataFrame(rows).to_string(index=False))

x = sc.loc[sc["power_kw"] > 1500, "gear_oil_temp_c"].dropna()
print("\nGear-oil temperature when power > 1500 kW (all turbines):")
print(x.quantile([0.5, 0.9, 0.99, 1.0]).round(1).to_string())
print(f"Share of rows at or above 59 C: {(x >= 59).mean() * 100:.1f}%")