from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw/scada_2019")
INTERIM = Path("data/interim/scada_2019.parquet")
OUT = Path("reports")
OUT.mkdir(exist_ok=True)
YEAR_START = pd.Timestamp("2019-01-01")
YEAR_END = pd.Timestamp("2020-01-01")
WIN = pd.Timedelta("10min")


def clean(c):
    return c.lstrip("# ").strip()


def header_row(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if "," in line:
                return i


def read_status(n):
    f = next(RAW.glob(f"Status_Kelmarsh_{n}_*.csv"))
    st = pd.read_csv(f, skiprows=header_row(f))
    st.columns = [clean(c) for c in st.columns]
    st["start"] = pd.to_datetime(st["Timestamp start"])
    end = st["Timestamp end"].astype(str).str.strip()
    st["end"] = pd.to_datetime(end.where(end != "-"), errors="coerce")
    st["turbine"] = n
    return st


st = pd.concat([read_status(n) for n in range(1, 7)], ignore_index=True)
st["iec"] = st["IEC category"].fillna("(none)")
st["Message"] = st["Message"].fillna("(none)")
st = st.sort_values(["turbine", "start"], kind="stable").reset_index(drop=True)
st["minutes"] = (st["end"] - st["start"]).dt.total_seconds() / 60
st["cf_hours"] = (
    st.groupby("turbine")["start"].shift(-1).fillna(YEAR_END) - st["start"]
).dt.total_seconds() / 3600

pd.set_option("display.width", 200)

# ---------- 1. Catalog ----------
keys = ["Status", "Code", "Message", "iec"]
a = st.groupby(keys).agg(entries=("start", "size"), turbines=("turbine", "nunique"))
b = (st[st["end"].notna()].groupby(keys)
     .agg(with_end=("minutes", "size"), total_hours=("minutes", lambda x: x.sum() / 60),
          median_min=("minutes", "median")))
cat = a.join(b).fillna({"with_end": 0, "total_hours": 0}).reset_index()
cat["total_hours"] = cat["total_hours"].round(1)
cat["median_min"] = cat["median_min"].round(1)
cat.to_csv(OUT / "status_catalog_2019.csv", index=False)
show = cat.copy()
show["Message"] = show["Message"].astype(str).str.slice(0, 36)
print("=== 1a. Catalog: top 30 by total hours (entries with an end time) ===")
print(show.sort_values("total_hours", ascending=False).head(30).to_string(index=False))
print("\n=== 1b. Catalog: top 20 by number of entries ===")
print(show.sort_values("entries", ascending=False).head(20).to_string(index=False))
print(f"\nDistinct (status, code, message, IEC) combinations: {len(cat)}")

# ---------- 2. Mystery groups ----------
print("\n=== 2a. Entries with a missing IEC category (carry-forward hours) ===")
nn = st[st["iec"] == "(none)"]
print(nn.groupby("turbine")["cf_hours"].agg(entries="size", hours="sum").round(1).to_string())
top = (nn.groupby(["Status", "Message"])["cf_hours"].agg(entries="size", hours="sum")
       .round(1).sort_values("hours", ascending=False).head(12))
print(top.to_string())

print("\n=== 2b. Turbine 1: 'Out of Environmental Specification' ===")
t1 = st[(st["turbine"] == 1) & (st["iec"] == "Out of Environmental Specification")].copy()
print(f"Entries: {len(t1)}, carry-forward hours: {t1['cf_hours'].sum():.1f}")
t1["Message"] = t1["Message"].astype(str).str.slice(0, 36)
print(t1.sort_values("cf_hours", ascending=False)[
    ["start", "Status", "Code", "Message", "cf_hours"]].head(10).round(1).to_string(index=False))
print("Hours by month:")
print(t1.groupby(t1["start"].dt.month)["cf_hours"].sum().round(1).to_string())

# ---------- 3. Interval coverage and physical validation ----------
iv = st[st["end"].notna()].copy()


def group_name(status, iec):
    if status == "Stop":
        return "Stop: " + iec
    if status == "Informational":
        return "Info w/ end: " + iec
    return status


iv["grp"] = [group_name(s, i) for s, i in zip(iv["Status"], iv["iec"])]
groups = sorted(iv["grp"].unique())


def merge(pairs):
    out = []
    for s, e in sorted(pairs):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def coverage(intervals, n_win):
    cov = np.zeros(n_win)
    for s, e in intervals:
        i0 = max(int((s - YEAR_START) // WIN), 0)
        i1 = min(int((e - YEAR_START) // WIN), n_win - 1)
        for i in range(i0, i1 + 1):
            ws = YEAR_START + i * WIN
            ov = (min(e, ws + WIN) - max(s, ws)).total_seconds()
            if ov > 0:
                cov[i] += ov / 600
    return np.clip(cov, 0, 1)


sc = pd.read_parquet(INTERIM, columns=["timestamp", "turbine", "power_kw", "wind_speed_ms"])
cov = pd.DataFrame(0.0, index=sc.index, columns=groups)
for n, g in sc.groupby("turbine"):
    g = g.sort_values("timestamp")
    expected = pd.date_range(YEAR_START, periods=len(g), freq="10min")
    assert (g["timestamp"].values == expected.values).all(), "timestamps are not a regular grid"
    for grp in groups:
        sub = iv[(iv["turbine"] == n) & (iv["grp"] == grp)]
        cov.loc[g.index, grp] = coverage(merge(list(zip(sub["start"], sub["end"]))), len(g))

disturb = [g for g in groups if g != "Info w/ end: Full Performance"]
none_cov = cov[disturb].sum(axis=1) == 0

sc["wbin"] = np.floor(sc["wind_speed_ms"] * 2) / 2
ref_ok = none_cov & sc["power_kw"].notna() & sc["wind_speed_ms"].notna()
ref = sc[ref_ok].groupby("wbin")["power_kw"].median()
sc["ref_kw"] = sc["wbin"].map(ref)

hw = sc[(sc["ref_kw"] >= 300) & sc["power_kw"].notna()].copy()
hw["ratio"] = hw["power_kw"] / hw["ref_kw"]
hw["orig"] = hw.index
print(f"\n=== 3. High-wind rows used for validation (reference power >= 300 kW): {len(hw)} ===")

rows = []
base = hw[~(cov.loc[hw.index, disturb].sum(axis=1) > 0)]
rows.append(("(no explicit interval)", len(base), base["ratio"].mean(),
             (base["ratio"] < 0.1).mean() * 100))
for grp in groups:
    sel = hw[cov.loc[hw.index, grp] >= 0.99]
    if len(sel):
        rows.append((grp, len(sel), sel["ratio"].mean(), (sel["ratio"] < 0.1).mean() * 100))
print("\n3a. Power vs reference when a window is >=99% covered by an explicit interval:")
print(pd.DataFrame(rows, columns=["group", "rows_high_wind", "mean_power_ratio",
                                  "pct_not_producing"]).round(2).to_string(index=False))

states = st[["turbine", "start", "iec"]].sort_values("start")
q = hw[["orig", "timestamp", "turbine", "ratio"]].copy()
q["probe"] = q["timestamp"] + pd.Timedelta("5min")
q = q.sort_values("probe")
m = pd.merge_asof(q, states, left_on="probe", right_on="start", by="turbine",
                  direction="backward")
m["iec"] = m["iec"].fillna("(before first entry)")
print("\n3b. Same check using carry-forward IEC category (probe at timestamp + 5 min):")
t2 = m.groupby("iec").agg(rows_high_wind=("ratio", "size"), mean_power_ratio=("ratio", "mean"),
                          pct_not_producing=("ratio", lambda r: (r < 0.1).mean() * 100))
print(t2.round(2).sort_values("rows_high_wind", ascending=False).to_string())

# ---------- 4. Head to head ----------
m = m.set_index("orig")
nonprod = hw.index[hw["ratio"] < 0.1]
explicit = cov.loc[nonprod, disturb].max(axis=1) >= 0.5
carry = m.loc[nonprod, "iec"] != "Full Performance"
print(f"\n=== 4. High-wind rows producing < 10% of reference: {len(nonprod)} ===")
print(f"Explained by an explicit interval covering >=50% of the window: {explicit.mean()*100:.1f}%")
print(f"Explained by a non-'Full Performance' carry-forward state:      {carry.mean()*100:.1f}%")
print(f"Explained by either:                                            {(explicit | carry).mean()*100:.1f}%")
un = nonprod[~(explicit | carry).values]
print(f"Explained by neither: {len(un)} rows")
if len(un):
    u = hw.loc[un]
    print("By turbine:")
    print(u["turbine"].value_counts().sort_index().to_string())
    print("Busiest dates:")
    print(u["timestamp"].dt.date.value_counts().head(10).to_string())