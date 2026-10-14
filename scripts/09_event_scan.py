from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
INTERIM = Path("data/interim")
OUT = Path("reports")
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 400)


def clean(c):
    return c.lstrip("# ").strip()


def header_row(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if "," in line:
                return i


frames = []
for folder in sorted(RAW.glob("scada_*")):
    for f in sorted(folder.glob("Status_Kelmarsh_*.csv")):
        s = pd.read_csv(f, skiprows=header_row(f))
        s.columns = [clean(c) for c in s.columns]
        s["turbine"] = int(f.name.split("_")[2])
        frames.append(s)
st = pd.concat(frames, ignore_index=True)
st["start"] = pd.to_datetime(st["Timestamp start"])
end = st["Timestamp end"].astype(str).str.strip()
st["end"] = pd.to_datetime(end.where(end != "-"), errors="coerce")
st["hours"] = (st["end"] - st["start"]).dt.total_seconds() / 3600
st["Message"] = st["Message"].fillna("(none)").astype(str)
print(f"Status files read: {len(frames)}; entries: {len(st)}; "
      f"span {st['start'].min()} to {st['start'].max()}")

# ---------- 1. Long warnings and stops ----------
cols = ["turbine", "start", "end", "days", "Status", "Code", "Message"]
long_ev = st[st["Status"].isin(["Warning", "Stop"]) & (st["hours"] >= 24)].copy()
long_ev["days"] = (long_ev["hours"] / 24).round(1)
long_ev = long_ev.sort_values(["turbine", "start"])
print("\n=== 1. Warning/Stop entries lasting 24 h or longer ===")
print(long_ev[cols].to_string(index=False))
long_ev[cols].to_csv(OUT / "long_episodes_all_years.csv", index=False)

# ---------- 2. Oil filter entries ----------
of = st[st["Message"].str.contains("oil filter", case=False)].sort_values(["turbine", "start"])
print("\n=== 2. Every 'oil filter' entry, all years ===")
print(of[["turbine", "start", "end", "Status", "Code", "Message"]].to_string(index=False))

# ---------- 3. Drivetrain / hydraulic related messages ----------
kw = "oil|filter|gear|bearing|temperature|pressure|vibration|hydraulic|pump"
h = st[st["Message"].str.contains(kw, case=False, regex=True)]
tab = (h.groupby(["Code", "Message"])
         .agg(entries=("start", "size"), turbines=("turbine", "nunique"),
              total_hours=("hours", "sum"), first=("start", "min"), last=("start", "max"))
         .round(1).sort_values("entries", ascending=False))
print("\n=== 3. Drivetrain/hydraulic-related messages (top 30 by entries) ===")
print(tab.head(30).to_string())

# ---------- 4. Farm-wide power gaps ----------
print("\n=== 4. Farm-wide power gaps (all six turbines missing), longest 5 per year ===")
for f in sorted(INTERIM.glob("scada_*.parquet")):
    d = pd.read_parquet(f, columns=["timestamp", "turbine", "power_kw"])
    P = d.pivot(index="timestamp", columns="turbine", values="power_kw")
    allmiss = P.isna().all(axis=1)
    run = (allmiss != allmiss.shift()).cumsum()
    g = (P.index.to_series()[allmiss].groupby(run[allmiss])
         .agg(start="min", end="max", rows="size").sort_values("rows", ascending=False))
    print(f"\n{f.stem}: {int(allmiss.sum())} rows ({allmiss.mean() * 100:.1f}%) "
          f"missing on all six turbines")
    print(g.head(5).to_string(index=False))

# ---------- 5. Out-of-range rows in 2016 and 2017 ----------
print("\n=== 5. Out-of-range rows in 2016 and 2017 ===")
LIMITS = {"wind_speed_ms": (0, 40), "power_kw": (-100, 2200), "rotor_rpm": (0, 20)}
for tag in ["scada_2016", "scada_2017"]:
    p = INTERIM / f"{tag}.parquet"
    if not p.exists():
        continue
    d = pd.read_parquet(p)
    lim = dict(LIMITS)
    lim.update({c: (-40, 150) for c in d.columns if c.endswith("_temp_c")})
    bad = d[d["out_of_range"]]
    print(f"\n{tag}: {len(bad)} rows")
    for _, r in bad.iterrows():
        viol = {c: round(r[c], 1) for c, (lo, hi) in lim.items()
                if pd.notna(r[c]) and not (lo <= r[c] <= hi)}
        print(r["timestamp"], f"T{int(r['turbine'])}", viol)