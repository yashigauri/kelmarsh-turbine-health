from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw/scada_2019")
INTERIM = Path("data/interim/scada_2019.parquet")
OUT = Path("reports")
OUT.mkdir(exist_ok=True)
YEAR_END = pd.Timestamp("2020-01-01")


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


status = {n: read_status(n) for n in range(1, 7)}

# ---------- Part A: timestamp alignment using high-output stops ----------
print("=== A. Timestamp alignment test ===")
sc = pd.read_parquet(INTERIM, columns=["timestamp", "turbine", "power_kw"])
power = {n: g.set_index("timestamp")["power_kw"] for n, g in sc.groupby("turbine")}

recs = []
for n, st in status.items():
    ev = st[(st["Status"] == "Stop") & st["end"].notna()].copy()
    ev["dur_min"] = (ev["end"] - ev["start"]).dt.total_seconds() / 60
    ev = ev[ev["dur_min"] >= 40]
    p = power[n]
    for _, r in ev.iterrows():
        a = r["start"].floor("10min")
        f = (r["start"] - a).total_seconds() / 600

        def at(off):
            return p.get(a + pd.Timedelta(minutes=off), np.nan)

        base = np.nanmean([at(-30), at(-20), at(-10)])
        recs.append({"turbine": n, "start": r["start"], "dur_min": round(r["dur_min"]),
                     "message": r["Message"], "frac": round(f, 2), "base_kw": round(base),
                     "P_row_floor": at(0), "P_row_floor_plus10": at(10), "P_plus30": at(30)})
ev_df = pd.DataFrame(recs)
ok = ev_df[(ev_df["base_kw"] > 500) & (ev_df["P_plus30"] < 0.3 * ev_df["base_kw"])].copy()
ok["ratio_floor_row"] = (ok["P_row_floor"] / ok["base_kw"]).round(2)
ok["ratio_next_row"] = (ok["P_row_floor_plus10"] / ok["base_kw"]).round(2)
print(f"Stops of 40+ min: {len(ev_df)}; usable (started at >500 kW and really stopped): {len(ok)}")
if len(ok):
    print(ok[["turbine", "start", "dur_min", "message", "frac", "base_kw",
              "ratio_floor_row", "ratio_next_row"]].head(25).to_string(index=False))
    mid = ok[(ok["frac"] >= 0.2) & (ok["frac"] <= 0.8)]
    print(f"\nEvents starting 20-80% into a 10-min bin: {len(mid)}")
    print("Mean ratio, row labelled at floor(start):     ", round(mid["ratio_floor_row"].mean(), 2))
    print("Mean ratio, row labelled at floor(start)+10:  ", round(mid["ratio_next_row"].mean(), 2))
    print("Reading: window-START labels -> ~0.5 and ~0.0;  window-END labels -> ~1.0 and ~0.5")

# ---------- Part B: carry-forward timeline structure ----------
print("\n=== B. Status log as a carry-forward timeline (% of year by IEC category) ===")
rows = []
for n, st in status.items():
    s = st.sort_values("start", kind="stable").reset_index(drop=True)
    nxt = s["start"].shift(-1).fillna(YEAR_END)
    s["hours"] = (nxt - s["start"]).dt.total_seconds() / 3600
    unknown_h = (s["start"].min() - pd.Timestamp("2019-01-01")).total_seconds() / 3600
    pct = (s.groupby(s["IEC category"].fillna("(NaN)"))["hours"].sum() / 8760 * 100).round(1)
    d = pct.to_dict()
    d.update({"unknown_before_first_entry": round(unknown_h / 8760 * 100, 1),
              "rows": len(s), "same_start_dupes": int(s["start"].duplicated().sum())})
    d["turbine"] = n
    rows.append(d)
print(pd.DataFrame(rows).set_index("turbine").T.to_string())

# ---------- Part C: does the timeline match the physics? ----------
print("\n=== C. SCADA behaviour by IEC category (carry-forward state) ===")
states = pd.concat([status[n][["turbine", "start", "IEC category"]] for n in status])
states = states.rename(columns={"IEC category": "iec"}).sort_values("start")
sc_all = pd.read_parquet(INTERIM, columns=["timestamp", "turbine", "power_kw", "wind_speed_ms"])
for label, shift in [("probe at timestamp+5min (assumes timestamp = window START)", 5),
                     ("probe at timestamp-5min (assumes timestamp = window END)", -5)]:
    q = sc_all.copy()
    q["probe"] = q["timestamp"] + pd.Timedelta(minutes=shift)
    q = q.sort_values("probe")
    m = pd.merge_asof(q, states, left_on="probe", right_on="start", by="turbine",
                      direction="backward")
    m["iec"] = m["iec"].fillna("(none / before first entry)")
    t = m.groupby("iec").agg(rows=("power_kw", "size"),
                             mean_power_kw=("power_kw", "mean"),
                             mean_wind_ms=("wind_speed_ms", "mean"))
    t["pct_rows"] = t["rows"] / len(m) * 100
    print("\n" + label)
    print(t.round(1).sort_values("rows", ascending=False).to_string())

# ---------- Part D: Turbine 4 out-of-range rows ----------
print("\n=== D. Out-of-range rows (provisional bounds) ===")
full = pd.read_parquet(INTERIM)
bad = full[full["out_of_range"]]
print(bad.dropna(axis=1, how="all").T.to_string())

# ---------- Part E: single-turbine vs farm-wide stops ----------
print("\n=== E. How many turbines does each Stop event overlap? ===")
stops = pd.concat([st[(st["Status"] == "Stop") & st["end"].notna()] for st in status.values()],
                  ignore_index=True)[["turbine", "start", "end", "Code", "Message", "IEC category"]]
n_turb = []
for _, r in stops.iterrows():
    hit = stops[(stops["start"] <= r["end"]) & (stops["end"] >= r["start"])]
    n_turb.append(hit["turbine"].nunique())
stops["turbines_affected"] = n_turb
print("All stops:")
print(stops["turbines_affected"].value_counts().sort_index().to_string())
fo = stops[stops["IEC category"] == "Forced outage"]
print("\nForced-outage stops:")
print(fo["turbines_affected"].value_counts().sort_index().to_string())
wide = stops[stops["turbines_affected"] >= 4]
print("\nStops overlapping 4+ turbines, by message (top 10):")
print(wide.groupby("Message").size().sort_values(ascending=False).head(10).to_string())
stops.to_csv(OUT / "stops_2019_with_overlap.csv", index=False)