from pathlib import Path
import re
import pandas as pd

RAW = Path("data/raw/scada_2019")
OUT = Path("reports")
OUT.mkdir(exist_ok=True)

WANTED = [
    "Date and time", "Power (kW)", "Energy Export (kWh)",
    "Energy Export counter (kWh)", "Wind speed (m/s)",
    "Wind speed Sensor 1 (m/s)", "Wind speed Sensor 2 (m/s)",
    "Turbine Power setpoint (kW)",
]


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


# ---------- A. All six turbines, selected columns ----------
rows = []
for n in range(1, 7):
    f = next(RAW.glob(f"Turbine_Data_Kelmarsh_{n}_*.csv"))
    s = next(RAW.glob(f"Status_Kelmarsh_{n}_*.csv"))
    header_total = None
    with open(s, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.search(r"Sum production:\s*(\d+)\s*kWh", line)
            if m:
                header_total = int(m.group(1))
                break
    df = read_cols(f, set(WANTED)).reindex(columns=WANTED)
    ctr = df["Energy Export counter (kWh)"].dropna()
    rows.append({
        "turbine": n,
        "rows": len(df),
        "duplicate_timestamps": int(df["Date and time"].duplicated().sum()),
        "header_total_kWh": header_total,
        "sum_energy_export_kWh": round(df["Energy Export (kWh)"].sum()),
        "counter_diff_kWh": round(ctr.max() - ctr.min()) if len(ctr) else None,
        "power_x_8760_kWh": round(df["Power (kW)"].mean() * 8760),
        "power_missing_rows": int(df["Power (kW)"].isna().sum()),
        "energy_over_342_rows": int((df["Energy Export (kWh)"] > 342).sum()),
        "max_energy_10min": round(df["Energy Export (kWh)"].max(), 1),
        "wind_s1_over_s2": round(df["Wind speed Sensor 1 (m/s)"].mean()
                                 / df["Wind speed Sensor 2 (m/s)"].mean(), 3),
        "setpoint_below_rated_pct": round(
            (df["Turbine Power setpoint (kW)"] < 2050).mean() * 100, 1),
    })
summary = pd.DataFrame(rows)
print(summary.T.to_string(header=False))
summary.to_csv(OUT / "all_turbines_2019_checks.csv", index=False)

# ---------- B. Turbine 1, all columns ----------
f1 = next(RAW.glob("Turbine_Data_Kelmarsh_1_*.csv"))
full = read_cols(f1)
t = full["Date and time"]
nn = full.drop(columns="Date and time").notna()
pct = nn.mean() * 100
print(f"\nCompletely empty columns: {int((pct == 0).sum())} of {len(pct)}")
print(f"Fully populated columns:  {int((pct == 100).sum())}")

partial = []
for c in pct[(pct > 0) & (pct < 100)].index:
    valid = t[nn[c]]
    partial.append({"column": c, "non_null_pct": round(pct[c], 1),
                    "first_valid": valid.min(), "last_valid": valid.max()})
pd.DataFrame(partial).sort_values("non_null_pct").to_csv(
    OUT / "turbine1_partial_columns.csv", index=False)
print(f"Saved {len(partial)} partially populated columns to reports/")

# ---------- C. The 4.6% hole ----------
probe = "Wind speed, Standard deviation (m/s)"
miss = full[probe].isna()
run_id = (miss != miss.shift()).cumsum()
runs = full.loc[miss].groupby(run_id[miss]).agg(
    start=("Date and time", "min"), end=("Date and time", "max"),
    n=("Date and time", "size"))
print(f"\n'{probe}' missing in {int(miss.sum())} rows, {len(runs)} separate gaps. Longest 8:")
print(runs.sort_values("n", ascending=False).head(8).to_string())

over = full.loc[full["Energy Export (kWh)"] > 342,
                ["Date and time", "Energy Export (kWh)", "Power (kW)", "Capacity factor"]]
print(f"\nRows with Energy Export > 342 kWh: {len(over)} (first 10)")
print(over.head(10).to_string())

# ---------- D. Status log ----------
s1 = next(RAW.glob("Status_Kelmarsh_1_*.csv"))
st = pd.read_csv(s1, skiprows=header_row(s1))
st.columns = [clean(c) for c in st.columns]
st["has_end"] = st["Timestamp end"].astype(str).str.strip() != "-"
print("\nStatus type vs has an end timestamp:")
print(pd.crosstab(st["Status"], st["has_end"]))
fo = st[st["IEC category"] == "Forced outage"][
    ["Timestamp start", "Timestamp end", "Duration", "Status", "Code", "Message"]]
print("\nForced outage entries (first 15):")
print(fo.head(15).to_string())