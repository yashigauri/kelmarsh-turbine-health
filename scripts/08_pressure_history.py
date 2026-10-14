from pathlib import Path
import pandas as pd

INTERIM = Path("data/interim")
OUT = Path("reports")
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 300)

files = sorted(INTERIM.glob("scada_*.parquet"))
print("Loading:", [f.stem for f in files])
cols = ["timestamp", "turbine", "power_kw", "gear_oil_temp_c",
        "gear_oil_pump_press", "gear_oil_inlet_press", "pressure_valid"]
df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
for c in ["gear_oil_pump_press", "gear_oil_inlet_press"]:
    df.loc[~df["pressure_valid"], c] = float("nan")

P = df.pivot(index="timestamp", columns="turbine", values="power_kw")
inband = (P > 800) & (P < 1800)
turbines = list(P.columns)


def diffs(col):
    W = df.pivot(index="timestamp", columns="turbine", values=col).where(inband)
    out = {}
    for t in turbines:
        peers = W.drop(columns=t)
        ok = (peers.notna().sum(axis=1) >= 3) & W[t].notna()
        out[t] = (W[t] - peers.median(axis=1))[ok]
    return out


def monthly(col):
    d = diffs(col)
    tab = pd.DataFrame({f"T{t}": s.resample("MS").median() for t, s in d.items()})
    tab["rows_T1"] = d[1].resample("MS").count()
    tab.index = tab.index.strftime("%Y-%m")
    return tab


pump = monthly("gear_oil_pump_press")
print("\n=== 1. Monthly median pump pressure: turbine minus peer median (in-band 800-1800 kW) ===")
print(pump.round(1).to_string())
pump.to_csv(OUT / "pump_pressure_monthly_all_years.csv")

print("\n=== 2. Monthly median gear-oil temperature: turbine minus peer median ===")
print(monthly("gear_oil_temp_c").round(2).to_string())

tcols = [f"T{t}" for t in turbines]
chg = pump[tcols].diff().stack()
big = chg.abs().sort_values(ascending=False).head(12).index
print("\n=== 3. Largest month-to-month changes in pump-pressure difference ===")
print(chg.loc[big].round(1).to_string())

d1 = diffs("gear_oil_pump_press")[1]
wk = d1.resample("W").agg(["median", "count"])
wk.to_csv(OUT / "t1_pump_pressure_weekly_all_years.csv")
print("\n=== 4. Turbine 1 weekly, Sep 2018 to Mar 2019 ===")
print(wk[(wk.index >= "2018-09-01") & (wk.index <= "2019-03-31")].round(1).to_string())

dd = d1.resample("D").agg(["median", "count"])
print("\n=== 5. Turbine 1 daily, 22 May to 5 June 2019 ===")
print(dd.loc["2019-05-22":"2019-06-05"].round(1).to_string())