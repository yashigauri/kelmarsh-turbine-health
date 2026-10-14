from pathlib import Path
import pandas as pd

RAW = Path("data/raw/scada_2019")
INTERIM = Path("data/interim/scada_2019.parquet")
pd.set_option("display.width", 200)


def clean(c):
    return c.lstrip("# ").strip()


def header_row(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if "," in line:
                return i


# ---------- 1. Which turbines carry the long warnings? ----------
frames = []
for n in range(1, 7):
    f = next(RAW.glob(f"Status_Kelmarsh_{n}_*.csv"))
    s = pd.read_csv(f, skiprows=header_row(f))
    s.columns = [clean(c) for c in s.columns]
    s["turbine"] = n
    frames.append(s)
st = pd.concat(frames, ignore_index=True)
cols = ["turbine", "Timestamp start", "Timestamp end", "Duration", "Code", "Message"]
print("=== 1. Long-running warnings (codes 1860 and 5000) ===")
print(st[st["Code"].isin([1860, 5000])][cols].to_string(index=False))

# ---------- 2. Turbine 1 vs its neighbours, 5 to 21 May ----------
df = pd.read_parquet(INTERIM, columns=["timestamp", "turbine", "wind_speed_ms", "power_kw"])
start, end = pd.Timestamp("2019-05-05 03:20"), pd.Timestamp("2019-05-21 09:50")
inblk = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
inblk["day"] = inblk["timestamp"].dt.date
print("\n=== 2a. Mean wind speed by day (m/s), turbines 1 to 6 ===")
print(inblk.pivot_table(index="day", columns="turbine", values="wind_speed_ms",
                        aggfunc="mean").round(1).to_string())
print("\n=== 2b. Mean power by day (kW), turbines 1 to 6 ===")
print(inblk.pivot_table(index="day", columns="turbine", values="power_kw",
                        aggfunc="mean").round(0).to_string())

power = df.pivot(index="timestamp", columns="turbine", values="power_kw")
wind = df.pivot(index="timestamp", columns="turbine", values="wind_speed_ms")
peer_power = power.drop(columns=1).mean(axis=1)
peer_wind = wind.drop(columns=1).mean(axis=1)
in_block = (power.index >= start) & (power.index <= end)
windy = (peer_wind > 5) & (peer_power > 300)
print("\n=== 2c. Windy rows where the other five turbines were producing ===")
for label, mask in [("inside the May block ", in_block), ("outside the May block", ~in_block)]:
    sel = windy & mask
    t1 = power.loc[sel, 1]
    print(f"{label}: {int(sel.sum())} rows | T1 mean power {t1.mean():.0f} kW | "
          f"peers mean {peer_power[sel].mean():.0f} kW | "
          f"T1 below 10% of peers: {(t1 < 0.1 * peer_power[sel]).mean() * 100:.1f}% of rows")