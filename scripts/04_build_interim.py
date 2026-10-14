from pathlib import Path
import re
import pandas as pd

RAW = Path("data/raw")
INTERIM = Path("data/interim")
REPORTS = Path("reports")
INTERIM.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(exist_ok=True)

RATED_KW = 2050.0
SKIP = {"scada_2016", "scada_2017", "scada_2018", "scada_2019"}  # already built

COLUMN_MAP = {
    "Date and time": "timestamp",
    "Wind speed (m/s)": "wind_speed_ms",
    "Wind speed Sensor 1 (m/s)": "wind_s1_ms",
    "Wind speed Sensor 2 (m/s)": "wind_s2_ms",
    "Wind speed, Standard deviation (m/s)": "wind_speed_std_ms",
    "Wind direction (°)": "wind_dir_deg",
    "Nacelle position (°)": "nacelle_pos_deg",
    "Power (kW)": "power_kw",
    "Power, Standard deviation (kW)": "power_std_kw",
    "Energy Export (kWh)": "energy_export_kwh",
    "Rotor speed (RPM)": "rotor_rpm",
    "Generator RPM (RPM)": "generator_rpm",
    "Blade angle (pitch position) A (°)": "pitch_a_deg",
    "Blade angle (pitch position) B (°)": "pitch_b_deg",
    "Blade angle (pitch position) C (°)": "pitch_c_deg",
    "Gear oil temperature (°C)": "gear_oil_temp_c",
    "Gear oil inlet temperature (°C)": "gear_oil_inlet_temp_c",
    "Gear oil inlet pressure (bar)": "gear_oil_inlet_press",
    "Gear oil pump pressure (bar)": "gear_oil_pump_press",
    "Front bearing temperature (°C)": "front_bearing_temp_c",
    "Rear bearing temperature (°C)": "rear_bearing_temp_c",
    "Rotor bearing temp (°C)": "rotor_bearing_temp_c",
    "Generator bearing front temperature (°C)": "gen_bearing_front_temp_c",
    "Generator bearing rear temperature (°C)": "gen_bearing_rear_temp_c",
    "Stator temperature 1 (°C)": "stator_temp_c",
    "Nacelle ambient temperature (°C)": "nacelle_ambient_temp_c",
    "Nacelle temperature (°C)": "nacelle_temp_c",
    "Ambient temperature (converter) (°C)": "converter_ambient_temp_c",
    "Transformer temperature (°C)": "transformer_temp_c",
    "Hub temperature (°C)": "hub_temp_c",
    "Metal particle count": "metal_particle_count",
    "Drive train acceleration (mm/ss)": "drivetrain_accel",
    "Tower Acceleration X (mm/ss)": "tower_accel_x",
    "Tower Acceleration y (mm/ss)": "tower_accel_y",
    "Grid frequency (Hz)": "grid_freq_hz",
    "Turbine Power setpoint (kW)": "power_setpoint_kw",
    "Potential power default PC (kW)": "bench_potential_power_kw",
    "Energy Theoretical (kWh)": "bench_energy_theoretical_kwh",
    "Lost Production to Downtime (kWh)": "bench_lost_downtime_kwh",
    "Lost Production to Performance (kWh)": "bench_lost_performance_kwh",
}

KEY_TEMPS = ["gear_oil_temp_c", "front_bearing_temp_c", "rear_bearing_temp_c",
             "gen_bearing_front_temp_c", "gen_bearing_rear_temp_c", "stator_temp_c"]
PITCH = ["pitch_a_deg", "pitch_b_deg", "pitch_c_deg"]
AVAIL = ["wind_speed_ms", "power_kw", "gear_oil_temp_c", "gear_oil_inlet_temp_c",
         "gear_oil_pump_press", "gear_oil_inlet_press", "front_bearing_temp_c",
         "gen_bearing_front_temp_c", "stator_temp_c", "pitch_a_deg", "rotor_rpm",
         "power_setpoint_kw"]

# Wide sanity bounds, used only to FLAG rows (provisional)
BOUNDS = {"wind_speed_ms": (0, 40), "power_kw": (-100, 2200), "rotor_rpm": (0, 20)}
for c in COLUMN_MAP.values():
    if c.endswith("_temp_c"):
        BOUNDS[c] = (-40, 150)


def clean(c):
    return c.lstrip("# ").strip()


def header_row(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if "," in line:
                return i


def header_total(status_path):
    with open(status_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.search(r"Sum production:\s*(\d+)\s*kWh", line)
            if m:
                return int(m.group(1))
    return None


def read_turbine(path, turbine_id):
    df = pd.read_csv(path, skiprows=header_row(path),
                     usecols=lambda c: clean(c) in COLUMN_MAP, low_memory=False)
    df.columns = [clean(c) for c in df.columns]
    df = df.rename(columns=COLUMN_MAP)
    absent = [v for v in COLUMN_MAP.values() if v not in df.columns]
    for c in absent:
        df[c] = float("nan")
    df = df[list(COLUMN_MAP.values())]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.insert(1, "turbine", turbine_id)
    return df, absent


def add_flags(df):
    # Impossible sentinel temperatures (e.g. -273.2 C) become NaN, with a flag
    sentinel = pd.Series(False, index=df.index)
    for c in [c for c in df.columns if c.endswith("_temp_c")]:
        bad = df[c] < -100
        sentinel |= bad
        df.loc[bad, c] = float("nan")
    df["sentinel_replaced"] = sentinel

    df["energy_from_power_kwh"] = df["power_kw"] / 6
    df["power_missing"] = df["power_kw"].isna()
    df["energy_export_over_physical"] = df["energy_export_kwh"] > RATED_KW / 6
    df["temps_available"] = df[KEY_TEMPS].notna().all(axis=1)
    df["pitch_available"] = df[PITCH].notna().all(axis=1)

    # Pressure is trusted only if it is present and, where the year logs gear-oil
    # temperature at all, that temperature was logged in the same row.
    # (In May 2019 Turbine 1 pressure collapsed while temperatures were blank.)
    present = df["gear_oil_pump_press"].notna() & df["gear_oil_inlet_press"].notna()
    if df["gear_oil_temp_c"].notna().mean() >= 0.5:
        df["pressure_valid"] = present & df["gear_oil_temp_c"].notna()
    else:
        df["pressure_valid"] = present

    oor = pd.Series(False, index=df.index)
    for col, (lo, hi) in BOUNDS.items():
        oor |= (df[col] < lo) | (df[col] > hi)
    df["out_of_range"] = oor
    df["near_setpoint_limit"] = (
        (df["power_setpoint_kw"] < RATED_KW)
        & (df["power_kw"] >= 0.95 * df["power_setpoint_kw"])
    )
    return df


for folder in sorted(RAW.glob("scada_*")):
    tag = folder.name
    if tag in SKIP:
        continue
    frames, rows, avail, absent_sets = [], [], {}, []
    print(f"\n##### {tag} #####")
    for n in range(1, 7):
        tf = next(folder.glob(f"Turbine_Data_Kelmarsh_{n}_*.csv"))
        sf = next(folder.glob(f"Status_Kelmarsh_{n}_*.csv"))
        df, absent = read_turbine(tf, n)
        absent_sets.append(set(absent))
        df = add_flags(df)
        total = header_total(sf)
        if total is None:  # fall back to the meter-based yearly total
            total = round(df["energy_export_kwh"].sum())
        from_power = df["energy_from_power_kwh"].sum()
        rows.append({
            "turbine": n,
            "rows": len(df),
            "first_timestamp": str(df["timestamp"].min()),
            "power_missing": int(df["power_missing"].sum()),
            "sentinel_rows": int(df["sentinel_replaced"].sum()),
            "out_of_range": int(df["out_of_range"].sum()),
            "temps_available_pct": round(df["temps_available"].mean() * 100, 1),
            "pressure_valid_pct": round(df["pressure_valid"].mean() * 100, 1),
            "header_total_kwh": total,
            "from_power_kwh": round(from_power),
            "diff_pct": round((from_power - total) / total * 100, 3) if total else None,
        })
        avail[n] = (df[AVAIL].notna().mean() * 100).round(0)
        frames.append(df)

    common = set.intersection(*absent_sets)
    anyabs = set.union(*absent_sets)
    print("Mapped columns absent from ALL six files:", sorted(common) or "none")
    if anyabs - common:
        print("Absent from SOME files only:", sorted(anyabs - common))

    summary = pd.DataFrame(rows)
    print(summary.T.to_string(header=False))
    summary.to_csv(REPORTS / f"cleaning_summary_{tag}.csv", index=False)

    av = pd.DataFrame(avail).T
    av.index.name = "turbine"
    print("\nSignal availability (% of rows populated):")
    print(av.to_string())
    av.to_csv(REPORTS / f"signal_availability_{tag}.csv")

    out = INTERIM / f"{tag}.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
    print(f"\nWrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")