from pathlib import Path
import pandas as pd

RAW = Path("data/raw/scada_2019")
OUT = Path("reports")
OUT.mkdir(exist_ok=True)


def read_greenbyte(path):
    """Skip the '#' comment block and find the real header row."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = []
        header_idx = None

        for i, line in enumerate(fh):
            lines.append(line.rstrip("\n"))

            if "," in line:
                header_idx = i
                break

    comments = lines[:header_idx]

    df = pd.read_csv(
        path,
        skiprows=header_idx,
        low_memory=False
    )

    df.columns = [
        c.lstrip("# ").strip()
        for c in df.columns
    ]

    return comments, df


# Find turbine 1 files
turb_file = next(
    RAW.glob("Turbine_Data_Kelmarsh_1_*.csv")
)

status_file = next(
    RAW.glob("Status_Kelmarsh_1_*.csv")
)


# ============================================================
# SCADA FILE
# ============================================================

comments, df = read_greenbyte(turb_file)

print(turb_file.name)

print("\n".join(comments))

print("\nShape (rows, columns):", df.shape)

print("First column:", df.columns[0])

print(
    "Timestamps (first 3, last 1):",
    df.iloc[:, 0].iloc[[0, 1, 2, -1]].tolist()
)

print("\nFirst 5 rows / first 8 columns:")
print(
    df.iloc[:5, :8].to_string()
)


# Column summary
summary = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "non_null_pct": (df.notna().mean() * 100).round(1),
})

num = df.select_dtypes("number")

summary = summary.join(
    num.agg(["min", "mean", "max"]).T.round(3)
)

summary.to_csv(
    OUT / "turbine1_2019_column_summary.csv"
)

print(
    f"\nSaved column summary ({len(summary)} columns) to reports/"
)


# ============================================================
# STATUS FILE
# ============================================================

comments_s, st = read_greenbyte(status_file)

print("\n" + status_file.name)

print("Shape:", st.shape)

print("\nFirst 12 status rows:")

print(
    st.head(12).to_string()
)


for col in [
    "Status",
    "IEC category",
    "Service contract category"
]:

    if col in st.columns:

        print(f"\n{col}:")

        print(
            st[col]
            .value_counts(dropna=False)
            .head(15)
        )