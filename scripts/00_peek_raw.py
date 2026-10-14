from pathlib import Path

RAW = Path("data/raw")

files = sorted(p for p in RAW.rglob("*") if p.is_file())

print(f"{len(files)} files found\n")

for p in files[:60]:
    print(f"{p.relative_to(RAW)}   {p.stat().st_size/1e6:.2f} MB")

if len(files) > 60:
    print(f"... and {len(files) - 60} more")


# Find CSV files
csvs = [p for p in files if p.suffix.lower() == ".csv"]

special = [
    p for p in csvs
    if "static" in p.name.lower()
    or "mapping" in p.name.lower()
]

others = [p for p in csvs if p not in special][:4]


# Print first 10 raw lines
for p in special + others:

    print("\n" + "=" * 80)
    print(p.relative_to(RAW))

    with open(p, "r", encoding="utf-8", errors="replace") as fh:

        for i, line in enumerate(fh):

            if i >= 10:
                break

            print(line.rstrip()[:220])