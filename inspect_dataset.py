
from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"C:\Users\SANKET\Desktop\AI_project\Dataset")
FILES = [
    "fire_archive_SV-C2_820222.csv",
    "fire_nrt_SV-C2_820222.csv",
]

for filename in FILES:
    path = DATA_DIR / filename

    print("\n" + "=" * 65)
    print(f"FILE: {filename}")
    print("=" * 65)

    if not path.exists():
        print("File not found:", path)
        continue

    df = pd.read_csv(path, low_memory=False)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nColumn names:")
    print(df.columns.tolist())

    # Dates
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    print("\nDate range:")
    print("Start:", df["acq_date"].min())
    print("End:  ", df["acq_date"].max())
    print("Unique dates:", df["acq_date"].nunique())

    # Geographic coverage
    for col in ["latitude", "longitude", "frp", "brightness"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\nGeographic bounds:")
    print("Latitude:", df["latitude"].min(), "to", df["latitude"].max())
    print("Longitude:", df["longitude"].min(), "to", df["longitude"].max())

    # Confidence and day/night
    for col in ["confidence", "daynight", "type", "satellite"]:
        if col in df.columns:
            print(f"\n{col} distribution:")
            print(df[col].value_counts(dropna=False).head(15).to_string())

    # Missing values
    print("\nMissing values:")
    print(df.isna().sum().to_string())

    # Duplicate records
    print("\nExact duplicate rows:", df.duplicated().sum())

    # FRP statistics
    if "frp" in df.columns:
        print("\nFire Radiative Power (FRP) statistics:")
        print(df["frp"].describe().round(2).to_string())

print("\nDataset audit complete. Original CSV files were not modified.")
