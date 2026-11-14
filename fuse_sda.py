import json
import pandas as pd
from pathlib import Path

DATA = Path("data/sda")

def load(name, key):
    with open(DATA / name, encoding="utf-8") as f:
        return pd.DataFrame(json.load(f)[key])

def attach_asof(left, right, columns):
    right = right.copy()
    right["ts"] = pd.to_datetime(right["ts"], utc=True)
    right = right.sort_values("ts")
    return pd.merge_asof(
        left.sort_values("ts"),
        right[["ts"] + columns],
        on="ts",
        direction="backward"
    )

features = pd.read_csv(
    "minute_features.csv",
    parse_dates=["ts"]
)
features["ts"] = pd.to_datetime(features["ts"], utc=True)

# Regional scintillation
s4 = load("scintillation_series.json", "series")
features = attach_asof(features, s4, ["s4"])

# Geomagnetic measurements
with open(DATA / "sgi_indices.json", encoding="utf-8") as f:
    geomagnetic = json.load(f)

# Merge Kp measurements
kp = pd.DataFrame(geomagnetic["kp"])
features = attach_asof(features, kp, ["kp"])

# Merge Dst measurements
dst = pd.DataFrame(geomagnetic["dst"])
features = attach_asof(features, dst, ["dst_nT"])

# Fleet-wide signal health
features["signal_bad"] = (
    (features["mean_cn0"] < 35) |
    (features["mean_sats"] < 8) |
    (features["rtk_fixed_rate"] < 0.5)
)

fleet = (
    features.groupby("ts")["signal_bad"]
    .sum()
    .rename("affected_fleet_count")
    .reset_index()
)
features = features.merge(fleet, on="ts", how="left")

# RFI reports: preserve the report arrival time.
rfi = load("rfi_reports.json", "reports")

if not rfi.empty:
    rfi["ts"] = pd.to_datetime(rfi["ts"], format="ISO8601", utc=True)
    rfi = rfi.sort_values("ts")
    rfi = rfi.rename(columns={
        "ts": "rfi_report_ts",
        "confidence": "rfi_confidence",
        "id": "rfi_id"
    })

    features = pd.merge_asof(
        features.sort_values("ts"),
        rfi[["rfi_report_ts", "rfi_confidence", "rfi_id"]],
        left_on="ts",
        right_on="rfi_report_ts",
        direction="backward"
    )

    features["rfi_age_minutes"] = (
        (features["ts"] - features["rfi_report_ts"])
        .dt.total_seconds() / 60
    )

    features["recent_rfi_report"] = (
        features["rfi_age_minutes"].between(0, 45)
    ).fillna(False)

# Save fused feature dataset.
features = features.sort_values(["tractor_id", "ts"])
features.to_csv("fused_features.csv", index=False)

print("SDA fusion successful!")
print("Total rows:", len(features))
print("Columns:", features.columns.tolist())
print()
print(features[
    ["tractor_id", "ts", "mean_cn0", "s4", "kp",
     "dst_nT", "affected_fleet_count",
     "recent_rfi_report"]
].head(10).to_string(index=False))
