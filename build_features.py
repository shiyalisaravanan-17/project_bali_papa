import pandas as pd

df = pd.read_csv(
    "data/farm/tractor_telemetry.csv.gz",
    parse_dates=["ts"]
)

df = df.sort_values(["tractor_id", "ts"])

df["abs_cross_track_error"] = df["cross_track_error_m"].abs()
df["rtk_fixed"] = (df["fix_state"] == "RTK_FIXED").astype(int)
df["link_ok"] = df["base_link_ok"].astype(int)

features = (
    df.groupby([
        "tractor_id",
        pd.Grouper(key="ts", freq="1min")
    ])
    .agg(
        mean_cn0=("mean_cn0_dbhz", "mean"),
        min_cn0=("min_cn0_dbhz", "min"),
        mean_sats=("num_sats", "mean"),
        min_sats=("num_sats", "min"),
        mean_error=("abs_cross_track_error", "mean"),
        max_error=("abs_cross_track_error", "max"),
        rtk_fixed_rate=("rtk_fixed", "mean"),
        link_ok_rate=("link_ok", "mean"),
        samples=("ts", "count")
    )
    .reset_index()
)

features.to_csv("minute_features.csv", index=False)

print("Feature extraction successful!")
print("Total one-minute records:", len(features))
print(features.head(10).to_string(index=False))
