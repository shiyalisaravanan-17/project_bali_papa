import json
import pandas as pd

df = pd.read_csv("fused_features.csv", parse_dates=["ts"])
df = df[
    (df["ts"] >= "2026-05-04T00:00:00Z") &
    (df["ts"] < "2026-05-06T00:00:00Z")
].copy()

wide = df.pivot(
    index="ts",
    columns="tractor_id",
    values=["mean_error", "mean_cn0", "rtk_fixed_rate"]
)

tractors = ["T1", "T2", "T3"]
candidates = []

for tractor in tractors:
    peers = [t for t in tractors if t != tractor]

    own_error = wide["mean_error"][tractor]
    peer_error = wide["mean_error"][peers].median(axis=1)

    excess = own_error - peer_error

    # Detect sustained tractor-specific guidance deviation.
    raw = (
        (excess >= 0.45) &
        (own_error >= 0.65)
    ).fillna(False)

    smooth = (
        raw.astype(float)
        .rolling(7, center=True, min_periods=4)
        .mean() >= 0.60
    ).fillna(False)

    groups = (
        smooth != smooth.shift(fill_value=False)
    ).cumsum()

    for _, segment in smooth[smooth].groupby(groups[smooth]):
        start = segment.index[0]
        end = segment.index[-1] + pd.Timedelta(minutes=1)

        duration = (end - start).total_seconds() / 60

        if duration < 8:
            continue

        period = (wide.index >= start) & (wide.index < end)

        mean_excess = excess[period].mean()
        mean_error = own_error[period].mean()

        candidates.append({
            "start": start.isoformat(),
            "end": end.isoformat(),
            "tractors": [tractor],
            "class": "SPOOFING",
            "action": "HALT_CROSSCHECK_AND_REPORT",
            "confidence": 0.60,
            "evidence": [
                "Sustained guidance deviation relative to peer tractors",
                f"Mean guidance error: {mean_error:.2f} m",
                f"Mean error excess relative to peers: {mean_excess:.2f} m",
                "Peer-relative anomaly suggests a tractor-specific "
                "position-integrity problem; spoofing requires cross-check"
            ]
        })

print("New spoofing candidates:", len(candidates))
for c in candidates:
    print(c["tractors"], c["start"], "to", c["end"])

with open("best_development_so_far.json", encoding="utf-8") as f:
    result = json.load(f)

result["detections"].extend(candidates)
result["detections"].sort(key=lambda x: x["start"])

with open("spoofing_test_development.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print("Saved spoofing_test_development.json")
