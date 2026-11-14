import json
import pandas as pd

features = pd.read_csv(
    "fused_features.csv",
    parse_dates=["ts"]
)

features = features[
    (features["ts"] >= "2026-05-04T00:00:00Z") &
    (features["ts"] < "2026-05-06T00:00:00Z")
].copy()

# Compare each tractor with the other two at each minute.
fleet = features.pivot(
    index="ts",
    columns="tractor_id",
    values=["mean_cn0", "mean_sats", "link_ok_rate"]
)

candidates = []

for tractor in ["T1", "T2", "T3"]:
    others = [t for t in ["T1", "T2", "T3"] if t != tractor]

    own_cn0 = fleet["mean_cn0"][tractor]
    peer_cn0 = fleet["mean_cn0"][others].mean(axis=1)

    own_sats = fleet["mean_sats"][tractor]
    peer_sats = fleet["mean_sats"][others].mean(axis=1)

    own_link = fleet["link_ok_rate"][tractor]


    # Persistent machine-specific degradation
    # Thresholds are experimental.

    raw_flag = (
        (peer_cn0 - own_cn0 >= 0.55) &
        (peer_sats - own_sats >= 0.25) &
        (peer_cn0 >= 42) &
        (own_link >= 0.95)
    ).fillna(False)

    # Detect persistent degradation over 21 minutes
    flag = (
        raw_flag.astype(float)
        .rolling(21, center=True, min_periods=11)
        .mean()
        >= 0.50
    ).fillna(False)

    starts = flag & ~flag.shift(fill_value=False)
    groups = starts.cumsum()


    for _, interval in flag[flag].groupby(groups[flag]):
        start = interval.index[0]
        end = interval.index[-1] + pd.Timedelta(minutes=1)

        duration = (end - start).total_seconds() / 60

        if duration < 30:
            continue


        period = (fleet.index >= start) & (fleet.index < end)

        cn0_gap = (peer_cn0 - own_cn0)[period].mean()
        sats_gap = (peer_sats - own_sats)[period].mean()

        candidates.append({
            "start": start.isoformat(),
            "end": end.isoformat(),
            "tractors": [tractor],
            "class": "EQUIPMENT_FAULT",
            "action": "SERVICE_MACHINE_OTHERS_CONTINUE",
            "confidence": 0.55,
            "evidence": [
                f"{tractor}: sustained receiver degradation from "
                f"{start.isoformat()} to {end.isoformat()}",
                f"Mean C/N0 deficit versus peers: {cn0_gap:.2f} dB-Hz",
                f"Mean satellite-count deficit versus peers: {sats_gap:.2f}",
                "Correction link remained mostly healthy; "
                "machine-specific antenna or receiver issue suspected."
            ]
        })

print("New equipment fault candidates:", len(candidates))

for c in candidates:
    print(
        c["tractors"],
        c["start"],
        "to",
        c["end"]
    )

# Keep existing predictions unchanged and evaluate additions.
with open("merged_development.json", encoding="utf-8") as f:
    submission = json.load(f)

submission["detections"].extend(candidates)
submission["detections"].sort(key=lambda x: x["start"])

with open("fault_test_development.json", "w", encoding="utf-8") as f:
    json.dump(submission, f, indent=2)

print("Saved fault_test_development.json")
