
import pandas as pd
import numpy as np
import json

DATA_FILE = "fused_features.csv"

df = pd.read_csv(DATA_FILE, parse_dates=["ts"])
df["ts"] = pd.to_datetime(df["ts"], utc=True)
df = df.sort_values(["ts", "tractor_id"])

# Initial candidate rules. These require training validation.
df["space_weather"] = (
    (df["s4"] >= 0.4) &
    (df["kp"] >= 5) &
    (df["affected_fleet_count"] >= 2)
)

df["jamming"] = (
    (df["mean_cn0"] < 36) &
    (df["affected_fleet_count"] >= 2) &
    (~df["space_weather"])
)

df["spoofing"] = (
    (df["max_error"] >= 1.0) &
    (df["mean_cn0"] >= 38) &
    (df["rtk_fixed_rate"] >= 0.8)
)

df["equipment_fault"] = (
    (df["link_ok_rate"] < 0.9) |
    (
        (df["mean_cn0"] < 35) &
        (df["affected_fleet_count"] == 1)
    )
)

# Multipath needs a separate one-second detector.
# We do not generate multipath predictions from minute means.

ACTION = {
    "SPACE_WEATHER": "HALT_OR_DEGRADED_MODE",
    "JAMMING": "HALT_AND_REPORT",
    "SPOOFING": "HALT_CROSSCHECK_AND_REPORT",
    "EQUIPMENT_FAULT": "SERVICE_MACHINE_OTHERS_CONTINUE",
}

RULES = {
    "SPACE_WEATHER": "space_weather",
    "JAMMING": "jamming",
    "SPOOFING": "spoofing",
    "EQUIPMENT_FAULT": "equipment_fault",
}

START = pd.Timestamp("2026-05-04T00:00:00Z")
END = pd.Timestamp("2026-05-06T00:00:00Z")

def detect_episodes(data, cause, column):
    results = []

    for tractor, group in data.groupby("tractor_id"):
        group = group.sort_values("ts").copy()
        active = group[group[column]]

        if active.empty:
            continue

        # Consecutive one-minute samples form an episode.
        gaps = active["ts"].diff().dt.total_seconds().fillna(60)
        blocks = (gaps > 60).cumsum()

        for _, episode in active.groupby(blocks):
            if len(episode) < 3:
                continue

            start = episode["ts"].iloc[0]
            end = episode["ts"].iloc[-1] + pd.Timedelta(minutes=1)

            end = min(end, END)

            if start < START or start >= END or end <= start:
                continue

            evidence = [
                f"{tractor}: {start.isoformat()} to {end.isoformat()}",
                f"Mean C/N0: {episode['mean_cn0'].mean():.2f} dB-Hz",
                f"Minimum satellite count: {episode['min_sats'].min():.0f}",
                f"Maximum cross-track error: {episode['max_error'].max():.2f} m",
                f"Mean S4: {episode['s4'].mean():.3f}",
                f"Mean Kp: {episode['kp'].mean():.2f}",
                f"Mean Dst: {episode['dst_nT'].mean():.1f} nT",
                f"Peak simultaneous degraded tractors: {episode['affected_fleet_count'].max():.0f}",
            ]

            results.append({
                "start": start.isoformat(),
                "end": end.isoformat(),
                "class": cause,
                "tractors": [tractor],
                "action": ACTION[cause],
                "confidence": 0.6,
                "evidence": evidence,
            })

    return results


detections = []

evaluation = df[(df["ts"] >= START) & (df["ts"] < END)]

for cause, rule in RULES.items():
    detections.extend(detect_episodes(evaluation, cause, rule))

detections.sort(key=lambda x: x["start"])

submission = {
    "participant": {
        "github_handle": "shiyalisaravanan-17"
    },
    "repository_url": (
        "https://github.com/"
        "shiyalisaravanan-17/project_bali_papa"
    ),
    "detections": detections,
}

with open("development_predictions.json", "w") as file:
    json.dump(submission, file, indent=2)

print("Advanced detector completed.")
print("Detected episodes:", len(detections))
print(pd.Series(
    [d["class"] for d in detections]
).value_counts().to_string() if detections else "No detections")
print("Saved to development_predictions.json")
