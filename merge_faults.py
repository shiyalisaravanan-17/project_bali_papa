import json
import pandas as pd

INPUT = "grouped_development.json"
OUTPUT = "merged_development.json"

with open(INPUT, encoding="utf-8") as f:
    submission = json.load(f)

detections = sorted(
    submission["detections"],
    key=lambda d: (d["class"], tuple(d["tractors"]), d["start"])
)

merged = []

for d in detections:
    current = dict(d)

    if (
        merged
        and current["class"] == "EQUIPMENT_FAULT"
        and merged[-1]["class"] == "EQUIPMENT_FAULT"
        and set(current["tractors"]) == set(merged[-1]["tractors"])
    ):
        previous = merged[-1]

        gap = (
            pd.Timestamp(current["start"]) -
            pd.Timestamp(previous["end"])
        ).total_seconds() / 60

        # Merge nearby intermittent dropouts.
        if gap <= 18:
            previous["end"] = max(previous["end"], current["end"])
            previous["evidence"] = list(dict.fromkeys(
                previous["evidence"] + current["evidence"]
            ))
            previous["confidence"] = min(
                previous["confidence"], current["confidence"]
            )
            continue

    merged.append(current)

submission["detections"] = sorted(
    merged,
    key=lambda d: d["start"]
)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(submission, f, indent=2)

print("Before:", len(detections))
print("After:", len(merged))
print("Saved:", OUTPUT)

for d in merged:
    if d["class"] == "EQUIPMENT_FAULT":
        print(d["tractors"], d["start"], "to", d["end"])
