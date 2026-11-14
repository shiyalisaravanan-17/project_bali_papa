import json
import pandas as pd

with open("development_predictions.json", encoding="utf-8") as f:
    submission = json.load(f)

detections = submission["detections"]

def overlap_ratio(a, b):
    a0 = pd.Timestamp(a["start"])
    a1 = pd.Timestamp(a["end"])
    b0 = pd.Timestamp(b["start"])
    b1 = pd.Timestamp(b["end"])

    overlap = max(0, (min(a1, b1) - max(a0, b0)).total_seconds())
    shortest = min(
        (a1 - a0).total_seconds(),
        (b1 - b0).total_seconds()
    )
    return overlap / shortest if shortest > 0 else 0

groups = []

for detection in sorted(detections, key=lambda x: x["start"]):
    matched = False

    for group in groups:
        # Merge only the same cause across different tractors.
        if group["class"] != detection["class"]:
            continue

        if set(group["tractors"]) & set(detection["tractors"]):
            continue

        if overlap_ratio(group, detection) < 0.7:
            continue

        group["start"] = min(group["start"], detection["start"])
        group["end"] = max(group["end"], detection["end"])
        group["tractors"] = sorted(
            set(group["tractors"]) | set(detection["tractors"])
        )
        group["confidence"] = min(
            group["confidence"], detection["confidence"]
        )
        group["evidence"] = list(dict.fromkeys(
            group["evidence"] + detection["evidence"]
        ))
        matched = True
        break

    if not matched:
        groups.append({
            **detection,
            "tractors": list(detection["tractors"]),
            "evidence": list(detection["evidence"])
        })

submission["detections"] = sorted(groups, key=lambda x: x["start"])

with open("grouped_development.json", "w", encoding="utf-8") as f:
    json.dump(submission, f, indent=2)

print("Original detections:", len(detections))
print("Grouped detections:", len(groups))
print("Saved grouped_development.json")
