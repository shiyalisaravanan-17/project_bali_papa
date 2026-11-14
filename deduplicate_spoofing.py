import json
import pandas as pd

with open("nine_match_development.json", encoding="utf-8") as f:
    result = json.load(f)

detections = result["detections"]

# Merge overlapping single-tractor spoofing episodes only.
spoof = [
    d for d in detections
    if d["class"] == "SPOOFING" and len(d["tractors"]) == 1
]

other = [
    d for d in detections
    if not (d["class"] == "SPOOFING" and len(d["tractors"]) == 1)
]

merged = []

for tractor in ["T1", "T2", "T3"]:
    items = sorted(
        [d for d in spoof if d["tractors"] == [tractor]],
        key=lambda d: d["start"]
    )

    for item in items:
        if merged and merged[-1]["tractors"] == [tractor]:
            prev = merged[-1]

            if pd.Timestamp(item["start"]) <= pd.Timestamp(prev["end"]):
                prev["end"] = max(
                    prev["end"],
                    item["end"]
                )
                prev["confidence"] = max(
                    prev["confidence"],
                    item["confidence"]
                )
                prev["evidence"] = list(dict.fromkeys(
                    prev["evidence"] + item["evidence"]
                ))
                continue

        merged.append(item.copy())

result["detections"] = sorted(
    other + merged,
    key=lambda d: d["start"]
)

with open("deduplicated_development.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print("Before:", len(detections), "predictions")
print("After:", len(result["detections"]), "predictions")
print("Saved deduplicated_development.json")
