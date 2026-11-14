import argparse
import json
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

with open(args.input, encoding="utf-8") as f:
    data = json.load(f)

jamming = sorted(
    [dict(d) for d in data["detections"] if d["class"] == "JAMMING"],
    key=lambda d: d["start"]
)

other = [
    d for d in data["detections"]
    if d["class"] != "JAMMING"
]

merged = []

for d in jamming:
    if merged:
        previous = merged[-1]

        gap = (
            pd.Timestamp(d["start"]) -
            pd.Timestamp(previous["end"])
        ).total_seconds() / 60

        if gap <= 3:
            previous["end"] = max(previous["end"], d["end"])
            previous["tractors"] = sorted(
                set(previous["tractors"]) | set(d["tractors"])
            )
            previous["confidence"] = min(
                previous["confidence"], d["confidence"]
            )
            previous["evidence"] = list(dict.fromkeys(
                previous["evidence"] + d["evidence"]
            ))
            continue

    merged.append(d.copy())

data["detections"] = sorted(
    other + merged,
    key=lambda d: d["start"]
)

with open(args.output, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Original jamming detections:", len(jamming))
print("Merged jamming detections:", len(merged))
print("Total predictions:", len(data["detections"]))

for d in merged:
    print(d["tractors"], d["start"], "to", d["end"])

print("Saved:", args.output)
