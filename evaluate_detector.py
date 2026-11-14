import json
import pandas as pd

with open("data/labels/train_episodes.json") as f:
    truth = json.load(f)["episodes"]

with open("development_predictions.json") as f:
    predicted = json.load(f)["detections"]

def overlap(a, b):
    start = max(pd.Timestamp(a["start"]), pd.Timestamp(b["start"]))
    end = min(pd.Timestamp(a["end"]), pd.Timestamp(b["end"]))
    if end <= start:
        return 0.0
    intersection = (end - start).total_seconds()
    union_start = min(pd.Timestamp(a["start"]), pd.Timestamp(b["start"]))
    union_end = max(pd.Timestamp(a["end"]), pd.Timestamp(b["end"]))
    return intersection / (union_end - union_start).total_seconds()

matched_truth = set()
matched_pred = set()

pairs = []
for i, p in enumerate(predicted):
    for j, t in enumerate(truth):
        if p["class"] != t["class"]:
            continue
        tractor_intersection = len(set(p["tractors"]) & set(t["tractors"]))
        tractor_union = len(set(p["tractors"]) | set(t["tractors"]))
        tractor_iou = tractor_intersection / tractor_union
        time_iou = overlap(p, t)
        if tractor_iou >= 0.5 and time_iou >= 0.3:
            pairs.append((time_iou, i, j))

for _, i, j in sorted(pairs, reverse=True):
    if i not in matched_pred and j not in matched_truth:
        matched_pred.add(i)
        matched_truth.add(j)

print("Predictions:", len(predicted))
print("Truth episodes:", len(truth))
print("Matched predictions:", len(matched_pred))
print("Unmatched predictions:", len(predicted) - len(matched_pred))
print("Missed truth episodes:", len(truth) - len(matched_truth))

for cause in sorted(set(t["class"] for t in truth)):
    total = sum(t["class"] == cause for t in truth)
    found = sum(truth[j]["class"] == cause for j in matched_truth)
    print(f"{cause}: {found}/{total} matched")
