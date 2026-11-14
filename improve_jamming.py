import json
import pandas as pd

df = pd.read_csv("fused_features.csv", parse_dates=["ts"])
df = df[
    (df["ts"] >= "2026-05-04T00:00:00Z") &
    (df["ts"] < "2026-05-06T00:00:00Z")
].copy()

tractors = ["T1", "T2", "T3"]
wide = df.pivot(index="ts", columns="tractor_id",
                values=["mean_cn0", "mean_sats",
                        "rtk_fixed_rate", "recent_rfi_report"])

c = wide["mean_cn0"]
s = wide["mean_sats"]
r = wide["rtk_fixed_rate"]
rfi = wide["recent_rfi_report"].fillna(False).astype(bool)

# Severe fleet-wide interference, independently of space weather
severe = ((c < 27) & (s < 9)).sum(axis=1) >= 2

# Localized interference: at least two receivers substantially
# weaker than the healthiest receiver, with RFI corroboration
best_cn0 = c.max(axis=1)
weak = (c < 41) & (best_cn0.sub(c) >= 2.0)
localized = (
    (weak.sum(axis=1) >= 2) &
    rfi.any(axis=1)
)

candidates = []

def extract_events(mask, cause, minimum_minutes=12):
    # Smooth short gaps without requiring uninterrupted readings
    smooth = (
        mask.astype(float)
        .rolling(7, center=True, min_periods=4)
        .mean() >= 0.60
    ).fillna(False)

    groups = (smooth != smooth.shift(fill_value=False)).cumsum()

    for _, segment in smooth[smooth].groupby(groups[smooth]):
        start = segment.index[0]
        end = segment.index[-1] + pd.Timedelta(minutes=1)

        if (end - start).total_seconds() / 60 < minimum_minutes:
            continue

        period = (wide.index >= start) & (wide.index < end)

        if cause == "severe":
            affected = [
                t for t in tractors
                if ((c.loc[period, t] < 27) &
                    (s.loc[period, t] < 9)).mean() >= 0.5
            ]
        else:
            affected = [
                t for t in tractors
                if weak.loc[period, t].mean() >= 0.5
            ]

        if len(affected) < 2:
            continue

        evidence = [
            "Sustained multi-tractor GNSS signal degradation",
            "Detection pattern: " + cause,
            "Average affected-tractor C/N0: " +
            str(round(c.loc[period, affected].mean().mean(), 2)) +
            " dB-Hz",
            "Affected tractors: " + ", ".join(affected)
        ]

        if cause == "localized":
            evidence.append(
                "Recent RFI report corroborates suspected interference"
            )
        else:
            evidence.append(
                "Severe low C/N0 and satellite loss; "
                "space weather may be concurrent"
            )

        candidates.append({
            "start": start.isoformat(),
            "end": end.isoformat(),
            "tractors": affected,
            "class": "JAMMING",
            "action": "HALT_AND_REPORT",
            "confidence": 0.65 if cause == "severe" else 0.55,
            "evidence": evidence
        })

extract_events(severe, "severe")
extract_events(localized, "localized")

print("New jamming candidates:", len(candidates))
for d in candidates:
    print(d["tractors"], d["start"], "to", d["end"])

with open("best_development_so_far.json", encoding="utf-8") as f:
    result = json.load(f)

result["detections"].extend(candidates)
result["detections"].sort(key=lambda d: d["start"])

with open("jamming_test_development.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print("Saved jamming_test_development.json")
