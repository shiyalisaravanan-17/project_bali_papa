# Straight Rows - GNSS Anomaly Detection

## Bluestaq-NDSU Straight Rows Hackathon 2026

**Participant:** shiyalisaravanan-17

**GitHub:** https://github.com/shiyalisaravanan-17/project_bali_papa

## Project Overview

This project detects and classifies GPS/GNSS guidance anomalies affecting
three simulated agricultural tractors near Casselton, North Dakota.

The solution combines tractor telemetry with Space Domain Awareness (SDA)
information to identify likely causes of guidance degradation and recommend
appropriate operational responses.

## Detection System

The system uses five main techniques:

1. **Space Weather:** Detects fleet-wide GNSS degradation using S4,
   geomagnetic Kp, and receiver measurements.
2. **Jamming:** Identifies interference using signal strength, satellite
   losses, fleet-wide comparisons, and RFI context.
3. **Spoofing:** Detects suspicious guidance-error drift, including
   deviations relative to neighboring tractors.
4. **Equipment Fault:** Identifies machine-specific receiver degradation,
   correction-link failures, and persistent signal deficits.
5. **Multipath:** Investigated using one-second signal measurements;
   not included in the current final prediction pipeline.

Multiple causes can overlap during the same time period.

## Technology

- Python 3.11+
- Pandas
- NumPy
- GNSS telemetry analysis
- Space Domain Awareness data fusion
- Rule-based anomaly detection
- Temporal episode grouping

## Installation

Clone the repository and install the dependencies:

```powershell
git clone https://github.com/shiyalisaravanan-17/project_bali_papa.git
cd project_bali_papa
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dataset

Download the official simulated Straight Rows dataset from:

https://github.com/Bluestaq/ndsu-hackathon-submissions/releases/latest

The provided `scripts/get_data.py` utility can download the package:

```powershell
python scripts/get_data.py
```

Ensure the extracted files are under `data/`, including
`data/farm/tractor_telemetry.csv.gz` and the `data/sda/` feeds.

The original dataset is not included in this repository.

## Reproduce the Final Submission

Run the following commands from the project directory:

```powershell
python build_features.py
python fuse_sda.py
python final_pipeline.py --mode evaluation
python consolidate_jamming.py --input submission.json --output consolidated_submission.json
Copy-Item consolidated_submission.json submission.json -Force
python submission/validate_submission.py submission.json
```

These commands generate the submitted evaluation predictions and
check their format using Bluestaq's official validator.

## Development Results

The development dataset includes 12 labeled incidents.

Our integrated detection system matched:

| Cause | Matched episodes |
|---|---|
| Space Weather | 2/2 |
| Jamming | 3/3 |
| Spoofing | 3/3 |
| Equipment Fault | 2/2 |
| Multipath | 0/2 |
| **Total** | **10/12** |

After jamming consolidation, the development run produced
14 predictions, with 10 matched and 4 unmatched.

These results use a simplified episode-matching evaluator.
They are not official hackathon scores.

## Development Testing

Run:

```powershell
python final_pipeline.py --mode development
python consolidate_jamming.py --input integrated_development.json --output consolidated_development.json
```

The labeled data was used to develop and tune thresholds, compare
detector versions, inspect failure modes, and reduce duplicate alerts.

## Operator Recommendations

| Cause | Action |
|---|---|
| SPACE_WEATHER | HALT_OR_DEGRADED_MODE |
| JAMMING | HALT_AND_REPORT |
| SPOOFING | HALT_CROSSCHECK_AND_REPORT |
| EQUIPMENT_FAULT | SERVICE_MACHINE_OTHERS_CONTINUE |
| MULTIPATH | CONTINUE_FALSE_ALARM |

## Limitations

- Thresholds were tuned using a small simulated development dataset.
- Hidden evaluation accuracy is unknown.
- Brief multipath events remain difficult to classify reliably.
- RFI reports may be delayed, missing, or inaccurate.
- Some detections may have inaccurate start or end times.
- Confidence values are heuristic rather than statistically calibrated.
- Retrospective analysis does not establish live detection latency.

Independent inertial measurements, receiver-level GNSS signal diagnostics,
and improved RFI localization could strengthen future attribution.

## Main Files

- `build_features.py` - Telemetry feature extraction
- `fuse_sda.py` - Space and farm data fusion
- `final_pipeline.py` - Integrated anomaly detection
- `consolidate_jamming.py` - Episode consolidation
- `evaluate_detector.py` - Development evaluation
- `submission.json` - Evaluation predictions
- `submission/validate_submission.py` - Official format validator

## Reproducibility

The solution uses simulated, unclassified data and requires no live
external API for inference. All processing can be run locally.

Developed individually for the Bluestaq-NDSU Straight Rows Hackathon.
