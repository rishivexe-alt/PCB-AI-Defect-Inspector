# Methodology and evaluation

## What the sources say
From the project report (no new evaluation was run when the report was written):

| Item | Value | Status |
|---|---|---|
| Framework | Ultralytics YOLOv8 (PyTorch) | Stated |
| Input size in dashboard | 640 x 640 | Visible in UI |
| Dataset split | 1,800 train / 62 val / 31 test | Stated (local counts) |
| Precision | 0.702 (70.2 %) | "Previously recorded validation metric" |
| Recall | 0.650 (65.0 %) | same |
| mAP@50 | 0.738 (73.8 %) | same |
| mAP@50-95 | 0.297 (29.7 %) | same |

## Caveats you must keep with these numbers
- They are **validation** metrics on a **62-image** validation set. They are not test-set results and carry high uncertainty.
- The report says they **must be regenerated** with the exact weights that ship with the repo. It is **unconfirmed** whether they belong to `pcb_best.pt` or `pcb_v2_finetuned.pt`.
- No per-class metrics, confusion matrix, latency benchmark, epochs, batch size, optimizer, seed, hardware or model variant (n/s/m/l/x) are available in the supplied materials. They are intentionally left blank rather than guessed.
- The report says the test split previously lacked spur and spurious_copper examples.

## Not model performance: the demo screenshot
One image, one run, `pcb_v2_finetuned.pt`, confidence 0.10, NMS IoU 0.45, max detections 100:

| Shown in dashboard | Value |
|---|---|
| Returned detections | 7 |
| At or above the 0.35 "confirmed" cutoff | 0 |
| Low-confidence (below 0.35) | 7 |
| Classes hit | 2 of 6 (Mouse Bite x6, Spur x1) |
| Highest confidence | 0.315 |
| Timings | preprocess 8.9 ms, inference 167.5 ms, postprocess 7.1 ms; whole call 187.7 ms (shown as CPU) |

This is a single inference on a single image. It says nothing about accuracy. The ground truth for that image was not shown; the file name suggests a mouse-bite sample, but that is not a label check.

## Recommended re-evaluation (to replace the historical numbers)
```powershell
# Adjust paths; this uses the standard Ultralytics CLI. Run on the exact weights you publish.
yolo detect val model=models/pcb_v2_finetuned.pt data=dataset/data.yaml split=test imgsz=640 save_json=True
```
Save the printed per-class table and plots under `reports/` and update this file and the README with the real output. (Command not run here: no weights or data were supplied.)

## Reading confidence
Confidence values are model scores, not calibrated probabilities. Lower thresholds return more boxes and more false positives; choose thresholds from validation data and the cost of a miss vs a false alarm.
