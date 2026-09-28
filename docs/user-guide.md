# User guide

Based on the in-app "How to Use" section and sidebar (screenshots 01-06).

1. **Choose weights** in the sidebar "Weights file" dropdown; confirm the "RUNNING <file>" chip and the "MODEL LOADED" badge.
2. **Upload** a board image (JPG, JPEG, PNG, BMP). Per the UI, EXIF rotation is corrected and the file is converted to RGB. File name, size in px and upload size are shown. A clear, top-down, evenly lit, in-focus bare-board photo gives the best result (from the app's own guidance).
3. **Set thresholds** in the sidebar: detection confidence, NMS IoU threshold, maximum detections. Input size is locked at 640 px.
4. Click **RUN INSPECTION** (the button lies below the visible region in the screenshots; its label is quoted from the How-to-Use text).
5. **Review** the summary cards, Visual Comparison (tabs: Original vs Annotated, Annotated Full Width, Enhanced Context, Legend), Defect Analytics (tabs: Counts by class, Confidence by detection, Summary table) and the Detection Register (sortable, columns include Detection ID, Defect, Model class name, Class ID, Confidence, Level, box coordinates).
6. **Export** the annotated PNG and the CSV inspection report.

| Setting | Effect |
|---|---|
| Detection confidence | Minimum score for a box to be returned. Lower = more boxes, more false positives. |
| NMS IoU threshold | Overlap above which the weaker of two overlapping same-class boxes is suppressed. |
| Maximum detections | Upper bound on boxes per image. |
| Inference time | Time for the predict call (and stage breakdown) on the current machine. Depends on hardware; not a benchmark. |

A result with zero detections does **not** prove the board is defect-free.
