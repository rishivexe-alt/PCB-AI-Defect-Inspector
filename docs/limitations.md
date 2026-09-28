# Limitations and responsible use

- Research and demonstration prototype; not certified electrical testing, not a production QA system.
- Predictions can be false positives or misses. **No detection does not prove a board is defect-free.**
- Confidence values are model outputs, not necessarily calibrated probabilities.
- Only visible (image-detectable) defects are addressed; electrical function is not tested.
- Historical validation metrics come from a very small (62-image) validation set and may not match the published weights.
- Class-wise reliability is unknown until per-class evaluation is run; the test split reportedly lacked spur and spurious_copper examples.
- Performance varies with resolution, lighting, orientation, board design and defect size.
- Uploaded images may be proprietary: the dashboard states exports are created in memory and the original upload is not modified, but review your own hosting/logging before uploading confidential designs.
- Human review is required for any decision.
