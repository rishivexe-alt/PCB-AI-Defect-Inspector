# Model weights

Weights (`*.pt`) are **not committed by default** (see `.gitignore`).

## Files referenced by the project materials

| File | Where it appears | Notes |
|---|---|---|
| `pcb_v2_finetuned.pt` | Dashboard screenshots (sidebar shows "v2 fine-tuned - pcb_v2_finetuned.pt", status RUNNING) | Used for the demo screenshots. Training details **not documented** in the supplied materials. |
| `pcb_best.pt` | Project report, section 7 (`models/pcb_best.pt`) | Report says a backup copy was made during a Desktop migration. |

**These are not interchangeable.** The historical validation metrics in the report are attributed to "the trained model" in the report, and the report names `pcb_best.pt`; the screenshots use `pcb_v2_finetuned.pt`. Which weights produced which metrics is **unconfirmed** - see `docs/methodology-and-evaluation.md`.

## Placing weights
Put the file(s) directly in this folder, e.g. `models/pcb_v2_finetuned.pt`. The dashboard's "Weights file" dropdown lists the weights it finds (mechanism not audited - `dashboard/app.py` was not supplied).

## Publishing weights (maintainer checklist)
1. Check the size: `Get-ChildItem models\*.pt | Select Name,@{n='MB';e={[math]::Round($_.Length/1MB,1)}}`
2. GitHub warns above 50 MiB and blocks files above 100 MiB per file. Above ~50 MiB use Git LFS or host the file externally (e.g. a GitHub Release asset, Hugging Face).
3. Confirm you may redistribute them (Ultralytics-based weights are AGPL-3.0; see `docs/licensing.md`; also the training-data licence).
4. Record the SHA-256: `Get-FileHash models\pcb_v2_finetuned.pt -Algorithm SHA256` and add it here.

| File | Size (MB) | SHA-256 | Download link |
|---|---|---|---|
| `pcb_v2_finetuned.pt` | _TODO_ | _TODO_ | _TODO_ |
