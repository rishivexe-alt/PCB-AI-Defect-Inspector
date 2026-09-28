# Dataset

The dataset is **not included** in this repository.

## What the project report states
- YOLO-style layout: train / validation / test, each with images and matching label files.
- Local split counts: **1,800 train / 62 validation / 31 test** images (matching image and label counts were checked during setup).
- Configuration file: `dataset/data.yaml` (paths, class count, class names).
- Six classes: 0 missing_hole, 1 mouse_bite, 2 open_circuit, 3 short, 4 spur, 5 spurious_copper.
- The report notes the test split was previously observed to lack spur and spurious_copper examples (to be reconfirmed).

## Provenance - NOT VERIFIED
The report does not name the dataset source or licence. One observation only: the demo image file name in the screenshot (`11_mouse_bite_01_jpg.rf.<hash>.jpg`) follows the naming pattern of Roboflow exports, and the six class names match those of well-known public PCB defect datasets. **Confirm the actual source, version, augmentation used and licence before publishing anything derived from it.**

## To do
- [ ] State the source + URL + version + licence here.
- [ ] If redistribution is not allowed, keep images out of Git and describe how to obtain them.
- [ ] Add `data.yaml` with **relative** paths (no personal paths).
