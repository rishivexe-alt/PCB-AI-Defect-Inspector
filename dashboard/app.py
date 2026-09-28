"""
PCB Defect Inspector - industrial AI visual inspection dashboard.

Streamlit front-end for the trained YOLOv8 PCB defect models stored in
``ROOT / "models"``. The file is chosen in the sidebar: models/pcb_v2_finetuned.pt
by default, models/pcb_v1_baseline.pt as the alternative.

Engineering rules this module obeys without exception:

* Exactly one weight file is loaded per run - the one selected in the sidebar.
  Nothing is substituted, retrained, remapped or re-scaled.
* Class identities always come from ``model.names`` at runtime. Predictions are
  reported exactly as returned - no class is forced, dropped or renamed.
* Inference runs once per explicit ``RUN INSPECTION`` click and the result is
  held in ``st.session_state``. Every downstream panel renders from that single
  stored result, so the panels can never disagree with each other.
* No performance figure is ever invented. Numbers appear only when the real
  ``eval_summary.csv`` / ``eval_counts.csv`` files contain them, always with
  provenance, and the performance panel can never crash the app.
* Bounding boxes are drawn with PIL, not the default Ultralytics plotter.
* Inference defaults to CPU.
* This is a research prototype. It is not certified electrical testing.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO


# ==================================================
# SECTION 1 - CONSTANTS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# Which trained weight file the dashboard may run, chosen in the sidebar.
# Entries are (key, sidebar label, file name, label used in the metrics CSVs).
MODEL_CHOICES: tuple[tuple[str, str, str, str | None], ...] = (
    ("v2_finetuned", "v2 fine-tuned", "pcb_v2_finetuned.pt", "v2_finetuned"),
    ("v1_baseline", "v1 baseline", "pcb_v1_baseline.pt", "v1_baseline"),
)

# Used when models/pcb_v2_finetuned.pt is absent. It has no entry in the
# metrics CSVs, so the performance panel reports that it has nothing to show.
MODEL_FALLBACK_CHOICE: tuple[str, str, str, str | None] = (
    "pcb_best",
    "pcb_best.pt (fallback)",
    "pcb_best.pt",
    None,
)

# The two metrics files, read with exactly these column names.
SUMMARY_FILE = "eval_summary.csv"
COUNTS_FILE = "eval_counts.csv"
SUMMARY_COLUMNS: tuple[str, ...] = (
    "model", "split", "cls", "precision", "recall", "mAP50", "mAP50_95",
)
COUNTS_COLUMNS: tuple[str, ...] = (
    "model", "split", "iou", "cls", "conf", "TP", "FP", "FN",
)

# Directory names the metrics CSVs may live in, checked in this order.
METRICS_DIR_NAMES: tuple[str, ...] = ("metrics", "metrics_hold")

# Shown for a class the metrics file has no row for. The split name is
# substituted so the sentence stays true for val as well as test.
NOT_EVALUATED = "Not evaluated - no {split} annotations"

# Default detection confidence. The fine-tuned v2 model scores its true
# detections lower than the baseline, so 0.10 surfaces defects the v1
# threshold of 0.15 dropped.
DEFAULT_CONFIDENCE = 0.10

APP_NAME = "PCB Defect Inspector"
APP_SUBTITLE = "YOLOv8 AI Visual Inspection"
APP_VERSION = "2.0"

# The model was trained and evaluated at 640 px. Fixed on purpose: inference at
# any other size would make on-screen results inconsistent with training.
IMGSZ = 640

ACCEPTED_UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp"]
ACCEPTED_UPLOAD_LABEL = "JPG, JPEG, PNG, BMP"

# Decoding guard. A PCB photograph well beyond this is almost always a phone
# panorama or a raw export, and resizing it silently would misreport geometry.
MAX_IMAGE_PIXELS = 50_000_000

# Class names documented in dataset/data.yaml. Used only to cross-check the
# loaded model and to pick presentation colours. model.names stays authoritative.
EXPECTED_CLASS_NAMES: dict[int, str] = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spur",
    5: "spurious_copper",
}
EXPECTED_CLASS_COUNT = 6

DISPLAY_NAMES: dict[str, str] = {
    "missing_hole": "Missing Hole",
    "mouse_bite": "Mouse Bite",
    "open_circuit": "Open Circuit",
    "short": "Short Circuit",
    "spur": "Spur",
    "spurious_copper": "Spurious Copper",
}

# One fixed, colourblind-aware palette. A colour always means the same class in
# the boxes, labels, cards, chart, legend, table and CSV.
CLASS_COLORS: dict[int, str] = {
    0: "#38BDF8",
    1: "#FB923C",
    2: "#A78BFA",
    3: "#F43F5E",
    4: "#34D399",
    5: "#FACC15",
}
FALLBACK_COLOR = "#CBD5E1"

UI = {
    "ok": "#34D399",
    "warn": "#FBBF24",
    "danger": "#F43F5E",
    "accent": "#22D3EE",
    "accent2": "#2DD4BF",
    "violet": "#A78BFA",
    "yellow": "#FACC15",
}

PROTOTYPE_NOTICE = (
    "AI prototype result only. This is visual inference on a single image. "
    "It is not certified electrical testing and it does not guarantee that a "
    "board is defect-free."
)

LIMITATIONS: list[str] = [
    "Research prototype built for a student project. It has not been qualified "
    "for production line use.",
    "Not certified electrical testing. No electrical, continuity or netlist "
    "property is measured - only visual appearance is inferred.",
    "The held-out test split contains no annotated Spur or Spurious Copper "
    "instances, so test-split results cannot speak to those two classes.",
    "Independent validation on real production images under varied lighting, "
    "camera and line conditions is still required.",
    "A prediction is not a certification. A board may be reported defective "
    "when it is not, or missed when it is.",
]

THRESHOLD_CAVEAT = (
    "Changing a threshold only changes what this dashboard reports for the "
    "current image. It does not retrain, fine-tune or otherwise improve the "
    "trained model. Lower confidence surfaces more detections and more false "
    "positives; higher confidence can hide real defects."
)

HOW_TO_USE: list[tuple[str, str]] = [
    ("1", "Upload a clear bare-board photograph in JPG, JPEG, PNG or BMP. Top-down, evenly lit and in focus gives the most reliable result."),
    ("2", "Set the detection confidence and NMS IoU thresholds in the sidebar. Both are passed straight to the model."),
    ("3", "Click RUN INSPECTION. The model runs once for the current image and settings."),
    ("4", "Review the annotated image, defect counts, confidence scores and bounding boxes."),
    ("5", "Download the annotated PNG and the CSV inspection report for your records."),
]


# ==================================================
# SECTION 2 - THEME
# ==================================================

THEME_CSS = """
<style>
:root {
  --pdb-bg:        #060A12;
  --pdb-bg-soft:   #0A1120;
  --pdb-card:      #0E1726;
  --pdb-card-2:    #121D30;
  --pdb-line:      rgba(148, 163, 184, 0.14);
  --pdb-line-soft: rgba(148, 163, 184, 0.08);
  --pdb-accent:    #22D3EE;
  --pdb-accent-2:  #2DD4BF;
  --pdb-text:      #E8EFF9;
  --pdb-muted:     #93A4BD;
  --pdb-dim:       #6C7E99;
  --pdb-radius:    16px;
}

html, body, [class*="css"], .stApp, button, input, select, textarea {
  font-family: Inter, "Segoe UI", system-ui, -apple-system, "Helvetica Neue",
    Arial, "Noto Sans", sans-serif;
}

#MainMenu, footer, [data-testid="stStatusWidget"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
#MainMenu > div, .stDeployButton { display: none; }

.stApp {
  background:
    radial-gradient(1100px 620px at 12% -8%, rgba(34, 211, 238, 0.10), transparent 62%),
    radial-gradient(900px 520px at 96% 4%, rgba(45, 212, 191, 0.08), transparent 58%),
    linear-gradient(180deg, var(--pdb-bg) 0%, var(--pdb-bg-soft) 100%);
  background-attachment: fixed;
  color: var(--pdb-text);
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: var(--pdb-text); letter-spacing: -0.01em; }
.stApp p, .stApp span, .stApp label, .stApp li { color: var(--pdb-text); }
.stApp [data-testid="stCaptionContainer"] p,
.stApp .stCaption { color: var(--pdb-muted) !important; }
hr, [data-testid="stDivider"] { border-color: var(--pdb-line) !important; }

.block-container { padding-top: 2rem; padding-bottom: 3.5rem; max-width: 1500px; }

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0B1526 0%, #08101D 100%);
  border-right: 1px solid var(--pdb-line);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
  background: var(--pdb-accent) !important; border-color: #0B1526 !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[role="listbox"] {
  background: rgba(18, 29, 48, 0.9) !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input { border-color: var(--pdb-line); }

.pdb-header {
  display: flex; align-items: center; gap: 1.05rem; flex-wrap: wrap;
  padding: 1.15rem 1.35rem; border: 1px solid var(--pdb-line);
  border-radius: var(--pdb-radius);
  background: linear-gradient(120deg, rgba(18, 29, 48, 0.92) 0%, rgba(11, 19, 33, 0.86) 100%);
  box-shadow: 0 18px 44px -28px rgba(0, 0, 0, 0.9);
}
.pdb-logo {
  width: 48px; height: 48px; flex: 0 0 48px; border-radius: 13px;
  background: linear-gradient(140deg, rgba(34, 211, 238, 0.20), rgba(45, 212, 191, 0.10));
  border: 1px solid rgba(34, 211, 238, 0.36);
  display: flex; align-items: center; justify-content: center;
}
.pdb-logo svg { width: 27px; height: 27px; }
.pdb-head-text { flex: 1 1 340px; min-width: 260px; }
.pdb-title {
  font-size: 1.62rem; font-weight: 700; line-height: 1.15; margin: 0;
  background: linear-gradient(92deg, #FFFFFF 0%, #A5F3FC 60%, #5EEAD4 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.pdb-subtitle {
  font-size: 0.78rem; font-weight: 700; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--pdb-accent); margin: 0.28rem 0 0;
}
.pdb-tagline { font-size: 0.83rem; color: var(--pdb-muted); margin: 0.35rem 0 0; }
.pdb-pill {
  display: inline-flex; align-items: center; gap: 0.5rem; white-space: nowrap;
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.10em;
  padding: 0.46rem 0.85rem; border-radius: 999px;
}
.pdb-pill .pdb-dot { width: 8px; height: 8px; border-radius: 50%; }
.pdb-pill--ok { color: #6EE7B7; background: rgba(52, 211, 153, 0.10); border: 1px solid rgba(52, 211, 153, 0.36); }
.pdb-pill--ok .pdb-dot { background: #34D399; box-shadow: 0 0 0 4px rgba(52, 211, 153, 0.16); }
.pdb-pill--bad { color: #FDA4AF; background: rgba(244, 63, 94, 0.10); border: 1px solid rgba(244, 63, 94, 0.36); }
.pdb-pill--bad .pdb-dot { background: #F43F5E; box-shadow: 0 0 0 4px rgba(244, 63, 94, 0.16); }
.pdb-rule {
  height: 2px; margin: 0.85rem 0 1.35rem; border-radius: 2px;
  background: linear-gradient(90deg, var(--pdb-accent) 0%, var(--pdb-accent-2) 32%, rgba(45,212,191,0) 100%);
}

.pdb-card { border: 1px solid var(--pdb-line); border-radius: var(--pdb-radius); background: linear-gradient(180deg, rgba(18, 29, 48, 0.78) 0%, rgba(12, 20, 34, 0.78) 100%); padding: 1.05rem 1.15rem; }
.pdb-note {
  border: 1px solid var(--pdb-line); border-left: 3px solid var(--pdb-accent);
  border-radius: 12px; background: rgba(34, 211, 238, 0.05);
  padding: 0.72rem 0.9rem; font-size: 0.83rem; color: var(--pdb-muted);
}
.pdb-note strong, .pdb-note b { color: var(--pdb-text); }
.pdb-note--warn { border-left-color: #FBBF24; background: rgba(251, 191, 36, 0.06); }
.pdb-note--stale { border-left-color: #FB923C; background: rgba(251, 146, 60, 0.07); }
.pdb-note--danger { border-left-color: #F43F5E; background: rgba(244, 63, 94, 0.07); }
.pdb-note--list { margin: 0.35rem 0 0; padding-left: 1.1rem; }
.pdb-note--list li { margin-bottom: 0.32rem; color: var(--pdb-muted); font-size: 0.85rem; }
.pdb-note--list li::marker { color: var(--pdb-accent); }

.pdb-sec { display: flex; align-items: center; gap: 0.7rem; margin: 0.2rem 0 0.15rem; }
.pdb-sec-bar { width: 4px; height: 20px; border-radius: 3px; flex: 0 0 4px; background: linear-gradient(180deg, var(--pdb-accent), var(--pdb-accent-2)); }
.pdb-sec-title { font-size: 1.10rem; font-weight: 650; margin: 0; }
.pdb-sec-num {
  font-size: 0.66rem; font-weight: 800; letter-spacing: 0.10em; color: #062A33;
  background: linear-gradient(140deg, var(--pdb-accent), var(--pdb-accent-2));
  border-radius: 6px; padding: 0.16rem 0.42rem; margin-left: 0.1rem;
}
.pdb-sec-sub { font-size: 0.82rem; color: var(--pdb-muted); margin: 0.1rem 0 0.7rem 0.9rem; }

.pdb-banner {
  display: flex; align-items: center; gap: 0.95rem; flex-wrap: wrap;
  border-radius: var(--pdb-radius); padding: 1.05rem 1.25rem; margin-bottom: 0.9rem;
  border: 1px solid var(--pdb-line);
}
.pdb-banner--danger { border-color: rgba(244, 63, 94, 0.42); background: linear-gradient(120deg, rgba(244, 63, 94, 0.16) 0%, rgba(244, 63, 94, 0.04) 70%); }
.pdb-banner--ok { border-color: rgba(52, 211, 153, 0.42); background: linear-gradient(120deg, rgba(52, 211, 153, 0.15) 0%, rgba(52, 211, 153, 0.04) 70%); }
.pdb-banner-icon { flex: 0 0 auto; display: flex; }
.pdb-banner-icon svg { width: 30px; height: 30px; }
.pdb-banner--danger .pdb-banner-icon { color: #FB7185; }
.pdb-banner--ok .pdb-banner-icon { color: #34D399; }
.pdb-banner-text { flex: 1 1 320px; min-width: 240px; }
.pdb-banner-title { font-size: 1.18rem; font-weight: 700; letter-spacing: 0.03em; margin: 0; }
.pdb-banner--danger .pdb-banner-title { color: #FFE4E6; }
.pdb-banner--ok .pdb-banner-title { color: #D1FAE5; }
.pdb-banner-sub { font-size: 0.82rem; color: var(--pdb-muted); margin: 0.25rem 0 0; }
.pdb-banner-badge { font-size: 0.70rem; font-weight: 700; letter-spacing: 0.12em; padding: 0.42rem 0.8rem; border-radius: 999px; border: 1px solid currentColor; }
.pdb-banner--danger .pdb-banner-badge { color: #FDA4AF; }
.pdb-banner--ok .pdb-banner-badge { color: #6EE7B7; }

.pdb-kpi-grid { display: grid; gap: 0.75rem; margin: 0.35rem 0 0.6rem 0; grid-template-columns: repeat(auto-fit, minmax(178px, 1fr)); }
.pdb-kpi {
  border: 1px solid var(--pdb-line); border-radius: 14px; padding: 0.85rem 0.9rem;
  background: linear-gradient(180deg, rgba(19, 30, 50, 0.86) 0%, rgba(12, 20, 34, 0.86) 100%);
  position: relative; overflow: hidden;
}
.pdb-kpi::before { content: ""; position: absolute; inset: 0 auto auto 0; height: 2px; width: 100%; background: var(--pdb-kpi-accent, var(--pdb-accent)); opacity: 0.75; }
.pdb-kpi-top { display: flex; align-items: center; gap: 0.45rem; margin-bottom: 0.4rem; }
.pdb-kpi-ic { color: var(--pdb-kpi-accent, var(--pdb-accent)); display: flex; }
.pdb-kpi-ic svg { width: 16px; height: 16px; }
.pdb-kpi-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: var(--pdb-dim); }
.pdb-kpi-value { font-size: 1.72rem; font-weight: 700; line-height: 1.05; color: var(--pdb-text); font-variant-numeric: tabular-nums; }
.pdb-kpi-hint { font-size: 0.71rem; color: var(--pdb-dim); margin-top: 0.3rem; }

.st-key-pdb-upload-card {
  border: 1px solid var(--pdb-line); border-radius: var(--pdb-radius);
  background: linear-gradient(180deg, rgba(18, 29, 48, 0.72) 0%, rgba(11, 19, 32, 0.72) 100%);
  padding: 1.05rem 1.15rem;
}
[data-testid="stFileUploaderDropzone"] {
  background: rgba(9, 16, 28, 0.72) !important;
  border: 1.5px dashed rgba(34, 211, 238, 0.42) !important;
  border-radius: 14px !important; padding: 1.15rem 1rem !important;
  transition: border-color 0.18s ease, background 0.18s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: rgba(34, 211, 238, 0.75) !important; background: rgba(15, 27, 45, 0.8) !important; }
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--pdb-muted) !important; }
[data-testid="stFileUploader"] button { color: var(--pdb-accent) !important; }

.pdb-empty {
  display: flex; flex-direction: column; align-items: center; text-align: center;
  border: 1px solid var(--pdb-line); border-radius: var(--pdb-radius);
  background:
    radial-gradient(460px 240px at 50% 0%, rgba(34, 211, 238, 0.10), transparent 70%),
    linear-gradient(180deg, rgba(18, 29, 48, 0.70) 0%, rgba(11, 19, 32, 0.70) 100%);
  padding: 2.1rem 1.6rem 1.7rem 1.6rem;
}
.pdb-empty svg { width: 168px; height: 108px; margin-bottom: 0.5rem; }
.pdb-empty-title { font-size: 1.32rem; font-weight: 700; margin: 0 0 0.4rem 0; }
.pdb-empty-text { font-size: 0.88rem; color: var(--pdb-muted); max-width: 620px; margin: 0 0 0.9rem 0; }
.pdb-chips { display: flex; gap: 0.45rem; flex-wrap: wrap; justify-content: center; }
.pdb-chip { font-size: 0.70rem; font-weight: 600; letter-spacing: 0.06em; color: var(--pdb-muted); border: 1px solid var(--pdb-line); border-radius: 999px; padding: 0.3rem 0.7rem; background: rgba(10, 17, 29, 0.6); }

.st-key-pdb-run button[kind="primary"] {
  background: linear-gradient(96deg, #0E7490 0%, #0891B2 45%, #14B8A6 100%) !important;
  border: 1px solid rgba(94, 234, 212, 0.5) !important;
  color: #F0FDFA !important; font-weight: 700 !important; letter-spacing: 0.14em;
  font-size: 0.98rem !important; padding: 0.72rem 1rem !important; border-radius: 13px !important;
  box-shadow: 0 14px 34px -20px rgba(20, 184, 166, 0.95) !important;
}
.st-key-pdb-run button[kind="primary"]:hover { filter: brightness(1.1); }
.st-key-pdb-run button[kind="primary"]:active { filter: brightness(0.97); }

.pdb-meta { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.15rem 0 0.1rem 0; }
.pdb-meta-chip {
  display: inline-flex; align-items: center; gap: 0.42rem;
  font-size: 0.74rem; color: var(--pdb-muted);
  border: 1px solid var(--pdb-line); border-radius: 9px;
  background: rgba(10, 17, 29, 0.62); padding: 0.34rem 0.66rem;
}
.pdb-meta-chip b { color: var(--pdb-text); font-weight: 600; }

.pdb-class-grid { display: grid; gap: 0.7rem; margin: 0.3rem 0 0.9rem 0; grid-template-columns: repeat(auto-fit, minmax(198px, 1fr)); }
.pdb-class {
  display: flex; align-items: center; gap: 0.7rem;
  border: 1px solid var(--pdb-line); border-radius: 13px; padding: 0.7rem 0.8rem;
  background: linear-gradient(180deg, rgba(18, 29, 48, 0.7) 0%, rgba(12, 20, 34, 0.7) 100%);
  border-left: 3px solid var(--pdb-class-color, var(--pdb-accent));
}
.pdb-class-dot { width: 12px; height: 12px; border-radius: 4px; flex: 0 0 12px; box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.05); }
.pdb-class-meta { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; }
.pdb-class-name { font-size: 0.80rem; font-weight: 600; color: var(--pdb-text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pdb-class-id { font-size: 0.66rem; color: var(--pdb-dim); letter-spacing: 0.05em; }
.pdb-class-count { font-size: 1.32rem; font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums; color: var(--pdb-text); }
.pdb-class--zero .pdb-class-count { color: var(--pdb-dim); }
.pdb-class-conf { font-size: 0.63rem; color: var(--pdb-dim); font-variant-numeric: tabular-nums; }

.pdb-legend-grid { display: grid; gap: 0.55rem; margin: 0.25rem 0 0.9rem 0; grid-template-columns: repeat(auto-fit, minmax(228px, 1fr)); }
.pdb-legend-item { display: flex; align-items: center; gap: 0.6rem; border: 1px solid var(--pdb-line-soft); border-radius: 11px; background: rgba(10, 17, 29, 0.55); padding: 0.48rem 0.7rem; }
.pdb-legend-swatch { width: 14px; height: 14px; border-radius: 4px; flex: 0 0 14px; }
.pdb-legend-text { font-size: 0.80rem; color: var(--pdb-text); }
.pdb-legend-key { font-size: 0.76rem; color: var(--pdb-muted); }
.pdb-line-key { width: 30px; height: 0; border-top: 3px solid #94A3B8; flex: 0 0 30px; }
.pdb-line-key--dashed { border-top-style: dashed; border-top-width: 2px; border-top-color: #7C8DA6; }

.pdb-steps { display: grid; gap: 0.65rem; grid-template-columns: repeat(auto-fit, minmax(232px, 1fr)); }
.pdb-step { display: flex; gap: 0.7rem; align-items: flex-start; border: 1px solid var(--pdb-line); border-radius: 13px; padding: 0.78rem 0.85rem; background: rgba(12, 20, 34, 0.62); }
.pdb-step-n { flex: 0 0 26px; width: 26px; height: 26px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.78rem; font-weight: 700; color: #062A33; background: linear-gradient(140deg, var(--pdb-accent), var(--pdb-accent-2)); }
.pdb-step-t { font-size: 0.82rem; color: var(--pdb-muted); margin: 0; }

.pdb-metric-row { display: grid; gap: 0.6rem; margin: 0.3rem 0 0.6rem 0; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
.pdb-metric { border: 1px solid var(--pdb-line); border-radius: 12px; padding: 0.7rem 0.8rem; background: rgba(12, 20, 34, 0.7); text-align: center; }
.pdb-metric-v { font-size: 1.42rem; font-weight: 700; color: var(--pdb-text); font-variant-numeric: tabular-nums; }
.pdb-metric-v--na { font-size: 1.02rem; font-weight: 600; color: var(--pdb-dim); }
.pdb-metric-k { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.11em; text-transform: uppercase; color: var(--pdb-dim); margin-top: 0.15rem; }

.pdb-kv { display: grid; gap: 0.3rem 0.9rem; grid-template-columns: auto 1fr; font-size: 0.82rem; }
.pdb-kv dt { color: var(--pdb-dim); }
.pdb-kv dd { margin: 0; color: var(--pdb-text); font-weight: 600; }

.st-key-pdb-dl-img button[kind="primary"] {
  background: linear-gradient(96deg, #0E7490 0%, #0891B2 100%) !important;
  border: 1px solid rgba(94, 234, 212, 0.45) !important; color: #F0FDFA !important;
  font-weight: 600 !important; border-radius: 11px !important;
}

.pdb-foot { margin-top: 1.6rem; padding-top: 0.9rem; border-top: 1px solid var(--pdb-line); font-size: 0.74rem; color: var(--pdb-dim); text-align: center; }

[data-testid="stDataFrame"] { border: 1px solid var(--pdb-line); border-radius: 13px; overflow: hidden; }
[data-baseweb="tab-list"] { gap: 0.3rem; border-bottom: 1px solid var(--pdb-line); }
[data-baseweb="tab"] { background: transparent; color: var(--pdb-muted); font-weight: 600; font-size: 0.86rem; padding: 0.5rem 0.9rem; }
[data-baseweb="tab"][aria-selected="true"] { color: var(--pdb-accent); }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { background-color: var(--pdb-accent); }
[data-testid="stExpander"] { border: 1px solid var(--pdb-line); border-radius: 13px; background: rgba(11, 19, 32, 0.66); }
[data-testid="stExpander"] summary { font-weight: 600; color: var(--pdb-text); }
[data-testid="stSelectbox"] [data-baseweb="select"] > div { background: rgba(12, 20, 34, 0.7); border-color: var(--pdb-line); }

@media (max-width: 900px) {
  .pdb-title { font-size: 1.34rem; }
  .pdb-empty { padding: 1.5rem 1.05rem; }
  .pdb-empty svg { width: 132px; height: 86px; }
}
</style>
"""


# ==================================================
# SECTION 3 - INLINE SVG ASSETS
# ==================================================

SVG_LOGO = (
    '<svg viewBox="0 0 48 48" fill="none" stroke="#7DE9F7" stroke-width="2"'
    ' stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="9" y="9" width="30" height="30" rx="5"/>'
    '<rect x="18" y="18" width="12" height="12" rx="2.5" fill="rgba(125,233,247,0.22)"/>'
    '<path d="M14 9V4M22 9V4M30 9V4M38 9V4M14 44v-5M22 44v-5M30 44v-5M38 44v-5"/>'
    '<path d="M9 14H4M9 22H4M9 30H4M9 38H4M44 14h-5M44 22h-5M44 30h-5M44 38h-5"/>'
    "</svg>"
)

SVG_EMPTY_BOARD = (
    '<svg viewBox="0 0 300 190" fill="none">'
    "<defs>"
    '<linearGradient id="pcbEdge" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#0E7490"/><stop offset="1" stop-color="#115E59"/>'
    "</linearGradient>"
    '<linearGradient id="scan" x1="0" y1="0" x2="1" y2="0">'
    '<stop offset="0" stop-color="#22D3EE" stop-opacity="0"/>'
    '<stop offset="0.5" stop-color="#22D3EE" stop-opacity="0.85"/>'
    '<stop offset="1" stop-color="#22D3EE" stop-opacity="0"/>'
    "</linearGradient>"
    "</defs>"
    '<rect x="22" y="16" width="256" height="158" rx="14" fill="#0B1626"'
    ' stroke="#1E3A5F" stroke-width="1.5"/>'
    '<rect x="38" y="32" width="96" height="60" rx="7" fill="url(#pcbEdge)" opacity="0.28"'
    ' stroke="#2DD4BF" stroke-width="1.1"/>'
    '<rect x="150" y="32" width="112" height="26" rx="5" fill="#0F2033"'
    ' stroke="#24466B" stroke-width="1"/>'
    '<g stroke="#1E88A8" stroke-width="1.2" opacity="0.85" stroke-linecap="round">'
    '<path d="M38 60h34v34h22"/><path d="M38 76h20v52h44"/>'
    '<path d="M150 58v34h-38v40"/><path d="M170 45h44v34h-22v40"/>'
    '<path d="M214 45h48v22h-30"/><path d="M214 132h48v22"/>'
    "</g>"
    '<g fill="#22D3EE">'
    '<circle cx="38" cy="60" r="3"/><circle cx="38" cy="76" r="3"/><circle cx="38" cy="128" r="3"/>'
    '<circle cx="94" cy="110" r="3"/><circle cx="112" cy="132" r="3"/><circle cx="150" cy="58" r="3"/>'
    '<circle cx="150" cy="132" r="3"/><circle cx="170" cy="45" r="3"/><circle cx="192" cy="119" r="3"/>'
    '<circle cx="214" cy="45" r="3"/><circle cx="214" cy="132" r="3"/><circle cx="262" cy="67" r="3"/>'
    '<circle cx="232" cy="45" r="3"/><circle cx="262" cy="154" r="3"/>'
    "</g>"
    '<rect x="34" y="86" width="232" height="26" fill="url(#scan)"/>'
    '<rect x="120" y="106" width="72" height="60" rx="7" stroke="#FB7185" stroke-width="1.6"'
    ' stroke-dasharray="6 5" fill="rgba(244,63,94,0.07)"/>'
    '<rect x="120" y="106" width="72" height="60" rx="7" stroke="#FB7185" stroke-width="1.1" opacity="0.55"/>'
    '<text x="156" y="140" fill="#FDA4AF" font-family="Inter, Segoe UI, sans-serif"'
    ' font-size="12" text-anchor="middle">0.87</text>'
    "</svg>"
)

ICON_CROSSHAIR = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"'
    ' stroke-linecap="round"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/>'
    '<path d="M12 1.6v3.1M12 19.3v3.1M1.6 12h3.1M19.3 12h3.1"/></svg>'
)
ICON_CHECK = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1"'
    ' stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>'
)
ICON_WARN = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"'
    ' stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>'
    '<path d="M12 9.5v4"/><path d="M12 17.2h.01"/></svg>'
)
ICON_CLOCK = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"'
    ' stroke-linecap="round"><circle cx="12" cy="12" r="8.6"/><path d="M12 6.8V12l3.4 2.2"/></svg>'
)
ICON_GAUGE = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"'
    ' stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M3.4 17.2a9 9 0 1 1 17.2 0"/><path d="m13.6 10.4 3.6-3.6"/>'
    '<circle cx="12" cy="14" r="1.7"/></svg>'
)
ICON_LAYERS = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"'
    ' stroke-linecap="round" stroke-linejoin="round">'
    '<path d="m12 2.9 9 4.8-9 4.8-9-4.8Z"/><path d="m3 12.6 9 4.8 9-4.8"/>'
    '<path d="m3 17.4 9 4.8 9-4.8"/></svg>'
)
ICON_ALERT_BANNER = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"'
    ' stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="8.6"/><path d="M12 7.6v5"/><path d="M12 16.1h.01"/></svg>'
)
ICON_SHIELD_OK = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"'
    ' stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 2.8 4.5 6v5.4c0 4.6 3.1 8.3 7.5 9.8 4.4-1.5 7.5-5.2 7.5-9.8V6Z"/>'
    '<path d="m8.9 12.2 2.2 2.2 4-4.4"/></svg>'
)


# ==================================================
# SECTION 4 - UTILITIES
# ==================================================

class ImageDecodeError(RuntimeError):
    """Raised when an uploaded file cannot be decoded into a usable image."""


def html(markup: str) -> None:
    """Render a locally built, trusted HTML fragment."""
    st.markdown(markup, unsafe_allow_html=True)


def inject_styles() -> None:
    html(THEME_CSS)


def esc(value: Any) -> str:
    """Escape a value for safe interpolation into HTML markup."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Convert a ``#rrggbb`` string into an RGB tuple, with a safe fallback."""
    value = color.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        return (203, 213, 225)
    try:
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
    except ValueError:
        return (203, 213, 225)


def fade_color(color: str, amount: float, background: tuple[int, int, int] = (8, 14, 24)) -> str:
    """Blend a palette colour towards the dark card background."""
    r, g, b = hex_to_rgb(color)
    br, bg, bb = background
    return "#{:02X}{:02X}{:02X}".format(
        int(round(r + (br - r) * amount)),
        int(round(g + (bg - g) * amount)),
        int(round(b + (bb - b) * amount)),
    )


def format_bytes(size: int) -> str:
    """Human readable byte count."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def display_label(raw_name: str) -> str:
    """Map a raw model class name onto its human readable dashboard label.

    This is presentation only. ``raw_name`` from ``model.names`` is always the
    identity that is stored, tabulated and exported.
    """
    if raw_name in DISPLAY_NAMES:
        return DISPLAY_NAMES[raw_name]
    cleaned = raw_name.replace("_", " ").strip()
    return cleaned.title() if cleaned else "Unknown Class"


def section_title(number: str, title: str, subtitle: str | None = None) -> None:
    """Consistent numbered section heading with an accent bar."""
    html(
        '<div class="pdb-sec"><span class="pdb-sec-bar"></span>'
        f'<span class="pdb-sec-title">{esc(title)}</span>'
        f'<span class="pdb-sec-num">{esc(number)}</span></div>'
    )
    if subtitle:
        html(f'<div class="pdb-sec-sub">{esc(subtitle)}</div>')


def as_number(value: Any) -> float | None:
    """Best-effort float conversion that never raises and never returns NaN."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def format_percent(value: Any) -> str:
    """Format a stored metric as a percentage, or say it is unavailable."""
    number = as_number(value)
    if number is None:
        return "Not reported"
    percent = number * 100.0 if abs(number) <= 1.0 else number
    return f"{percent:.1f}%"


# ==================================================
# SECTION 5 - MODEL BOOTSTRAP
# ==================================================

def model_choice(key: str) -> tuple[str, str, str, str | None]:
    """Registry entry for a model key, falling back to the v2 fine-tuned entry."""
    for choice in (*MODEL_CHOICES, MODEL_FALLBACK_CHOICE):
        if choice[0] == key:
            return choice
    return MODEL_CHOICES[0]


def default_model_key() -> str:
    """The model the dashboard runs unless the user picks another one.

    models/pcb_v2_finetuned.pt when it exists, otherwise models/pcb_best.pt.
    """
    if (MODELS_DIR / MODEL_CHOICES[0][2]).is_file():
        return MODEL_CHOICES[0][0]
    if (MODELS_DIR / MODEL_FALLBACK_CHOICE[2]).is_file():
        return MODEL_FALLBACK_CHOICE[0]
    return MODEL_CHOICES[0][0]


def model_path_for(key: str) -> Path:
    """Absolute path of the weight file for a model key."""
    return MODELS_DIR / model_choice(key)[2]


def active_model_key() -> str:
    """The model key currently selected in the sidebar."""
    selected = st.session_state.get("pdb_model_key")
    if isinstance(selected, str) and selected:
        return selected
    return default_model_key()


def active_model_path() -> Path:
    """The weight file the dashboard is actually running right now."""
    return model_path_for(active_model_key())


def metrics_model_label() -> str | None:
    """How the active weight file is named in the ``model`` column of the CSVs."""
    return model_choice(active_model_key())[3]


@st.cache_resource(show_spinner=False)
def load_model(model_path: str, modified_at: float) -> YOLO:
    """Load and cache the trained YOLO weights.

    ``modified_at`` is part of the cache key so the weights are re-read if the
    file itself ever changes on disk. Exactly one file is loaded per run, the one
    the user selected in the sidebar.
    """
    return YOLO(model_path)


def cuda_available() -> bool:
    """Whether a CUDA device is genuinely usable."""
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch rarely raises here
        return False


def available_devices() -> list[str]:
    """Device list for the sidebar. CPU is always present and always first."""
    devices = ["cpu"]
    if cuda_available():
        devices.append("cuda:0")
    return devices


def boot_model() -> tuple[YOLO | None, str | None, str | None]:
    """Return ``(model, user_message, technical_detail)`` for the header status."""
    model_path = active_model_path()
    if not model_path.is_file():
        return (
            None,
            "The selected model file could not be found, so no inspection can "
            "run. This dashboard will not substitute a different or pretrained "
            f"model. Place the trained weights at models/{model_path.name} and "
            "restart the app.",
            f"expected path: {model_path}",
        )
    try:
        stamp = model_path.stat().st_mtime
    except OSError as exc:
        return (
            None,
            "The selected model file exists but could not be read.",
            f"{type(exc).__name__}: {exc}",
        )

    try:
        model = load_model(str(model_path), stamp)
    except Exception as exc:
        return (
            None,
            "The selected model file was found but could not be loaded. This "
            "dashboard will not fall back to another model.",
            f"{type(exc).__name__}: {exc}",
        )

    if not getattr(model, "names", None):
        return (
            None,
            "The weights loaded but exposed no class name mapping, so results "
            "could not be labelled honestly.",
            f"object: {type(model).__name__}",
        )
    return model, None, None


def normalize_names(names: Any) -> dict[int, str]:
    """Normalise Ultralytics ``model.names`` into an ordered ``{id: name}`` map."""
    mapping: dict[int, str] = {}
    if isinstance(names, dict):
        items: Any = names.items()
    elif isinstance(names, (list, tuple)):
        items = enumerate(names)
    else:
        return mapping
    for key, value in items:
        try:
            mapping[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return dict(sorted(mapping.items()))


def class_mapping_warnings(class_names: dict[int, str]) -> list[str]:
    """Cross-check the loaded model against the documented dataset class map.

    Purely advisory. Nothing is remapped either way; the model's own names are
    what the dashboard reports.
    """
    messages: list[str] = []
    if len(class_names) != EXPECTED_CLASS_COUNT:
        messages.append(
            f"The loaded model exposes {len(class_names)} classes, while the "
            f"dataset was defined with {EXPECTED_CLASS_COUNT} PCB defect classes."
        )
    for class_id, expected in EXPECTED_CLASS_NAMES.items():
        actual = class_names.get(class_id)
        if actual is None:
            messages.append(
                f"Class ID {class_id} is missing from the model (dataset defined '{expected}')."
            )
        elif actual != expected:
            messages.append(
                f"Class ID {class_id} is '{actual}' in the model, while the dataset "
                f"defined '{expected}'."
            )
    return messages


# ==================================================
# SECTION 6 - DATA STRUCTURES
# ==================================================

@dataclass(frozen=True)
class Detection:
    """One real detection exactly as the trained model returned it."""

    detection_id: int
    class_id: int
    raw_name: str
    display_name: str
    color: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass(frozen=True)
class InspectionSettings:
    """Sidebar values, split into inference parameters and display parameters."""

    conf: float
    iou: float
    max_det: int
    cutoff: float
    show_labels: bool
    show_confidence: bool
    device: str

    def signature(self, image_digest: str) -> str:
        """Fingerprint of the inputs that actually change the model output."""
        payload = (
            f"{image_digest}|{self.conf:.4f}|{self.iou:.4f}|"
            f"{int(self.max_det)}|{IMGSZ}|{self.device}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def changed_inference_fields(self, other: "InspectionSettings") -> list[str]:
        """Human readable list of inference parameters that differ."""
        diffs: list[str] = []
        if abs(self.conf - other.conf) > 1e-9:
            diffs.append(f"confidence {other.conf:.2f} -> {self.conf:.2f}")
        if abs(self.iou - other.iou) > 1e-9:
            diffs.append(f"NMS IoU {other.iou:.2f} -> {self.iou:.2f}")
        if int(self.max_det) != int(other.max_det):
            diffs.append(f"max detections {other.max_det} -> {self.max_det}")
        if self.device != other.device:
            diffs.append(f"device {other.device} -> {self.device}")
        return diffs


@dataclass
class InspectionRecord:
    """The single stored inference result that every output panel consumes."""

    signature: str
    image_digest: str
    filename: str
    image: Image.Image = field(repr=False)
    image_width: int
    image_height: int
    file_size_bytes: int
    timestamp_utc: str
    inference_ms: float
    speed: dict[str, float]
    device: str
    conf: float
    iou: float
    max_det: int
    class_names: dict[int, str]
    detections: list[Detection]

    def counts_by_class(self) -> dict[int, int]:
        counts = {class_id: 0 for class_id in self.class_names}
        for detection in self.detections:
            counts[detection.class_id] = counts.get(detection.class_id, 0) + 1
        return counts

    def mean_confidence_by_class(self) -> dict[int, float]:
        totals: dict[int, float] = {}
        counts: dict[int, int] = {}
        for detection in self.detections:
            totals[detection.class_id] = totals.get(detection.class_id, 0.0) + detection.confidence
            counts[detection.class_id] = counts.get(detection.class_id, 0) + 1
        return {class_id: totals[class_id] / counts[class_id] for class_id in counts}

    def confirmed(self, cutoff: float) -> list[Detection]:
        return [d for d in self.detections if d.confidence >= cutoff]

    def low_confidence(self, cutoff: float) -> list[Detection]:
        return [d for d in self.detections if d.confidence < cutoff]

    def average_confidence(self) -> float | None:
        if not self.detections:
            return None
        return sum(d.confidence for d in self.detections) / len(self.detections)

    def highest_confidence(self) -> float | None:
        if not self.detections:
            return None
        return max(d.confidence for d in self.detections)


# ==================================================
# SECTION 7 - IMAGE LOADING
# ==================================================

def decode_image(data: bytes) -> Image.Image:
    """Decode, EXIF-correct and RGB-convert an uploaded image.

    Handles EXIF rotation, alpha, greyscale, palette and CMYK sources, and reads
    only the first frame of an animated file. The user's original file is never
    modified and never written back.
    """
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.seek(0)
            opened.load()
            oriented = ImageOps.exif_transpose(opened) or opened
            if oriented.mode != "RGB":
                if oriented.mode in ("I;16", "I", "F"):
                    oriented = oriented.point(lambda v: v * 255 // 65535 if v > 255 else v)
                oriented = oriented.convert("RGB")
            return oriented
    except ImageDecodeError:
        raise
    except Exception as exc:
        raise ImageDecodeError(
            "The uploaded file could not be decoded as an image. Please re-export "
            f"it as a standard {ACCEPTED_UPLOAD_LABEL} file."
        ) from exc


def check_image_size(image: Image.Image) -> str | None:
    """Return a friendly message when the image is impractical, else ``None``."""
    width, height = image.size
    if width < 32 or height < 32:
        return (
            f"The image is only {width} x {height} px. That is too small for the "
            "model to find PCB defects reliably. Please upload a higher-resolution "
            "photograph."
        )
    if width * height > MAX_IMAGE_PIXELS:
        return (
            f"The image is {width} x {height} px, which is larger than this "
            "dashboard accepts. Please downscale it before uploading so the "
            "reported bounding boxes stay truthful."
        )
    return None


def get_working_image(data: bytes, digest: str) -> Image.Image:
    """Decode once per distinct upload and reuse the result across reruns."""
    cache = st.session_state.setdefault("pdb_image_cache", {})
    cached = cache.get("image")
    if cache.get("digest") == digest and isinstance(cached, Image.Image):
        return cached
    image = decode_image(data)
    cache["digest"] = digest
    cache["image"] = image
    return image


# ==================================================
# SECTION 8 - INFERENCE
# ==================================================

def run_inference(
    model: YOLO,
    image: Image.Image,
    settings: InspectionSettings,
    signature: str,
    image_digest: str,
    filename: str,
    file_size: int,
) -> InspectionRecord:
    """Run exactly one prediction call and package the real result.

    Timing wraps the ``predict`` call only, so it measures preprocessing, the
    forward pass and postprocessing for this image on the selected device.
    Detections are reordered for display only - identities and scores are
    exactly what the model produced.
    """
    started = time.perf_counter()
    results = model.predict(
        source=image,
        imgsz=IMGSZ,
        conf=settings.conf,
        iou=settings.iou,
        agnostic_nms=False,
        max_det=settings.max_det,
        device=settings.device,
        save=False,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if not results:
        raise RuntimeError("The model returned no result for this image.")

    result = results[0]

    class_names = normalize_names(getattr(result, "names", None) or {})
    if not class_names:
        class_names = normalize_names(getattr(model, "names", None) or {})

    speed: dict[str, float] = {}
    raw_speed = getattr(result, "speed", None)
    if isinstance(raw_speed, dict):
        for key, value in raw_speed.items():
            number = as_number(value)
            if number is not None:
                speed[str(key)] = number

    detections: list[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confs = boxes.conf.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()
        total = min(len(xyxy), len(confs), len(classes))

        order = sorted(range(total), key=lambda i: confs[i], reverse=True)
        for rank, index in enumerate(order, start=1):
            class_id = int(classes[index])
            raw_name = class_names.get(class_id, f"class_{class_id}")
            x1, y1, x2, y2 = (float(v) for v in xyxy[index])
            detections.append(
                Detection(
                    detection_id=rank,
                    class_id=class_id,
                    raw_name=raw_name,
                    display_name=display_label(raw_name),
                    color=CLASS_COLORS.get(class_id, FALLBACK_COLOR),
                    confidence=float(confs[index]),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    width, height = image.size
    return InspectionRecord(
        signature=signature,
        image_digest=image_digest,
        filename=filename,
        image=image,
        image_width=width,
        image_height=height,
        file_size_bytes=file_size,
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        inference_ms=elapsed_ms,
        speed=speed,
        device=settings.device,
        conf=settings.conf,
        iou=settings.iou,
        max_det=settings.max_det,
        class_names=class_names,
        detections=detections,
    )


# ==================================================
# SECTION 9 - ANNOTATION (PIL)
# ==================================================

def _load_font(size: int) -> ImageFont.ImageFont:
    """Best-effort scalable font with a guaranteed fallback."""
    candidates = (
        "segoeuib.ttf",
        "segoeui.ttf",
        "arialbd.ttf",
        "arial.ttf",
        "DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


def _dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    width: int,
    dash: int,
) -> None:
    """Draw a dashed rectangle outline, used for low-confidence detections."""
    x0, y0, x1, y1 = box
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1

    def horizontal(y: int) -> None:
        x = x0
        while x < x1:
            draw.line([(x, y), (min(x + dash, x1), y)], fill=color, width=width)
            x += dash * 2

    def vertical(x: int) -> None:
        y = y0
        while y < y1:
            draw.line([(x, y), (x, min(y + dash, y1))], fill=color, width=width)
            y += dash * 2

    horizontal(y0)
    horizontal(y1)
    vertical(x0)
    vertical(x1)


def dim_image(image: Image.Image, amount: float = 0.32) -> Image.Image:
    """Darken a copy of the image so bright annotation lines read clearly."""
    return Image.blend(image.convert("RGB"), Image.new("RGB", image.size, (4, 8, 16)), amount)


def draw_boxes(
    image: Image.Image,
    detections: list[Detection],
    cutoff: float,
    show_labels: bool,
    show_confidence: bool,
) -> Image.Image:
    """Render the annotated inspection image with PIL.

    Confirmed detections (confidence at or above the display cutoff) get a solid
    thick outline. Low-confidence detections get a thinner, dashed, faded
    outline. Box coordinates come straight from the model and are clamped to the
    frame so a box can never fall outside the image.
    """
    canvas = image.convert("RGB")
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)

    stroke = max(2, int(round(min(width, height) / 420)))
    thin = max(1, stroke - 1)
    dash = max(8, stroke * 5)
    font_px = max(13, int(round(min(width, height) / 30)))
    font = _load_font(font_px)
    pad = max(4, stroke * 2)
    accent_bar = max(3, stroke)
    radius = max(4, stroke * 2)

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    pending_labels: list[tuple[int, int, str, tuple[int, int, int], bool]] = []

    for detection in detections:
        confirmed = detection.confidence >= cutoff
        box_color = hex_to_rgb(detection.color)
        stroke_color = box_color if confirmed else hex_to_rgb(fade_color(detection.color, 0.55))

        x1 = int(round(max(0.0, min(float(width - 1), detection.x1))))
        y1 = int(round(max(0.0, min(float(height - 1), detection.y1))))
        x2 = int(round(max(0.0, min(float(width - 1), detection.x2))))
        y2 = int(round(max(0.0, min(float(height - 1), detection.y2))))
        if x2 <= x1:
            x2 = min(width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(height - 1, y1 + 1)

        if confirmed:
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=stroke)
        else:
            _dashed_rectangle(draw, (x1, y1, x2, y2), stroke_color, thin, dash)

        if not (show_labels or show_confidence):
            continue

        if show_labels and show_confidence:
            text = f"{detection.display_name} {detection.confidence:.2f}"
        elif show_labels:
            text = detection.display_name
        else:
            text = f"{detection.confidence:.2f}"

        text_box = draw.textbbox((0, 0), text, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]

        box_w = text_w + 2 * pad + accent_bar + 4
        box_h = text_h + 2 * pad
        label_x = max(0, min(x1, max(0, width - box_w - 1)))
        label_y = y1 - box_h - 3
        if label_y < 0:
            label_y = min(y1 + 3, max(0, height - box_h - 1))
        label_y = max(0, label_y)

        overlay_draw.rounded_rectangle(
            [label_x, label_y, label_x + box_w, label_y + box_h],
            radius=radius,
            fill=(5, 10, 18, 224),
        )
        overlay_draw.rectangle(
            [label_x, label_y, label_x + accent_bar, label_y + box_h],
            fill=(*box_color, 255),
        )
        pending_labels.append(
            (label_x + accent_bar + pad, label_y + pad - text_box[1], text, box_color, confirmed)
        )

    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    if pending_labels:
        text_draw = ImageDraw.Draw(canvas)
        for tx, ty, text, box_color, confirmed in pending_labels:
            fill = (240, 247, 255) if confirmed else (176, 190, 210)
            text_draw.text((tx, ty), text, font=font, fill=fill)
    return canvas


# ==================================================
# SECTION 10 - METRICS FILES (read-only, no invention)
# ==================================================

METRICS_MISSING_NOTE = (
    "No eval_summary.csv was found. It is expected at reports/metrics/"
    f"{SUMMARY_FILE} (or reports/metrics_hold/{SUMMARY_FILE}). Performance figures "
    "are never estimated or hardcoded, so nothing is shown here."
)


def metrics_dir() -> Path | None:
    """Directory that actually holds the metrics CSVs, or ``None``."""
    for name in METRICS_DIR_NAMES:
        candidate = ROOT / "reports" / name
        if (candidate / SUMMARY_FILE).is_file():
            return candidate
    return None


def read_metrics_csv(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    """Read one metrics CSV, insisting on the exact columns it must have."""
    frame = pd.read_csv(path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} has no column(s): {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{path.name} has no rows")
    return frame


@st.cache_data(show_spinner=False)
def read_metric_tables() -> dict[str, Any]:
    """Read both metrics CSVs once. Returns the frames, or an explanation.

    Keys: ``dir`` (Path or None), ``summary`` (DataFrame or None),
    ``counts`` (DataFrame or None) and ``note`` (str, non-empty on failure).
    """
    directory = metrics_dir()
    if directory is None:
        return {"dir": None, "summary": None, "counts": None, "note": METRICS_MISSING_NOTE}
    try:
        summary = read_metrics_csv(directory / SUMMARY_FILE, SUMMARY_COLUMNS)
    except Exception as exc:
        return {
            "dir": directory,
            "summary": None,
            "counts": None,
            "note": f"{SUMMARY_FILE} could not be read: {type(exc).__name__}: {exc}",
        }
    counts = None
    counts_path = directory / COUNTS_FILE
    if counts_path.is_file():
        try:
            counts = read_metrics_csv(counts_path, COUNTS_COLUMNS)
        except Exception as exc:
            counts = None
            note = f"{COUNTS_FILE} could not be read: {type(exc).__name__}: {exc}"
            return {"dir": directory, "summary": summary, "counts": None, "note": note}
    return {"dir": directory, "summary": summary, "counts": counts, "note": ""}


def summary_models(summary: pd.DataFrame) -> list[str]:
    """Every ``model`` value present in the summary file, in stored order."""
    return [str(value) for value in summary["model"].dropna().unique()]


def summary_splits(summary: pd.DataFrame) -> list[str]:
    """Every ``split`` value present in the summary file, in stored order."""
    return [str(value) for value in summary["split"].dropna().unique()]


def summary_classes(summary: pd.DataFrame) -> list[str]:
    """Every per-class ``cls`` value in the file, in stored order.

    The aggregate ``ALL`` row is excluded. The class list comes from the file
    itself, so a class absent from the chosen split is still listed.
    """
    return [
        str(value)
        for value in summary["cls"].dropna().unique()
        if str(value) != "ALL"
    ]


def summary_subset(summary: pd.DataFrame, model: str, split: str) -> pd.DataFrame:
    """Every stored row for one ``(model, split)`` pair."""
    return summary[(summary["model"] == model) & (summary["split"] == split)]


def overall_row(summary: pd.DataFrame, model: str, split: str) -> pd.Series | None:
    """The ``cls == "ALL"`` row for one ``(model, split)``, or ``None``."""
    matches = summary_subset(summary, model, split)
    matches = matches[matches["cls"] == "ALL"]
    return None if matches.empty else matches.iloc[0]


def per_class_frame(summary: pd.DataFrame, model: str, split: str) -> pd.DataFrame:
    """Per-class table for one ``(model, split)``, one row per class in the file.

    A class with no row for this split is shown as not evaluated. Nothing is
    averaged, filled or carried over from another split.
    """
    stored = summary_subset(summary, model, split)
    stored = stored[stored["cls"] != "ALL"]
    by_class = {str(row["cls"]): row for _, row in stored.iterrows()}
    absent = NOT_EVALUATED.format(split=split)

    rows: list[dict[str, Any]] = []
    for name in summary_classes(summary):
        row = by_class.get(name)
        if row is None:
            entry: dict[str, Any] = {"Defect": display_label(name), "Class name": name}
            for column in ("precision", "recall", "mAP50", "mAP50_95"):
                entry[column] = absent
        else:
            entry = {
                "Defect": display_label(name),
                "Class name": name,
                "precision": format_percent(row["precision"]),
                "recall": format_percent(row["recall"]),
                "mAP50": format_percent(row["mAP50"]),
                "mAP50_95": format_percent(row["mAP50_95"]),
            }
        rows.append(entry)
    return pd.DataFrame(
        rows,
        columns=["Defect", "Class name", "precision", "recall", "mAP50", "mAP50_95"],
    )


def _as_count(value: Any) -> int | None:
    """Integer count from a stored cell, or ``None`` when it is blank."""
    if value is None:
        return None
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    return int(number)


def counts_frame(counts: pd.DataFrame, model: str, split: str) -> tuple[pd.DataFrame, str]:
    """TP/FP/FN table for one ``(model, split)``, plus a provenance note.

    Columns are one group per stored ``iou`` value. A cell is left blank where
    the file has no row, never filled in from elsewhere. The note states the
    confidence threshold the counts were taken at, read from the file.
    """
    subset = counts[(counts["model"] == model) & (counts["split"] == split)]
    if subset.empty:
        return pd.DataFrame(), f"No rows for {model} / {split} in {COUNTS_FILE}."

    def rounded(column: str) -> pd.Series:
        return pd.to_numeric(subset[column], errors="coerce").round(4)

    subset = subset.assign(_iou=rounded("iou"), _conf=rounded("conf"))
    ious = sorted(value for value in subset["_iou"].dropna().unique())
    confs = sorted(value for value in subset["_conf"].dropna().unique())
    classes = [str(value) for value in subset["cls"].dropna().unique()]

    columns = ["Defect", "Class name"]
    for iou in ious:
        for metric in ("TP", "FP", "FN"):
            columns.append(f"{metric} @ IoU {iou:.2f}")

    rows: list[dict[str, Any]] = []
    for name in classes:
        entry: dict[str, Any] = {"Defect": display_label(name), "Class name": name}
        for iou in ious:
            match = subset[(subset["cls"] == name) & (subset["_iou"] == iou)]
            for metric in ("TP", "FP", "FN"):
                cell = None if match.empty else _as_count(match.iloc[0][metric])
                entry[f"{metric} @ IoU {iou:.2f}"] = cell
        rows.append(entry)

    frame = pd.DataFrame(rows, columns=columns)
    total: dict[str, Any] = {"Defect": "Total", "Class name": ""}
    for column in columns[2:]:
        values = [value for value in frame[column] if value is not None]
        total[column] = sum(values) if values else None
    frame = pd.concat([frame, pd.DataFrame([total], columns=columns)], ignore_index=True)

    if len(confs) == 1:
        conf_text = f"a confidence threshold of {confs[0]:.2f}"
    else:
        listed = ", ".join(f"{value:.2f}" for value in confs)
        conf_text = f"the confidence thresholds stored in the file ({listed})"
    iou_text = (
        f"the single IoU of {ious[0]:.2f}"
        if len(ious) == 1
        else "each of the stored IoU values "
        + ", ".join(f"{value:.2f}" for value in ious)
    )
    note = (
        f"Counts for {model} / {split} as stored in {COUNTS_FILE}, taken at "
        f"{conf_text} and at {iou_text}. A blank cell means the file has no row "
        "for that class at that IoU. A class the split does not annotate has no "
        "true positives and no false negatives by construction, so its row is "
        "not a score; any false positives on it are the model firing on a class "
        "the split never labels."
    )
    return frame, note


# ==================================================
# SECTION 11 - CSV REPORT
# ==================================================

REPORT_FIELDS = [
    "section",
    "image_filename",
    "inspection_timestamp_utc",
    "key",
    "value",
    "detection_id",
    "class_id",
    "model_class_name",
    "display_name",
    "confidence",
    "confidence_level",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
    "area_px",
]


def build_csv_report(record: InspectionRecord, cutoff: float) -> bytes:
    """Build the in-memory CSV report on one stable schema.

    ``metadata`` rows describe the inspection, ``class_summary`` rows give the
    count for every class the model exposes, and ``detection`` rows list the real
    detections. The header and the metadata block are always present, even when
    there are no detections at all.
    """
    confirmed = len(record.confirmed(cutoff))
    low = len(record.low_confidence(cutoff))
    average = record.average_confidence()
    highest = record.highest_confidence()
    counts = record.counts_by_class()

    metadata: list[tuple[str, str]] = [
        ("product", f"{APP_NAME} {APP_VERSION}"),
        ("disclaimer", PROTOTYPE_NOTICE),
        ("image_filename", record.filename),
        ("image_width_px", str(record.image_width)),
        ("image_height_px", str(record.image_height)),
        ("image_file_size_bytes", str(record.file_size_bytes)),
        ("image_sha256_prefix", record.image_digest),
        ("inspection_timestamp_utc", record.timestamp_utc),
        ("model_file", active_model_path().name),
        ("model_classes", str(len(record.class_names))),
        ("inference_device", record.device),
        ("imgsz", str(IMGSZ)),
        ("confidence_threshold_used", f"{record.conf:.4f}"),
        ("nms_iou_used", f"{record.iou:.4f}"),
        ("max_detections_used", str(record.max_det)),
        ("display_low_confidence_cutoff", f"{cutoff:.4f}"),
        ("inference_time_ms", f"{record.inference_ms:.1f}"),
        ("total_detections", str(len(record.detections))),
        ("confirmed_detections", str(confirmed)),
        ("low_confidence_detections", str(low)),
        ("average_confidence", "" if average is None else f"{average:.4f}"),
        ("highest_confidence", "" if highest is None else f"{highest:.4f}"),
    ]
    for stage in ("preprocess", "inference", "postprocess"):
        if stage in record.speed:
            metadata.append((f"model_reported_{stage}_ms", f"{record.speed[stage]:.1f}"))

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=REPORT_FIELDS, restval="", lineterminator="\r\n")
    writer.writeheader()

    def base(section: str) -> dict[str, str]:
        return {
            "section": section,
            "image_filename": record.filename,
            "inspection_timestamp_utc": record.timestamp_utc,
        }

    for key, value in metadata:
        writer.writerow({**base("metadata"), "key": key, "value": value})

    for class_id, raw_name in record.class_names.items():
        writer.writerow(
            {
                **base("class_summary"),
                "key": f"count_{raw_name}",
                "value": str(counts.get(class_id, 0)),
                "class_id": class_id,
                "model_class_name": raw_name,
                "display_name": display_label(raw_name),
            }
        )

    for detection in record.detections:
        level = "Confirmed" if detection.confidence >= cutoff else "Low Confidence"
        writer.writerow(
            {
                **base("detection"),
                "detection_id": detection.detection_id,
                "class_id": detection.class_id,
                "model_class_name": detection.raw_name,
                "display_name": detection.display_name,
                "confidence": f"{detection.confidence:.4f}",
                "confidence_level": level,
                "x1": f"{detection.x1:.1f}",
                "y1": f"{detection.y1:.1f}",
                "x2": f"{detection.x2:.1f}",
                "y2": f"{detection.y2:.1f}",
                "width": f"{detection.width:.1f}",
                "height": f"{detection.height:.1f}",
                "area_px": f"{detection.area:.0f}",
            }
        )

    return buffer.getvalue().encode("utf-8")


# ==================================================
# SECTION 12 - UI: HEADER
# ==================================================

def render_header(model_loaded: bool, class_names: dict[int, str] | None = None) -> None:
    """Product header with a real model status pill."""
    if model_loaded:
        pill = (
            '<span class="pdb-pill pdb-pill--ok"><span class="pdb-dot"></span>'
            "MODEL LOADED</span>"
        )
    else:
        pill = (
            '<span class="pdb-pill pdb-pill--bad"><span class="pdb-dot"></span>'
            "MODEL NOT LOADED</span>"
        )

    subtitle = esc(APP_SUBTITLE)
    if model_loaded and class_names:
        subtitle = f"{subtitle} &middot; {len(class_names)} CLASSES &middot; {IMGSZ} PX"

    html(
        '<div class="pdb-header">'
        f'<div class="pdb-logo">{SVG_LOGO}</div>'
        '<div class="pdb-head-text">'
        f'<h1 class="pdb-title">{esc(APP_NAME)}</h1>'
        f'<p class="pdb-subtitle">{subtitle}</p>'
        '<p class="pdb-tagline">Automatic bare-board defect localisation, '
        "classification and confidence reporting for PCB manufacturing.</p>"
        "</div>"
        f"{pill}"
        "</div>"
        '<div class="pdb-rule"></div>'
    )


# ==================================================
# SECTION 13 - UI: SIDEBAR
# ==================================================

def render_model_selector() -> None:
    """Small sidebar selector for which trained weight file to run.

    Must be called before the model is loaded. Switching weights discards the
    stored inspection result, because detections from one checkpoint must never
    be presented next to another.
    """
    keys = [choice[0] for choice in MODEL_CHOICES]
    chosen = st.session_state.get("pdb_model_key")
    if chosen not in keys:
        chosen = default_model_key()
        st.session_state["pdb_model_key"] = chosen

    st.sidebar.markdown('<div class="pdb-sb-group">Model</div>', unsafe_allow_html=True)
    picked = st.sidebar.selectbox(
        "Weights file",
        options=keys,
        format_func=lambda key: f"{model_choice(key)[1]} - {model_choice(key)[2]}",
        key="pdb_model_key",
        help=(
            "Which trained checkpoint this dashboard runs and which rows the "
            "performance panel shows. Both are real trained weight files in the "
            "models folder; nothing is downloaded or substituted."
        ),
    )

    if st.session_state.get("pdb_model_applied_key") not in (None, picked):
        st.session_state.pop("pdb_inspection", None)
        st.session_state.pop("pdb_image_cache", None)
    st.session_state["pdb_model_applied_key"] = picked

    path = model_path_for(picked)
    if not path.is_file():
        st.sidebar.warning(f"{path.name} was not found in the models folder.")
    st.sidebar.markdown(
        '<div class="pdb-meta" style="margin-top:-0.35rem;">'
        f'<span class="pdb-meta-chip">RUNNING <b>{esc(path.name)}</b></span></div>',
        unsafe_allow_html=True,
    )


def render_sidebar() -> InspectionSettings:
    """Build the settings sidebar and return the current values."""
    devices = available_devices()
    default_device = "cpu"

    st.sidebar.markdown(
        '<div class="pdb-sb-title">Inspection Settings</div>'
        '<div style="font-size:0.79rem;color:#93A4BD;margin:0.15rem 0 0.2rem 0;">'
        "Everything here is passed straight to the trained model or used only for "
        "presentation.</div>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<div class="pdb-sb-group">Detection</div>', unsafe_allow_html=True)
    confidence = st.sidebar.slider(
        "Detection confidence",
        min_value=0.05,
        max_value=0.90,
        value=DEFAULT_CONFIDENCE,
        step=0.05,
        format="%.2f",
        key="pdb_conf",
        help=(
            "Minimum confidence a detection needs before it is reported. The "
            "default is 0.10 rather than 0.15 because the fine-tuned model "
            "scores its true detections lower, so the higher threshold dropped "
            "real defects. Lower values surface more detections but can raise "
            "false positives."
        ),
    )
    iou = st.sidebar.slider(
        "NMS IoU threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        format="%.2f",
        key="pdb_iou",
        help=(
            "Overlap tolerance for Non-Maximum Suppression, sent to the model as "
            "its inference IoU. Lower values suppress overlapping boxes more "
            "aggressively."
        ),
    )
    max_det = st.sidebar.number_input(
        "Maximum detections",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        key="pdb_max_det",
        help=(
            "Upper bound on how many boxes the model may return. Raise it only if "
            "a dense board looks clipped."
        ),
    )

    st.sidebar.markdown('<div class="pdb-sb-group">Runtime</div>', unsafe_allow_html=True)
    st.sidebar.number_input(
        "Input image size (locked)",
        min_value=IMGSZ,
        max_value=IMGSZ,
        value=IMGSZ,
        step=IMGSZ,
        disabled=True,
        key="pdb_imgsz",
        help=(
            f"Fixed at {IMGSZ} px, the size the model was trained and evaluated at. "
            "Running at another size would make the displayed results inconsistent "
            "with training, so it cannot be changed here."
        ),
    )
    device = st.sidebar.selectbox(
        "Inference device",
        options=devices,
        index=devices.index(default_device),
        key="pdb_device",
        help=(
            "CPU is the default so results are reproducible on any machine. "
            "CUDA is offered only when a working GPU is actually detected."
        ),
    )
    if device != "cpu":
        st.sidebar.markdown(
            '<div class="pdb-note pdb-note--warn" style="font-size:0.76rem;">'
            "GPU inference selected. Speed will improve, but a CPU run of the same "
            "image and settings can differ slightly in the detections it returns."
            "</div>",
            unsafe_allow_html=True,
        )

    st.sidebar.markdown('<div class="pdb-sb-group">Presentation</div>', unsafe_allow_html=True)
    cutoff = st.sidebar.slider(
        "Low-confidence cutoff",
        min_value=0.00,
        max_value=1.00,
        value=0.35,
        step=0.01,
        format="%.2f",
        key="pdb_cutoff",
        help=(
            "Display only. Detections at or above this confidence are drawn solid "
            "and labelled Confirmed; the rest are drawn dashed. It never changes a "
            "class ID, a confidence value or anything the model computed."
        ),
    )
    show_labels = st.sidebar.toggle(
        "Show class labels", value=True, key="pdb_show_labels",
        help="Draw the defect name above each bounding box.",
    )
    show_confidence = st.sidebar.toggle(
        "Show confidence values", value=True, key="pdb_show_conf",
        help="Append the real confidence score to each bounding box label.",
    )

    st.sidebar.markdown('<div class="pdb-sb-group">Session</div>', unsafe_allow_html=True)
    if st.sidebar.button("Clear stored result", width="stretch"):
        st.session_state.pop("pdb_inspection", None)
        st.session_state.pop("pdb_image_cache", None)
        st.rerun()

    st.sidebar.markdown('<div class="pdb-sb-group">Notes</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<div class="pdb-note" style="font-size:0.78rem;">{esc(THRESHOLD_CAVEAT)}</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f'<div class="pdb-note pdb-note--warn" style="font-size:0.78rem;">{esc(PROTOTYPE_NOTICE)}</div>',
        unsafe_allow_html=True,
    )

    return InspectionSettings(
        conf=float(confidence),
        iou=float(iou),
        max_det=int(max_det),
        cutoff=float(cutoff),
        show_labels=bool(show_labels),
        show_confidence=bool(show_confidence),
        device=str(device),
    )


# ==================================================
# SECTION 14 - UI: UPLOAD + EMPTY STATE
# ==================================================

def render_empty_state() -> None:
    html(
        '<div class="pdb-empty">'
        f"{SVG_EMPTY_BOARD}"
        '<p class="pdb-empty-title">Ready for PCB Inspection</p>'
        '<p class="pdb-empty-text">Upload a photograph of a bare printed circuit '
        "board. The trained model will localise and classify defects across every "
        "class it was trained on, reporting a confidence score and a bounding box "
        "for each finding.</p>"
        '<div class="pdb-chips">'
        '<span class="pdb-chip">JPG / JPEG</span>'
        '<span class="pdb-chip">PNG</span>'
        '<span class="pdb-chip">BMP</span>'
        f'<span class="pdb-chip">{IMGSZ} px inference</span>'
        '<span class="pdb-chip">CPU by default</span>'
        '<span class="pdb-chip">EXIF corrected</span>'
        "</div></div>"
    )


def render_meta_strip(filename: str, size_px: tuple[int, int], file_size: int) -> None:
    width, height = size_px
    html(
        '<div class="pdb-meta">'
        f'<span class="pdb-meta-chip">FILE <b>{esc(filename)}</b></span>'
        f'<span class="pdb-meta-chip">SIZE <b>{width} &times; {height} px</b></span>'
        f'<span class="pdb-meta-chip">UPLOAD <b>{esc(format_bytes(file_size))}</b></span>'
        f'<span class="pdb-meta-chip">MODEL <b>{esc(active_model_path().name)}</b></span>'
        "</div>"
    )


# ==================================================
# SECTION 15 - UI: BANNER + KPIs
# ==================================================

def render_result_banner(record: InspectionRecord, cutoff: float) -> None:
    """Prominent result banner. Never claims a board is defect-free."""
    total = len(record.detections)
    confirmed = len(record.confirmed(cutoff))
    low = total - confirmed

    if total > 0:
        classes_hit = len({d.class_id for d in record.detections})
        banner_class = "pdb-banner--danger"
        title = "DEFECTS DETECTED"
        icon = ICON_ALERT_BANNER
        badge = f"{total} FINDING{'' if total == 1 else 'S'}"
        sub = (
            f"{confirmed} confirmed at or above the {cutoff:.2f} cutoff &middot; "
            f"{low} below it &middot; {classes_hit} of {len(record.class_names)} "
            "trained classes present in this image"
        )
    else:
        banner_class = "pdb-banner--ok"
        title = "NO DEFECTS DETECTED AT CURRENT SETTINGS"
        icon = ICON_SHIELD_OK
        badge = "0 FINDINGS"
        sub = (
            f"The model returned no boxes at confidence &ge; {record.conf:.2f} and "
            f"NMS IoU {record.iou:.2f}. Nothing was flagged - that is not proof "
            "that the board is defect-free."
        )

    html(
        f'<div class="pdb-banner {banner_class}">'
        f'<div class="pdb-banner-icon">{icon}</div>'
        '<div class="pdb-banner-text">'
        f'<p class="pdb-banner-title">{title}</p>'
        f'<p class="pdb-banner-sub">{sub}</p>'
        "</div>"
        f'<span class="pdb-banner-badge">{badge}</span>'
        "</div>"
        f'<div class="pdb-note pdb-note--warn">{esc(PROTOTYPE_NOTICE)}</div>'
    )


def _kpi_card(label: str, value: str, hint: str, accent: str, icon: str) -> str:
    return (
        f'<div class="pdb-kpi" style="--pdb-kpi-accent: {accent};">'
        f'<div class="pdb-kpi-top"><span class="pdb-kpi-ic">{icon}</span>'
        f'<span class="pdb-kpi-label">{esc(label)}</span></div>'
        f'<div class="pdb-kpi-value">{esc(value)}</div>'
        f'<div class="pdb-kpi-hint">{esc(hint)}</div>'
        "</div>"
    )


def render_kpis(record: InspectionRecord, cutoff: float) -> None:
    """Five KPI cards, every value derived from the one stored result."""
    total = len(record.detections)
    confirmed = len(record.confirmed(cutoff))
    low = total - confirmed
    average = record.average_confidence()
    highest = record.highest_confidence()
    classes_hit = len({d.class_id for d in record.detections})

    cards = [
        _kpi_card(
            "Total detections", str(total),
            "Boxes returned after conf + NMS", UI["accent"], ICON_CROSSHAIR,
        ),
        _kpi_card(
            "Confirmed", str(confirmed),
            f"Confidence at or above {cutoff:.2f}", UI["ok"], ICON_CHECK,
        ),
        _kpi_card(
            "Low confidence", str(low),
            f"Below {cutoff:.2f}, at or above {record.conf:.2f}", UI["warn"], ICON_WARN,
        ),
        _kpi_card(
            "Classes hit", f"{classes_hit}/{len(record.class_names)}",
            "Trained classes present in this image", UI["violet"], ICON_LAYERS,
        ),
        _kpi_card(
            "Inference time", f"{record.inference_ms:.0f} ms",
            f"One predict() call on {record.device}", UI["accent2"], ICON_CLOCK,
        ),
        _kpi_card(
            "Confidence",
            f"{highest:.3f}" if highest is not None else "N/A",
            f"Highest of {len(record.detections)} detection(s)"
            if highest is not None
            else "No detections to score",
            UI["yellow"], ICON_GAUGE,
        ),
    ]
    html('<div class="pdb-kpi-grid">' + "".join(cards) + "</div>")

    if record.speed:
        parts = [
            f"{esc(stage)} {record.speed[stage]:.1f} ms"
            for stage in ("preprocess", "inference", "postprocess")
            if stage in record.speed
        ]
        if parts:
            html(
                '<div class="pdb-note" style="margin-top:-0.15rem;">'
                "Stage timings as reported by the model itself: "
                + " &middot; ".join(parts)
                + f". Wall clock for the whole call: {record.inference_ms:.1f} ms."
                "</div>"
            )


# ==================================================
# SECTION 16 - UI: COMPARISON + LEGEND
# ==================================================

def render_comparison(record: InspectionRecord, settings: InspectionSettings) -> None:
    """Original versus annotated views, with a dimmed-context view."""
    annotated = draw_boxes(
        record.image,
        record.detections,
        cutoff=settings.cutoff,
        show_labels=settings.show_labels,
        show_confidence=settings.show_confidence,
    )
    enhanced = draw_boxes(
        dim_image(record.image),
        record.detections,
        cutoff=settings.cutoff,
        show_labels=settings.show_labels,
        show_confidence=settings.show_confidence,
    )

    tab_compare, tab_annotated, tab_enhanced, tab_legend = st.tabs(
        ["Original vs Annotated", "Annotated (Full Width)", "Enhanced Context", "Legend"]
    )

    def show_annotated(target: Any, image_obj: Image.Image) -> None:
        if record.detections:
            target.image(image_obj, width="stretch")
        else:
            target.image(record.image, width="stretch")
            html(
                '<div class="pdb-note" style="margin-top:0.6rem;">'
                "No boxes to draw at the current settings, so the original image "
                "is shown unchanged. An absence of detections is not a "
                "certification that the board is defect-free."
                "</div>"
            )

    with tab_compare:
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown("**Original board**")
            st.image(record.image, width="stretch")
        with right:
            st.markdown("**AI inspection result**")
            show_annotated(st, annotated)

    with tab_annotated:
        st.markdown("**AI inspection result, full width**")
        show_annotated(st, annotated)

    with tab_enhanced:
        st.markdown("**Dimmed context view**")
        html(
            '<div class="pdb-note" style="margin-bottom:0.5rem;">The board is '
            "darkened in this view only. The boxes, coordinates and confidences are "
            "identical to the other tabs.</div>"
        )
        show_annotated(st, enhanced)

    with tab_legend:
        render_legend(record)


def render_legend(record: InspectionRecord) -> None:
    """Class colour legend plus the confirmed / low-confidence line key."""
    counts = record.counts_by_class()
    items: list[str] = []
    for class_id, raw_name in record.class_names.items():
        color = CLASS_COLORS.get(class_id, FALLBACK_COLOR)
        items.append(
            '<div class="pdb-legend-item">'
            f'<div class="pdb-legend-swatch" style="background:{color};"></div>'
            f'<div class="pdb-legend-text"><b>{esc(display_label(raw_name))}</b>'
            f" &middot; id {class_id} &middot; {esc(raw_name)}"
            f' &middot; {counts.get(class_id, 0)} found</div></div>'
        )
    html('<div class="pdb-legend-grid">' + "".join(items) + "</div>")
    html(
        '<div class="pdb-legend-grid">'
        '<div class="pdb-legend-item"><div class="pdb-line-key"></div>'
        '<div class="pdb-legend-key">Solid, thick outline - confirmed detection, '
        "at or above the low-confidence cutoff.</div></div>"
        '<div class="pdb-legend-item"><div class="pdb-line-key pdb-line-key--dashed"></div>'
        '<div class="pdb-legend-key">Dashed, faded outline - low-confidence '
        "detection. Review these manually.</div></div>"
        "</div>"
    )


# ==================================================
# SECTION 17 - UI: CLASS SUMMARY + CHARTS
# ==================================================

def render_class_summary(record: InspectionRecord) -> pd.DataFrame:
    """One card per class the model exposes, including zero-count classes."""
    counts = record.counts_by_class()
    means = record.mean_confidence_by_class()

    rows: list[dict[str, Any]] = [
        {
            "class_id": class_id,
            "model_class_name": raw_name,
            "Defect": display_label(raw_name),
            "Detections": int(counts.get(class_id, 0)),
            "Mean confidence": (
                round(means[class_id], 4) if class_id in means else None
            ),
            "Color": CLASS_COLORS.get(class_id, FALLBACK_COLOR),
        }
        for class_id, raw_name in record.class_names.items()
    ]
    frame = pd.DataFrame(
        rows,
        columns=["class_id", "model_class_name", "Defect", "Detections", "Mean confidence", "Color"],
    )

    cards: list[str] = []
    for row in rows:
        color = row["Color"]
        count = row["Detections"]
        mean = row["Mean confidence"]
        zero_class = " pdb-class--zero" if count == 0 else ""
        mean_html = (
            f'<div class="pdb-class-conf">mean {mean:.3f}</div>'
            if mean is not None
            else '<div class="pdb-class-conf">no detections</div>'
        )
        cards.append(
            f'<div class="pdb-class{zero_class}" style="--pdb-class-color:{color};">'
            f'<div class="pdb-class-dot" style="background:{color};"></div>'
            '<div class="pdb-class-meta">'
            f'<div class="pdb-class-name">{esc(row["Defect"])}</div>'
            f'<div class="pdb-class-id">ID {row["class_id"]} &middot; {esc(row["model_class_name"])}</div>'
            f"{mean_html}</div>"
            f'<div class="pdb-class-count">{count}</div>'
            "</div>"
        )
    html('<div class="pdb-class-grid">' + "".join(cards) + "</div>")
    return frame


def _axis_style(**kwargs: Any) -> alt.Axis:
    return alt.Axis(**kwargs)


def render_count_chart(frame: pd.DataFrame) -> None:
    """Horizontal bar chart of real counts, every class always present."""
    label_order = list(frame["Defect"])
    base = alt.Chart(frame).encode(
        y=alt.Y(
            "Defect:N",
            sort=label_order,
            title=None,
            axis=_axis_style(
                labelColor="#CBD5E1",
                titleColor=None,
                tickColor="rgba(0,0,0,0)",
                domainColor="rgba(0,0,0,0)",
                gridColor="rgba(0,0,0,0)",
                labelFontSize=12,
                labelPadding=8,
            ),
        ),
        x=alt.X(
            "Detections:Q",
            title="Detections",
            axis=_axis_style(
                labelColor="#8CA0BA",
                titleColor="#8CA0BA",
                gridColor="rgba(148,163,184,0.14)",
                domainColor="rgba(148,163,184,0.25)",
                tickColor="rgba(0,0,0,0)",
                labelFontSize=11,
            ),
        ),
    )
    bars = base.mark_bar(height=17, cornerRadius=4).encode(color=alt.Color("Color:N", legend=None))
    labels = base.mark_text(align="left", dx=6, fontSize=12, color="#E2E8F0").encode(
        text=alt.Text("Detections:Q")
    )
    chart = (bars + labels).properties(height=alt.Step(34)).configure_view(strokeWidth=0)
    st.altair_chart(chart, width="stretch", theme=None)


def render_confidence_chart(record: InspectionRecord) -> None:
    """Per-detection confidence bars, coloured by class."""
    if not record.detections:
        html(
            '<div class="pdb-note">There are no detections at the current '
            "settings, so there is no confidence to plot.</div>"
        )
        return

    frame = pd.DataFrame(
        [
            {
                "Detection": f"#{d.detection_id}",
                "Defect": d.display_name,
                "Confidence": round(d.confidence, 4),
                "Color": d.color,
            }
            for d in record.detections
        ]
    )

    base = alt.Chart(frame).encode(
        x=alt.X(
            "Detection:N",
            sort=list(frame["Detection"]),
            title=None,
            axis=_axis_style(labelColor="#8CA0BA", labelAngle=0, tickColor="rgba(0,0,0,0)"),
        ),
        y=alt.Y(
            "Confidence:Q",
            title="Confidence",
            scale=alt.Scale(domain=[0, 1]),
            axis=_axis_style(
                labelColor="#8CA0BA",
                titleColor="#8CA0BA",
                gridColor="rgba(148,163,184,0.14)",
                domainColor="rgba(148,163,184,0.25)",
                tickColor="rgba(0,0,0,0)",
            ),
        ),
    )
    bars = base.mark_bar(width=14, cornerRadius=3).encode(color=alt.Color("Color:N", legend=None))
    rule = alt.Chart(pd.DataFrame({"y": [record.conf]})).mark_rule(color="#FBBF24", strokeWidth=1.5, strokeDash=[5, 4]).encode(y="y:Q")
    chart = (bars + rule).properties(height=alt.Step(30)).configure_view(strokeWidth=0)
    st.altair_chart(chart, width="stretch", theme=None)
    html(
        '<div class="pdb-note" style="margin-top:0.35rem;">'
        f"The dashed line is the model's own confidence filter at {record.conf:.2f}. "
        "Bars are coloured by the class of that detection.</div>"
    )


# ==================================================
# SECTION 18 - UI: DETECTION TABLE
# ==================================================

def render_detection_table(record: InspectionRecord, cutoff: float) -> None:
    """Sortable detection register built from the stored prediction only."""
    if not record.detections:
        html(
            '<div class="pdb-note">No detections to tabulate at the current '
            "settings, so there is nothing to list. Lower the confidence "
            "threshold or the NMS IoU in the sidebar and run the inspection again."
            "</div>"
        )
        return

    frame = pd.DataFrame(
        [
            {
                "Detection ID": d.detection_id,
                "Defect": d.display_name,
                "Model class name": d.raw_name,
                "Class ID": d.class_id,
                "Confidence": round(d.confidence, 4),
                "Confidence Progress": round(d.confidence, 4),
                "Level": "Confirmed" if d.confidence >= cutoff else "Low Confidence",
                "X1": round(d.x1, 1),
                "Y1": round(d.y1, 1),
                "X2": round(d.x2, 1),
                "Y2": round(d.y2, 1),
                "Width": round(d.width, 1),
                "Height": round(d.height, 1),
                "Area px": round(d.area, 0),
                "Colour": d.color,
            }
            for d in record.detections
        ]
    )

    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        height=min(440, 38 * len(frame) + 46),
        column_config={
            "Detection ID": st.column_config.NumberColumn("Detection ID", format="%d", width="small"),
            "Defect": st.column_config.TextColumn("Defect", width="medium"),
            "Model class name": st.column_config.TextColumn("Model class name", width="medium"),
            "Class ID": st.column_config.NumberColumn("Class ID", format="%d", width="small"),
            "Confidence": st.column_config.NumberColumn("Confidence", format="%.3f", width="small"),
            "Confidence Progress": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.2f", width="small"
            ),
            "Level": st.column_config.TextColumn("Level", width="small"),
            "X1": st.column_config.NumberColumn("X1", format="%.1f", width="small"),
            "Y1": st.column_config.NumberColumn("Y1", format="%.1f", width="small"),
            "X2": st.column_config.NumberColumn("X2", format="%.1f", width="small"),
            "Y2": st.column_config.NumberColumn("Y2", format="%.1f", width="small"),
            "Width": st.column_config.NumberColumn("Width", format="%.1f", width="small"),
            "Height": st.column_config.NumberColumn("Height", format="%.1f", width="small"),
            "Area px": st.column_config.NumberColumn("Area px", format="%.0f", width="small"),
            "Colour": st.column_config.TextColumn("Colour", width="small"),
        },
    )
    html(
        '<div style="font-size:0.73rem;color:#6C7E99;margin-top:0.35rem;">'
        "Every value above comes straight from the model. The Colour column "
        "carries the exact hex used for that class in the boxes, labels, cards, "
        "chart, legend and CSV. Coordinates are pixels on the uploaded image; "
        "width is X2 - X1, height is Y2 - Y1, area is their product."
        "</div>"
    )


# ==================================================
# SECTION 19 - UI: EXPORT
# ==================================================

def render_downloads(record: InspectionRecord, settings: InspectionSettings) -> None:
    """Annotated PNG and CSV report, both built fully in memory."""
    annotated = draw_boxes(
        record.image,
        record.detections,
        cutoff=settings.cutoff,
        show_labels=settings.show_labels,
        show_confidence=settings.show_confidence,
    )
    image_buffer = io.BytesIO()
    annotated.save(image_buffer, format="PNG")
    png_bytes = image_buffer.getvalue()

    csv_bytes = build_csv_report(record, settings.cutoff)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(record.filename).stem).strip("_") or "pcb"
    stamp = record.timestamp_utc.replace("-", "").replace(":", "").replace("T", "_").rstrip("Z")

    html(
        f'<div class="pdb-meta">'
        f'<span class="pdb-meta-chip">ANNOTATED PNG <b>{esc(format_bytes(len(png_bytes)))}</b></span>'
        f'<span class="pdb-meta-chip">CSV REPORT <b>{esc(format_bytes(len(csv_bytes)))}</b></span>'
        f'<span class="pdb-meta-chip">GENERATED <b>{esc(record.timestamp_utc)}</b></span>'
        "</div>"
    )

    left, right = st.columns(2)
    with left, st.container(key="pdb-dl-img"):
        st.download_button(
            "Download annotated image (PNG)",
            data=png_bytes,
            file_name=f"pcb_annotated_{stem}_{stamp}.png",
            mime="image/png",
            type="primary",
            width="stretch",
            icon=":material/download:",
            help="Exactly the annotated board shown above, honouring the current label and confidence toggles.",
        )
    with right:
        st.download_button(
            "Download inspection report (CSV)",
            data=csv_bytes,
            file_name=f"pcb_inspection_{stem}_{stamp}.csv",
            mime="text/csv",
            width="stretch",
            icon=":material/download:",
            help=(
                "One stable schema: metadata rows, a class summary row per trained "
                "class, then one row per detection. The header and metadata block "
                "are present even when nothing was detected."
            ),
        )

    html(
        '<div class="pdb-note" style="margin-top:0.6rem;">'
        "Both files are generated in memory inside this session. Nothing is "
        "written into the project folders and your original upload is never "
        "modified.</div>"
    )


# ==================================================
# SECTION 20 - UI: MODEL + METRICS
# ==================================================

def _metric_tile(label: str, value: str) -> str:
    available = value not in {"Not reported", "Not yet evaluated"}
    css_class = "pdb-metric-v" if available else "pdb-metric-v pdb-metric-v--na"
    return (
        f'<div class="pdb-metric"><div class="{css_class}">{esc(value)}</div>'
        f'<div class="pdb-metric-k">{esc(label)}</div></div>'
    )


def _short_reason(exc: Exception, limit: int = 130) -> str:
    """One short line describing a failure, safe to drop into a warning."""
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_metrics_panel() -> None:
    """Offline performance figures straight from the two metrics CSVs.

    Wrapped so that no metrics problem can ever take the dashboard down. Any
    failure becomes a small warning in this tab and nothing else changes.
    """
    try:
        _render_metrics_panel()
    except Exception as exc:
        st.warning(f"Performance panel unavailable: {_short_reason(exc)}")


def _render_metrics_panel() -> None:
    data = read_metric_tables()
    summary = data["summary"]
    directory = data["dir"]
    note = data["note"]

    if summary is None:
        html(
            '<div class="pdb-note pdb-note--warn">'
            "<b>No performance data available.</b><br>"
            + esc(note)
            + "</div>"
        )
        return

    st.caption(
        f"Reading {directory.relative_to(ROOT) if directory else '?'} / {SUMMARY_FILE} "
        f"- {len(summary)} stored row(s)."
    )

    model_options = summary_models(summary)
    if not model_options:
        raise ValueError(f"{SUMMARY_FILE} has no model values")
    split_options = summary_splits(summary)
    if not split_options:
        raise ValueError(f"{SUMMARY_FILE} has no split values")

    active_label = metrics_model_label()
    model_index = model_options.index(active_label) if active_label in model_options else 0
    split_index = split_options.index("test") if "test" in split_options else 0

    left, right = st.columns(2)
    with left:
        model = st.selectbox(
            "Model",
            options=model_options,
            index=model_index,
            key="pdb_metric_model",
            help=(
                "Model labels taken directly from the 'model' column of "
                f"{SUMMARY_FILE}. This only changes the rows shown; use the "
                "sidebar selector to change the weights the dashboard runs."
            ),
        )
    with right:
        split = st.selectbox(
            "Split",
            options=split_options,
            index=split_index,
            key="pdb_metric_split",
            help=f"Taken directly from the 'split' column of {SUMMARY_FILE}.",
        )

    if not isinstance(model, str) or not isinstance(split, str):
        raise TypeError("model and split must be strings")

    row = overall_row(summary, model, split)
    if row is None:
        raise ValueError(f"no cls == 'ALL' row for {model} / {split}")
    html(
        '<div class="pdb-metric-row">'
        + _metric_tile("Precision", format_percent(row["precision"]))
        + _metric_tile("Recall", format_percent(row["recall"]))
        + _metric_tile("mAP@50", format_percent(row["mAP50"]))
        + _metric_tile("mAP@50-95", format_percent(row["mAP50_95"]))
        + "</div>"
    )
    html(
        '<div class="pdb-note">Overall row (<code>cls = ALL</code>) for model '
        f"<b>{esc(model)}</b>, split <b>{esc(split)}</b>, exactly as stored in "
        f"<code>{esc(SUMMARY_FILE)}</code>.</div>"
    )

    if active_label is not None and model != active_label:
        html(
            '<div class="pdb-note pdb-note--stale">You are looking at a '
            f"different model (<b>{esc(model)}</b>) from the one the dashboard "
            f"is running (<b>{esc(active_label)}</b>, "
            f"<code>{esc(active_model_path().name)}</code>).</div>"
        )

    st.markdown("**Per-class results**")
    st.dataframe(
        per_class_frame(summary, model, split),
        hide_index=True,
        width="stretch",
    )
    html(
        '<div class="pdb-note" style="font-size:0.78rem;">One row per class '
        f"present in <code>{esc(SUMMARY_FILE)}</code>. A class with no row for "
        f"the <b>{esc(split)}</b> split is marked as not evaluated rather than "
        "being given a borrowed or estimated value.</div>"
    )

    st.markdown("**True positives, false positives and false negatives**")
    counts = data["counts"]
    if counts is None:
        html('<div class="pdb-note pdb-note--warn">' + esc(note or
             f"{COUNTS_FILE} was not found, so no counts are shown.") + "</div>")
    else:
        frame, counts_note = counts_frame(counts, model, split)
        if frame.empty:
            html('<div class="pdb-note pdb-note--warn">' + esc(counts_note) + "</div>")
        else:
            st.dataframe(frame, hide_index=True, width="stretch")
            html('<div class="pdb-note" style="font-size:0.78rem;">' + esc(counts_note) + "</div>")

    html(
        '<div class="pdb-note pdb-note--warn">These are offline dataset '
        "evaluation figures produced by a separate evaluation script. They are "
        "not metrics for the image on screen, and precision, recall and mAP "
        "cannot be derived from a single image's detections. Every number above "
        "is read from the CSV; none is estimated or hardcoded.</div>"
    )


def render_model_information(
    class_names: dict[int, str],
    record: InspectionRecord | None,
) -> None:
    """Model identity, class map and genuine offline metrics."""
    with st.expander("Model & Performance", expanded=False, icon=":material/engineering:"):
        tab_identity, tab_classes, tab_metrics = st.tabs(
            ["Model identity", "Class map", "Offline metrics"]
        )

        with tab_identity:
            model_path = active_model_path()
            try:
                model_size = format_bytes(model_path.stat().st_size)
            except OSError:
                model_size = "Unavailable"
            try:
                model_stamp = datetime.fromtimestamp(
                    model_path.stat().st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M UTC")
            except OSError:
                model_stamp = "Unavailable"

            streamlit_version = getattr(st, "__version__", None) or "unknown"
            try:
                import ultralytics

                ultralytics_version = getattr(ultralytics, "__version__", "unknown")
            except Exception:  # pragma: no cover - already imported at module load
                ultralytics_version = "unknown"

            rows: list[tuple[str, str]] = [
                ("Weight file", model_path.name),
                ("Weights path", str(model_path.relative_to(ROOT))),
                ("Sidebar choice", model_choice(active_model_key())[1]),
                ("Metrics row label", metrics_model_label() or "none stored in the CSVs"),
                ("File size", model_size),
                ("Weights modified", model_stamp),
                ("Architecture", "YOLOv8 (from the trained checkpoint)"),
                ("Class count", str(len(class_names))),
                ("Inference image size", f"{IMGSZ} x {IMGSZ} px"),
                ("Device in use", record.device if record else "cpu (default)"),
                ("CUDA available", "yes" if cuda_available() else "no"),
                ("Streamlit", str(streamlit_version)),
                ("Ultralytics", str(ultralytics_version)),
                ("Torch", str(getattr(torch, "__version__", "unknown"))),
                ("Dashboard version", APP_VERSION),
            ]
            st.dataframe(
                pd.DataFrame(rows, columns=["Field", "Value"]),
                hide_index=True,
                width="stretch",
                height=38 * len(rows) + 42,
            )
            html(
                '<div class="pdb-note" style="margin-top:0.6rem;">'
                "This dashboard loads exactly one model at a time, from "
                f"<code>{esc(str(model_path.relative_to(ROOT)))}</code>, chosen with "
                "the weights selector in the sidebar. It never falls back to a "
                "pretrained or alternate checkpoint, and it never re-trains or "
                "fine-tunes anything at runtime.</div>"
            )

        with tab_classes:
            html(
                '<div class="pdb-note" style="margin-bottom:0.6rem;">'
                "Class identities are read from <code>model.names</code> at "
                "runtime. The dashboard label is presentation only - the model "
                "class name is what gets stored, tabulated and exported.</div>"
            )
            class_frame = pd.DataFrame(
                [
                    {
                        "ID": class_id,
                        "Model class name": raw_name,
                        "Dashboard label": display_label(raw_name),
                        "Matches dataset": "yes" if EXPECTED_CLASS_NAMES.get(class_id) == raw_name else "no",
                        "Colour": CLASS_COLORS.get(class_id, FALLBACK_COLOR),
                    }
                    for class_id, raw_name in class_names.items()
                ],
                columns=["ID", "Model class name", "Dashboard label", "Matches dataset", "Colour"],
            )
            st.dataframe(class_frame, hide_index=True, width="stretch")

        with tab_metrics:
            render_metrics_panel()

        st.markdown("**Limitations and intended use**")
        html(
            '<div class="pdb-note pdb-note--warn">'
            "<b>This system is a prototype. Read these before trusting any result.</b>"
            '<ol class="pdb-note--list">'
            + "".join(f"<li>{esc(item)}</li>" for item in LIMITATIONS)
            + "</ol></div>"
        )


# ==================================================
# SECTION 21 - UI: HOW TO USE
# ==================================================

def render_how_to_use() -> None:
    """Compact five-step operating guide."""
    section_title("07", "How to Use", "From upload to exported report.")
    html(
        '<div class="pdb-steps">'
        + "".join(
            f'<div class="pdb-step"><span class="pdb-step-n">{esc(number)}</span>'
            f'<p class="pdb-step-t">{esc(text)}</p></div>'
            for number, text in HOW_TO_USE
        )
        + "</div>"
        f'<div class="pdb-note" style="margin-top:0.7rem;">{esc(THRESHOLD_CAVEAT)}</div>'
    )


# ==================================================
# SECTION 22 - MAIN FLOW
# ==================================================

def main() -> None:
    """Compose the dashboard."""
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=":material/memory:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    render_model_selector()

    model, boot_message, boot_detail = boot_model()
    class_names: dict[int, str] = (
        normalize_names(getattr(model, "names", None) or {}) if model is not None else {}
    )
    render_header(model is not None, class_names or None)

    if model is None:
        st.error(
            boot_message
            or "The trained model could not be loaded, so no inspection can be run."
        )
        if boot_detail:
            with st.expander("Technical detail"):
                st.code(boot_detail, language="text")
        st.stop()

    mapping_warnings = class_mapping_warnings(class_names)
    if mapping_warnings:
        with st.expander(
            f"Model class map differs from the documented dataset "
            f"({len(mapping_warnings)} item(s))"
        ):
            st.warning(
                "The loaded model's class names do not exactly match the training "
                "dataset definition. Everything below still uses the model's own "
                "class IDs and names unchanged - nothing is remapped."
            )
            for message in mapping_warnings:
                st.write(f"- {message}")

    html(
        '<div class="pdb-note pdb-note--warn" style="margin-bottom:1.1rem;">'
        "<b>AI PROTOTYPE</b> &mdash; built for research and demonstration. "
        f"{esc(PROTOTYPE_NOTICE)}</div>"
    )

    settings = render_sidebar()

    section_title(
        "01",
        "Board Image",
        f"Accepted formats: {ACCEPTED_UPLOAD_LABEL}. Drag and drop, or browse. "
        "EXIF rotation is corrected and the file is converted to RGB for you.",
    )
    with st.container(key="pdb-upload-card"):
        uploaded = st.file_uploader(
            "Upload a PCB image",
            type=ACCEPTED_UPLOAD_TYPES,
            key="pdb_upload",
            label_visibility="collapsed",
            help=(
                "A single PCB image. The original file is only ever read, never "
                "modified or written back."
            ),
        )

    image: Image.Image | None = None
    image_digest = ""
    file_size = 0
    filename = ""
    image_error: str | None = None

    if uploaded is None:
        render_empty_state()
    else:
        filename = uploaded.name or "uploaded_board"
        raw_bytes = uploaded.getvalue()
        file_size = len(raw_bytes)
        image_digest = hashlib.sha256(raw_bytes).hexdigest()[:16]

        if file_size == 0:
            image_error = "The uploaded file is empty. Please choose a different image."
        else:
            try:
                image = get_working_image(raw_bytes, image_digest)
            except ImageDecodeError as exc:
                image_error = str(exc)
            else:
                image_error = check_image_size(image)

        if image_error:
            st.error(f"{image_error}")
            st.caption(f"Supported formats are {ACCEPTED_UPLOAD_LABEL}.")
        elif image is not None:
            render_meta_strip(filename, image.size, file_size)

            with st.container(key="pdb-run"):
                run_clicked = st.button(
                    "RUN INSPECTION",
                    type="primary",
                    width="stretch",
                    icon=":material/play_arrow:",
                    help="Runs the model once on the uploaded image using the current detection settings.",
                )
            html(
                '<div class="pdb-note" style="margin:-0.35rem 0 0.9rem 0;">'
                f"Runs once per click at {IMGSZ} px on <b>{esc(settings.device)}</b> "
                f"with confidence &ge; {settings.conf:.2f}, NMS IoU {settings.iou:.2f}, "
                f"max {settings.max_det} detections. Changing a display toggle never "
                "re-runs the model.</div>"
            )

            if run_clicked:
                signature = settings.signature(image_digest)
                try:
                    with st.spinner(
                        f"Running {APP_SUBTITLE} at {IMGSZ} px on {settings.device}..."
                    ):
                        st.session_state["pdb_inspection"] = run_inference(
                            model=model,
                            image=image,
                            settings=settings,
                            signature=signature,
                            image_digest=image_digest,
                            filename=filename,
                            file_size=file_size,
                        )
                except Exception as exc:
                    st.session_state.pop("pdb_inspection", None)
                    st.error(
                        "The inspection could not be completed. The model did not "
                        "return a usable result, so nothing is being reported for "
                        "this image."
                    )
                    with st.expander("Technical detail"):
                        st.code(f"{type(exc).__name__}: {exc}", language="text")

    record: InspectionRecord | None = st.session_state.get("pdb_inspection")

    if record is not None and image is not None:
        is_stale = record.signature != settings.signature(image_digest)

        if is_stale:
            reasons: list[str] = []
            if record.image_digest != image_digest:
                reasons.append("the uploaded image changed")
            reasons.extend(
                settings.changed_inference_fields(
                    InspectionSettings(
                        conf=record.conf,
                        iou=record.iou,
                        max_det=record.max_det,
                        cutoff=settings.cutoff,
                        show_labels=settings.show_labels,
                        show_confidence=settings.show_confidence,
                        device=record.device,
                    )
                )
            )
            html(
                '<div class="pdb-note pdb-note--stale">'
                "<b>Stale result.</b> Everything below belongs to an earlier run "
                f"because {esc(' and '.join(reasons) if reasons else 'the inputs changed')}. "
                "Click RUN INSPECTION again to refresh it.</div>"
            )

        section_title(
            "02",
            "Inspection Result",
            f"{record.filename} · inspected {record.timestamp_utc} UTC",
        )
        render_result_banner(record, settings.cutoff)
        render_kpis(record, settings.cutoff)

        section_title(
            "03",
            "Visual Comparison",
            "Boxes come from the model's own coordinates. Solid = confirmed, "
            "dashed = low confidence.",
        )
        render_comparison(record, settings)

        section_title(
            "04",
            "Defect Analytics",
            "Counts and confidence for every class the model was trained on, "
            "including classes with zero detections.",
        )
        chart_frame = render_class_summary(record)
        tab_counts, tab_conf, tab_summary = st.tabs(
            ["Counts by class", "Confidence by detection", "Summary table"]
        )
        with tab_counts:
            render_count_chart(chart_frame)
        with tab_conf:
            render_confidence_chart(record)
        with tab_summary:
            st.dataframe(
                chart_frame,
                hide_index=True,
                width="stretch",
                column_config={
                    "class_id": st.column_config.NumberColumn("Class ID", format="%d", width="small"),
                    "Mean confidence": st.column_config.NumberColumn("Mean confidence", format="%.3f", width="small"),
                },
            )

        section_title(
            "05",
            "Detection Register",
            f"{len(record.detections)} detection(s) for this image. Sort any column; "
            "every value comes straight from the model.",
        )
        render_detection_table(record, settings.cutoff)

        section_title("06", "Export", "Both artifacts are produced in memory.")
        render_downloads(record, settings)

    elif image_error:
        html(
            '<div class="pdb-note pdb-note--danger" style="margin-top:1rem;">'
            "Replace the file above to continue. No inspection can run until a "
            "valid image is loaded.</div>"
        )

    st.divider()
    render_how_to_use()

    st.divider()
    section_title(
        "08",
        "Model & Performance",
        "Verifiable model identity, the class map read from the weights, and any "
        "offline metrics actually present on disk.",
    )
    render_model_information(class_names, record)

    html(
        '<div class="pdb-foot">'
        f"{esc(APP_NAME)} {esc(APP_VERSION)} &middot; {esc(APP_SUBTITLE)} &middot; "
        f"weights: {esc(active_model_path().name)} &middot; fixed {IMGSZ} px inference &middot; "
        "Research prototype - not certified electrical testing."
        "</div>"
    )


if __name__ == "__main__":
    main()
