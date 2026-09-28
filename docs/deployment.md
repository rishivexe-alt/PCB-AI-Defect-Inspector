# Deployment and GitHub readiness

**Status: no deployment has been performed or tested.**

## GitHub file limits
- Files > 50 MiB trigger a warning; files > 100 MiB are blocked. Use Git LFS or external hosting for large weights. Check your weights' size first (`models/README.md`).
- Never commit `.venv`, datasets you cannot redistribute, secrets, or `.env` files (`.gitignore` covers these).

## Streamlit Community Cloud (feasible, untested)
Needs: a GitHub repo containing `dashboard/app.py`, a working `requirements.txt`, and the weights available at runtime (committed, LFS, or downloaded on start-up from a URL you control). Points to test:
- Python version selection and install time/size of PyTorch + Ultralytics.
- Whether OpenCV needs the headless build (`opencv-python-headless`) on the host.
- Memory limits and cold-start model loading; check Streamlit's current published resource limits.
- Model file availability and licence.

Steps: push to a public GitHub repo -> share.streamlit.io -> New app -> select repo/branch -> main file `dashboard/app.py` -> Deploy. Then record the URL here only after you have tested it.

## Vercel
Vercel does not natively run a conventional Streamlit server. Alternatives: Streamlit Community Cloud, Hugging Face Spaces, a container on a VM/Cloud Run, or split architecture (separate frontend + inference API).
