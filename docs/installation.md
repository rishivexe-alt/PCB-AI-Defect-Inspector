# Installation

Report states the intended setup: Windows, Python 3.12, project virtual environment. Nothing below was executed in this environment because the app and weights were not supplied - treat it as **unverified** until you run it in a clean folder.

## Windows PowerShell
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run dashboard\app.py
```
Open the URL Streamlit prints (commonly http://localhost:8501). If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or call `.venv\Scripts\python.exe` directly.

## Linux / macOS
```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip && pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

## Clean-environment check (do before publishing)
```powershell
git clone <your-repo-url> $env:TEMP\pcb-clean ; cd $env:TEMP\pcb-clean
py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# copy weights into models\ , then:
python -m streamlit run dashboard\app.py
```
