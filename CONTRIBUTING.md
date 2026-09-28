# Contributing

Thanks for your interest in PCB Defect Inspector. This is a research/demonstration prototype, so contributions that improve **reproducibility, honesty of evaluation, and robustness** are the most valuable.

## Ways to help
- Report bugs (include OS, Python version, `pip freeze`, model file name, and a screenshot or traceback).
- Improve documentation or fix inaccuracies.
- Add tests for model loading, image handling, and CSV export.
- Contribute per-class evaluation results with the exact weights, dataset version and settings used.

## Ground rules
1. **No invented numbers.** Any metric, speed or accuracy claim must state the weights, dataset split, image size and settings, and link to a saved evaluation output.
2. **Do not commit** datasets, weights you do not have rights to redistribute, `.venv`, secrets, or personal paths.
3. **Do not commit third-party screenshots or images** unless their licence permits it.
4. Keep the inference behaviour unchanged unless the change is justified, described, and tested.

## Workflow
1. Fork and create a branch: `git checkout -b fix/short-description`
2. Make focused changes; keep commits small with clear messages.
3. Run the app from the repository root and check the flow end to end:
   `python -m streamlit run dashboard/app.py`
4. Open a pull request describing what changed, why, and how you tested it.

By contributing you agree that your contributions are licensed under the repository licence (see `LICENSE` and `docs/licensing.md`).
