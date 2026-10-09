# FLOD_MAS

Vision inspection of glue lines on fabric panels. Tkinter desktop app for a Windows
inspection PC (Python 3.12); this Linux machine is used for development and experiments.

## Layout

- `main_1366.py` — entry point. `inspection/dashboard.py` builds the segmenter with
  `create_segmenter(CONFIG['segmentation'])`.
- `auto_trigger/` — hand-cycle trigger and fabric classifier gate.
- `segmentation/` — providers. `provider.py` is the Roboflow cloud workflow (default);
  `glue_line.py` is the local detector for pale fabric (opt-in).
- `inspection/` — pipeline, per-fabric measurement, storage, result tabs.
- `strip_analysis.py`, `sam_detection.py` — centreline, width, grading.
- `CalibrateAPP/`, `plane_scale.py` — camera calibration and millimetre scale.
- `tools/replay_capture.py` — offline replay of image files through the pipeline.
- Docs: `AUTOMATIC_INSPECTION.md`, `CONFIGURATION.md`, `CALIBRATION_PIPELINE.md`.
  `GLUE_LINE_FINDINGS.md` records what was measured for glue-line detection and how.

## Commands

- Tests: `python -m unittest discover -s tests -p "test_*.py"`. The system Python here is 3.14
  and lacks some dependencies; use an environment with `scikit-image` and `scipy`.
- GPU experiments: `~/flod_experiments/gpu/bin/python`.
- Experiment scripts, trained models and result pages that are not part of the app live in
  `~/flod_experiments/` (for example `segformer/`, `out/`); `GLUE_LINE_FINDINGS.md` says which.

## Rules for changes

- Work on a separate branch, never on `main`. Do not commit until the change is validated and
  the user asks.
- New behaviour is opt-in: add a config key whose default reproduces the old behaviour, and
  read it with `.get` so older `config.json` files still load.
- Do not put `null` in `config.json`. The startup settings form saves it back as the string
  "None".
- `.env` holds `ROBOFLOW_API_KEY` and is git-ignored. Running a cloud workflow uploads the
  image to Roboflow.

## Glue-line vision

Before touching segmentation, measurement, capture or model choice for the glue line, load the
`glue-line-vision` skill. It records what the glue actually looks like, what has been tried and
measured, and what remains unverified.

## MCP servers

`.mcp.json` adds two project-scoped servers, both vendor-hosted and both needing a sign-in via
`/mcp` before their tools appear:
- `roboflow` — manage Roboflow projects, datasets, labelling, training and Workflows.
- `huggingface` — search models, datasets and docs on the Hub.
