# vision-tracking-service

ML gaze tracking service: given a webcam image, predicts the on-screen Point of Regard (PoR) using L2CS-Net (ResNet50, Gaze360 weights) plus a per-user calibration profile. FastAPI app, shipped as a Docker container. Part of the attention-tracking system — see `../CLAUDE.md`. Its client is `activity-orchestration-client/src/backend/clients/vision_tracking_client.py`.

## Layout

- `main.py` — imports `app` from `src.app.routes`; runs uvicorn on 127.0.0.1:8000 when run directly. The Dockerfile runs `uvicorn main:app` on 0.0.0.0:8000.
- `src/app/routes.py` — FastAPI routes; a `lifespan` builds a single global `VisionTrackingEngine` in the `resources` dict.
- `src/app/models.py` — Pydantic response models.
- `src/backend/vision_tracking_engine.py` — facade composing the three components below.
- `src/backend/gaze_predictor.py` — wraps `l2cs.Pipeline` (CPU). Returns `(head_x, head_y, pitch, yaw)`; head position = face bbox centre with **x mirrored** (`image_width - cx`).
- `src/backend/calibration_agents.py` — `CalibrationAgent` ABC, `LinearCalibrationAgent` (default) and `InterpolationAgent` (previous IDW model, `CALIBRATION_AGENT=idw`).
- `src/backend/calibration_map.py` — `CalibrationMap` dataclass of parallel lists (monitor x/y, head x/y, theta, phi).
- `src/backend/calibration_profile_store.py` — SQLAlchemy/SQLite store (`calibration.db`), map stored as a `PickleType` blob.
- `tools.py` — old manual experiment helpers; **stale** (uses outdated signatures), not used by the app.
- `models/` — put `L2CSNet_gaze360.pkl` here (gitignored; download link in README).

## Commands

```bash
# Docker (normal way to run). Model file must be in models/ before building.
docker compose up --build
# Local
pip install -r requirements.txt   # pulls torch + CUDA wheels and l2cs from git; heavy
python main.py
# Tests (need torch + l2cs installed because gaze_predictor imports l2cs)
pytest tests/
pytest tests/test_linear_calibration_agent.py::test_one_bad_calibration_dot_has_little_influence
```

CI (`.github/workflows/python-ci.yml`) runs `pytest tests/` on Python 3.12 for pushes/PRs to `main` and `develop`. The Docker image uses Python 3.10.

## Tests

Tests cover behaviour at three levels: the calibration agents (calibration maths; `tests/test_linear_calibration_agent.py`, `tests/test_interpolation_agent.py`), `VisionTrackingEngine` (calibrate/predict/profiles, run against **both** agents via the parametrized `engine` fixture), and the HTTP API (the orchestrator contract). None load L2CS: `tests/conftest.py::FakeGazePredictor` returns scripted features for solid-colour frames (the pixel value identifies the frame and survives JPEG). The `api` fixture is the only place that patches app internals. Each docstring has a "Why" explaining the design intent; read it before changing a test. `xfail(strict=True)` marks known bugs; remove the marker when fixing. Model accuracy is measured by `../attention-benchmark`, not unit tests.

## API

`GET /health`, `POST /cal_point` (form `x`, `y`, `file`), `POST /predict` (form `file`) → `{"prediction": [x, y]}`, `POST /save_profile?name=`, `GET /list_profiles`, `POST /load_profile?profile_id=`, `POST /delete_profile?profile_id=`, `POST /reset_profile`. These shapes are a contract with the orchestrator — change both repos together.

## How prediction works

`GazePredictor` returns head position (face-box centre, x mirrored) and two gaze angles: `theta` = L2CS `pitch`, `phi` = L2CS `yaw`. On real data pitch tracks screen x (r≈0.95) and yaw tracks screen y (r≈−0.9). L2CS's own `draw_gaze` uses pitch for horizontal displacement, so this naming is intentional.

**`LinearCalibrationAgent` (default):** screen x and y are each a linear function of both angles, fitted by Huber-weighted least squares on standardised angles (light ridge), and predictions are clamped to the bounding box of the calibration targets. Head position is deliberately unused. The fit is cached and recomputed when the map changes (the `calibration_map` setter, `calibration_step`, `initialize_cal_map`).

**`InterpolationAgent` (`CALIBRATION_AGENT=idw`):** inverse-distance weighting per axis over `(angle, head position)`. It mixes pixels and radians, so sub-pixel face-box jitter dominated it. Kept for A/B benchmark comparisons and for its tests.

Why the switch (attention-benchmark, see `../docs/IMPROVEMENTS.md` 3.1): median error went from 359→292 px (single monitor) and 665→295 px (three monitors), and predictions on the right monitor from 41%→80%. Offline experiments showed:
- head-position features break when posture changes between calibration and use;
- averaging calibration frames and polynomial terms didn't help;
- about 290 px (leave-one-out) is the single-frame precision ceiling of this camera (640×480, face ~120 px wide) and model.

Further gains need better inputs, not a better calibration fit: multi-frame bursts at prediction time, a higher camera resolution, or a stronger gaze model.

## Gotchas

- **Changing the calibration model invalidates benchmark baselines.** Run both benchmarks with `--compare` (`../attention-benchmark`) and keep the agent's tests' "Why" sections in mind.
- **Pickled profiles:** `CalibrationMap` is pickled into SQLite. Moving/renaming the class or its module breaks loading existing profiles. Add fields with defaults only.
- **docker-compose mounts the named volume over the whole `/app`**, not just the DB. After the first run, the volume's copy of the code shadows the image, so `docker compose up --build` won't pick up code changes until the volume is removed (`docker compose down -v`, which also deletes saved profiles).
- **Single global stateful engine:** one active calibration map shared by all clients. Built for a single user; not safe for concurrent calibration.
- **No face / not calibrated:** the engine returns `[null, null]` when the predictor finds no face, or when the agent raises `NotCalibratedError` (`ZeroDivisionError` for the IDW agent). `/cal_point` skips no-face frames and answers `"added": false` (status 200). `load_profile` with an unknown id still sets the map to `{}`, which breaks later predictions (IMPROVEMENTS 1.7).
- Inference is pinned to CPU (`torch.device("cpu")`) even though CUDA wheels are in `requirements.txt` on `main` (`develop` trims requirements to direct deps).
- `/reset_profile` ignores the `profile_id` the client sends.
