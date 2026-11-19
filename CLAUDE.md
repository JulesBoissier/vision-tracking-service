# vision-tracking-service

ML gaze tracking service: given a webcam image, predicts the on-screen Point of Regard (PoR) using L2CS-Net (ResNet50, Gaze360 weights) plus a per-user calibration profile. FastAPI app, shipped as a Docker container. Part of the attention-tracking system — see `../CLAUDE.md`. Its client is `activity-orchestration-client/src/backend/clients/vision_tracking_client.py`.

## Layout

- `main.py` — imports `app` from `src.app.routes`; runs uvicorn on 127.0.0.1:8000 when run directly. The Dockerfile runs `uvicorn main:app` on 0.0.0.0:8000.
- `src/app/routes.py` — FastAPI routes; a `lifespan` builds a single global `VisionTrackingEngine` in the `resources` dict.
- `src/app/models.py` — Pydantic response models.
- `src/backend/vision_tracking_engine.py` — facade composing the three components below.
- `src/backend/gaze_predictor.py` — wraps `l2cs.Pipeline` (CPU). Returns `(head_x, head_y, pitch, yaw)`; head position = face bbox centre with **x mirrored** (`image_width - cx`).
- `src/backend/calibration_agents.py` — `CalibrationAgent` ABC + `InterpolationAgent`.
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
pytest tests/test_calibration_agents.py::TestInterpolationAgent::test_calibration_step
```

CI (`.github/workflows/python-ci.yml`) runs `pytest tests/` on Python 3.12 for pushes/PRs to `main` and `develop`. The Docker image uses Python 3.10.

## Tests

Tests cover behaviour at three levels: `InterpolationAgent` (calibration maths), `VisionTrackingEngine` (calibrate/predict/profiles), and the HTTP API (the orchestrator contract). None load L2CS: `tests/conftest.py::FakeGazePredictor` returns scripted features for solid-colour frames (the pixel value identifies the frame and survives JPEG). The `api` fixture is the only place that patches app internals. Each docstring has a "Why" explaining the design intent; read it before changing a test. `xfail(strict=True)` marks known bugs; remove the marker when fixing. Model accuracy is measured by `../attention-benchmark`, not unit tests.

## API

`GET /health`, `POST /cal_point` (form `x`, `y`, `file`), `POST /predict` (form `file`) → `{"prediction": [x, y]}`, `POST /save_profile?name=`, `GET /list_profiles`, `POST /load_profile?profile_id=`, `POST /delete_profile?profile_id=`, `POST /reset_profile`. These shapes are a contract with the orchestrator — change both repos together.

## How prediction works

`InterpolationAgent._interpolate` is inverse-distance weighting (power 1) over all calibration points, done **independently per axis**: screen-x from `(theta, head_x)`, screen-y from `(phi, head_y)`. Distance = `sqrt(Δangle² + position_interpolation_weight · Δhead²)`. Note `theta` = L2CS `pitch` and `phi` = L2CS `yaw`; L2CS's own `draw_gaze` uses `pitch` for horizontal displacement, so this pairing is intentional. Don't "fix" it without testing on real data.

## Gotchas

- **Pickled profiles:** `CalibrationMap` is pickled into SQLite. Moving/renaming the class or its module breaks loading existing profiles. Add fields with defaults only.
- **docker-compose mounts the named volume over the whole `/app`**, not just the DB. After the first run, the volume's copy of the code shadows the image, so `docker compose up --build` won't pick up code changes until the volume is removed (`docker compose down -v`, which also deletes saved profiles).
- **Single global stateful engine:** one active calibration map shared by all clients. Built for a single user; not safe for concurrent calibration.
- **No face / not calibrated:** the engine returns `[null, null]` when the predictor finds no face, or when the agent has no calibration points (`ZeroDivisionError` inside `InterpolationAgent`). `/cal_point` skips no-face frames and answers `"added": false` (status 200). `load_profile` with an unknown id sets the map to `{}`, which breaks later predictions (IMPROVEMENTS 1.7).
- Head position is in pixels and gaze angles are in radians, so `position_interpolation_weight` (default 1) lets head movement dominate the distance. Tune it with that in mind.
- Inference is pinned to CPU (`torch.device("cpu")`) even though CUDA wheels are in `requirements.txt` on `main` (`develop` trims requirements to direct deps).
- `/reset_profile` ignores the `profile_id` the client sends.
