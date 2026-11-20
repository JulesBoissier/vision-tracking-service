"""Vision service settings: the only place that reads the environment."""

import os
from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class Settings:
    # "linear" (robust linear fit, default) or "idw" (previous inverse-distance interpolation).
    calibration_agent: str = "linear"
    model_path: str = os.path.join("models", "L2CSNet_gaze360.pkl")
    # Relative paths are resolved from the working directory. The Docker image sets
    # sqlite:////app/data/calibration.db, inside the volume.
    calibration_db_url: str = "sqlite:///calibration.db"
    # Linear agent: use face width (distance to the camera) as a calibration feature.
    calibration_distance_term: bool = True
    # Implicit calibration points (clicks) kept per session, and their weight in the fit.
    implicit_calibration_max: int = 50
    implicit_calibration_weight: float = 1.0

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        env = os.environ if env is None else env
        return cls(
            calibration_agent=env.get("CALIBRATION_AGENT", cls.calibration_agent).lower(),
            model_path=env.get("MODEL_PATH", cls.model_path),
            calibration_db_url=env.get("CALIBRATION_DB_URL", cls.calibration_db_url),
            calibration_distance_term=env.get("CALIBRATION_DISTANCE_TERM", "1").strip().lower()
            not in ("0", "false", "no", "off"),
            implicit_calibration_max=int(env.get("IMPLICIT_CALIBRATION_MAX", cls.implicit_calibration_max)),
            implicit_calibration_weight=float(
                env.get("IMPLICIT_CALIBRATION_WEIGHT", cls.implicit_calibration_weight)
            ),
        )
