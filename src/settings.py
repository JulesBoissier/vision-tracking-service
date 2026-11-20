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

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        env = os.environ if env is None else env
        return cls(
            calibration_agent=env.get("CALIBRATION_AGENT", cls.calibration_agent).lower(),
            model_path=env.get("MODEL_PATH", cls.model_path),
            calibration_db_url=env.get("CALIBRATION_DB_URL", cls.calibration_db_url),
        )
