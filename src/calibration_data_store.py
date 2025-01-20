from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    PickleType,
    String,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class CalibrationProfile(Base):
    __tablename__ = "calibration_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    profile_name = Column(String, unique=True, nullable=False, index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), index=True)
    calibration_map = Column(PickleType)  # Storing dict as binary


class CalibrationDataStore:
    """Handles database interactions for calibration profiles."""

    def __init__(self, db_url="sqlite:///calibration.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_profile(self, profile_name: str, calibration_map: dict):
        """Save a calibration profile by name, updating it if it already exists."""
        session = self.Session()
        existing_profile = (
            session.query(CalibrationProfile)
            .filter_by(profile_name=profile_name)
            .first()
        )

        if existing_profile:
            existing_profile.calibration_map = calibration_map
        else:
            new_profile = CalibrationProfile(
                profile_name=profile_name, calibration_map=calibration_map
            )
            session.add(new_profile)

        session.commit()
        session.close()

    def load_profile(self, profile_id: int) -> dict:
        """Load a calibration profile by ID."""
        session = self.Session()
        profile = session.query(CalibrationProfile).filter_by(id=profile_id).first()
        session.close()
        return profile.calibration_map if profile else {}

    def list_profiles(self) -> list:
        """Return a list of available profiles with their IDs and names."""
        session = self.Session()
        profiles = session.query(
            CalibrationProfile.id,
            CalibrationProfile.profile_name,
            CalibrationProfile.updated_at,
        ).all()
        session.close()
        return [
            {"id": p.id, "profile_name": p.profile_name, "updated_at": p.updated_at}
            for p in profiles
        ]

    def delete_profile(self, profile_id: int):
        """Delete a calibration profile by ID."""
        session = self.Session()
        profile = session.query(CalibrationProfile).filter_by(id=profile_id).first()
        if profile:
            session.delete(profile)
            session.commit()
        session.close()
