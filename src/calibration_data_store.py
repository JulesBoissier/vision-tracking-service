from sqlalchemy import Column, Integer, PickleType, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class CalibrationProfile(Base):
    __tablename__ = "calibration_profiles"

    # id = Column(Integer, primary_key=True, index=True)
    profile_name = Column(String, primary_key=True, index=True)
    # last_update = Column(String, index=True)
    calibration_map = Column(PickleType)  # Storing dict as binary


class CalibrationDataStore:
    """Handles database interactions for calibration profiles."""

    def __init__(self, db_url="sqlite:///calibration.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_profile(self, profile_name: str, calibration_map: dict):
        """Save calibration map under the given profile name."""
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

    def load_profile(self, profile_name: str) -> dict:
        """Load the calibration map for a given profile name."""
        session = self.Session()
        profile = (
            session.query(CalibrationProfile)
            .filter_by(profile_name=profile_name)
            .first()
        )
        session.close()
        return profile.calibration_map if profile else {}

    def list_profiles(self) -> list:
        """Return a list of available calibration profiles."""
        session = self.Session()
        profiles = [p.profile_name for p in session.query(CalibrationProfile).all()]
        session.close()
        return profiles


# if __name__ == '__main__':
#     cm = CalibrationMap()
#     cm.add_calibration_point(100, 200, 30, 45)

#     cds = CalibrationDataStore()

#     cds.save_profile(profile_name = "hello", calibration_map=cm)

#     cm.add_calibration_point(100, 200, 30, 45)

#     #cds = CalibrationDataStore()

#     cds.save_profile(profile_name = "hello2", calibration_map=cm)

#     print(cds.list_profiles())


#     print(cds.load_profile("hello2"))
