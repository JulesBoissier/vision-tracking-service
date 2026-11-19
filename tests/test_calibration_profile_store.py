"""Behaviour of ``CalibrationProfileStore``, the SQLite persistence for calibration profiles."""

from src.backend.calibration_map import CalibrationMap
from src.backend.calibration_profile_store import CalibrationProfileStore


def sample_map(offset=0) -> CalibrationMap:
    m = CalibrationMap()
    m.add_calibration_point(0 + offset, 0, 300, 200, -0.4, -0.3)
    m.add_calibration_point(1920 + offset, 1080, 310, 205, 0.4, 0.3)
    return m


def test_round_trips_calibration_map(store):
    """A saved map loads back with every value intact.

    Why: profiles are what make calibration reusable. Losing or reordering a single value
    changes every prediction made with that profile.
    """
    store.save_profile("desk", sample_map())
    profile_id = store.list_profiles()[0]["id"]

    assert store.load_profile(profile_id) == sample_map()


def test_saving_existing_name_overwrites_it(store):
    """Saving under an existing name replaces that profile instead of adding a second one.

    Why: names are unique so a user can recalibrate "desk" without accumulating stale
    copies. The id stays the same so anything referring to it keeps working.
    """
    store.save_profile("desk", sample_map())
    first_id = store.list_profiles()[0]["id"]

    store.save_profile("desk", sample_map(offset=5))

    profiles = store.list_profiles()
    assert len(profiles) == 1
    assert profiles[0]["id"] == first_id
    assert store.load_profile(first_id) == sample_map(offset=5)


def test_profiles_survive_reopening_database(tmp_path):
    """Profiles written by one store instance are readable by a new instance on the same file.

    Why: the service is restarted (container rebuilds, reboots) and must find profiles saved
    in previous runs on its volume.
    """
    url = f"sqlite:///{tmp_path / 'cal.db'}"
    CalibrationProfileStore(url).save_profile("desk", sample_map())

    reopened = CalibrationProfileStore(url)

    assert [p["profile_name"] for p in reopened.list_profiles()] == ["desk"]


def test_deleting_unknown_profile_is_harmless(store):
    """Deleting an id that doesn't exist changes nothing and doesn't raise.

    Why: delete is idempotent so a retried or duplicated request from the client is safe.
    """
    store.save_profile("desk", sample_map())

    store.delete_profile(9999)

    assert len(store.list_profiles()) == 1
