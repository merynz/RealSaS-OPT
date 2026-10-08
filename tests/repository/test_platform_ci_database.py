import pytest

from tools import platform_ci_database as db


def test_database_url_and_name_contract():
    name = "realsas_ci_1791468000_37790000000_1_main"
    assert db.validate_name(name) == name
    assert db.database_url(name, socket_dir="/tmp/socket", port=55432) == (
        f"postgresql:///{name}?host=/tmp/socket&port=55432"
    )
    with pytest.raises(ValueError):
        db.validate_name("postgres")


def test_gc_drops_only_old_idle_ci_databases(monkeypatch):
    old_idle = "realsas_ci_1791000000_1_1_main"
    old_active = "realsas_ci_1791000000_2_1_migrations"
    fresh = "realsas_ci_1791467900_3_1_main"
    invalid = "realsas_ci_manual"
    dropped = []

    def fake_run(_url, sql, capture=True):
        if "SELECT datname FROM pg_database" in sql:
            return "\n".join([old_idle, old_active, fresh, invalid])
        if old_idle in sql and "pg_stat_activity" in sql:
            return "0"
        if old_active in sql and "pg_stat_activity" in sql:
            return "1"
        raise AssertionError(sql)

    def fake_drop(_url, name, force=False):
        dropped.append((name, force))

    monkeypatch.setattr(db, "run_psql", fake_run)
    monkeypatch.setattr(db, "drop_database", fake_drop)
    report = db.gc_databases("admin", max_age_seconds=3600, now=1791468000)
    assert report["deleted"] == [old_idle]
    assert report["retained_active"] == [old_active]
    assert report["ignored_invalid"] == [invalid]
    assert dropped == [(old_idle, False)]


def test_create_refuses_existing_database(monkeypatch):
    name = "realsas_ci_1791468000_4_1_main"
    monkeypatch.setattr(db, "run_psql", lambda *args, **kwargs: "1")
    with pytest.raises(RuntimeError, match="ALREADY_EXISTS"):
        db.create_database("admin", name)
