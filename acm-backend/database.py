from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, List, Optional

log = logging.getLogger("ACM.db")

import os
DB_PATH = Path(os.getenv("ACM_DB_PATH", "acm_sim.db"))


@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    """
    Yield a SQLite connection with WAL mode, foreign-keys enabled,
    and row_factory set to sqlite3.Row for dict-like access.
    Auto-commits on success, rolls back on exception.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()



DDL = """
-- ── objects ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS objects (
    id                TEXT    PRIMARY KEY,
    obj_type          TEXT    NOT NULL CHECK(obj_type IN ('SATELLITE','DEBRIS')),
    rx                REAL    NOT NULL,
    ry                REAL    NOT NULL,
    rz                REAL    NOT NULL,
    vx                REAL    NOT NULL,
    vy                REAL    NOT NULL,
    vz                REAL    NOT NULL,
    mass_kg           REAL    NOT NULL DEFAULT 550.0,
    fuel_kg           REAL    NOT NULL DEFAULT 50.0,
    status            TEXT    NOT NULL DEFAULT 'NOMINAL',
    nominal_rx        REAL    NOT NULL,
    nominal_ry        REAL    NOT NULL,
    nominal_rz        REAL    NOT NULL,
    last_burn_time_s  REAL    NOT NULL DEFAULT -1000000000.0,
    collision_count   INTEGER NOT NULL DEFAULT 0,
    uptime_s          REAL    NOT NULL DEFAULT 0.0,
    outage_s          REAL    NOT NULL DEFAULT 0.0,
    updated_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ── maneuver_queue ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS maneuver_queue (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    satellite_id    TEXT    NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    burn_id         TEXT    NOT NULL,
    burn_time_abs   REAL    NOT NULL,
    dv_x            REAL    NOT NULL,
    dv_y            REAL    NOT NULL,
    dv_z            REAL    NOT NULL,
    fuel_kg         REAL    NOT NULL,
    queued_at       TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(satellite_id, burn_id)
);

-- ── cdm_events ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cdm_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    satellite_id    TEXT    NOT NULL,
    debris_id       TEXT    NOT NULL,
    distance_km     REAL    NOT NULL,
    status          TEXT    NOT NULL,
    sim_time        TEXT    NOT NULL,
    recorded_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ── burn_log ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS burn_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    satellite_id    TEXT    NOT NULL,
    burn_id         TEXT    NOT NULL,
    burn_time_abs   REAL    NOT NULL,
    dv_x            REAL    NOT NULL,
    dv_y            REAL    NOT NULL,
    dv_z            REAL    NOT NULL,
    dv_mag_ms       REAL    NOT NULL,
    fuel_consumed_kg REAL   NOT NULL,
    mass_after_kg   REAL    NOT NULL,
    fuel_after_kg   REAL    NOT NULL,
    sim_time        TEXT    NOT NULL,
    executed_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ── sim_meta ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sim_meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_mq_sat   ON maneuver_queue(satellite_id, burn_time_abs);",
    "CREATE INDEX IF NOT EXISTS idx_cdm_sat  ON cdm_events(satellite_id, sim_time);",
    "CREATE INDEX IF NOT EXISTS idx_burn_sat ON burn_log(satellite_id, burn_time_abs);",
]


def init_db() -> None:
    """
    Create all tables and indexes if they do not exist.
    Safe to call on every startup.
    """
    with get_conn() as conn:
        conn.executescript(DDL)
        for idx in _INDEXES:
            conn.execute(idx)
    log.info(f"Database initialised at {DB_PATH.resolve()}")



def save_sim_meta(sim_time: datetime, total_collisions: int,
                  total_maneuvers: int, active_cdm: int) -> None:
    meta = {
        "sim_time":            sim_time.isoformat(),
        "total_collisions":    str(total_collisions),
        "total_maneuvers":     str(total_maneuvers),
        "active_cdm_warnings": str(active_cdm),
    }
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO sim_meta(key, value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            meta.items(),
        )


def load_sim_meta() -> Optional[dict]:
    """Return sim meta dict, or None if not yet persisted."""
    with get_conn() as conn:
        rows = conn.execute("SELECT key, value FROM sim_meta").fetchall()
    if not rows:
        return None
    return {r["key"]: r["value"] for r in rows}


def upsert_object(obj) -> None:
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO objects (
                id, obj_type,
                rx, ry, rz, vx, vy, vz,
                mass_kg, fuel_kg, status,
                nominal_rx, nominal_ry, nominal_rz,
                last_burn_time_s, collision_count,
                uptime_s, outage_s, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
            ON CONFLICT(id) DO UPDATE SET
                rx=excluded.rx, ry=excluded.ry, rz=excluded.rz,
                vx=excluded.vx, vy=excluded.vy, vz=excluded.vz,
                mass_kg=excluded.mass_kg, fuel_kg=excluded.fuel_kg,
                status=excluded.status,
                nominal_rx=excluded.nominal_rx,
                nominal_ry=excluded.nominal_ry,
                nominal_rz=excluded.nominal_rz,
                last_burn_time_s=excluded.last_burn_time_s,
                collision_count=excluded.collision_count,
                uptime_s=excluded.uptime_s,
                outage_s=excluded.outage_s,
                updated_at=datetime('now')
        """, (
            obj.id, obj.obj_type,
            obj.r[0], obj.r[1], obj.r[2],
            obj.v[0], obj.v[1], obj.v[2],
            obj.mass_kg, obj.fuel_kg, obj.status,
            obj.nominal_r[0], obj.nominal_r[1], obj.nominal_r[2],
            obj.last_burn_time_s, obj.collision_count,
            obj.uptime_s, obj.outage_s,
        ))


def upsert_objects_batch(obj_list: list) -> None:
    """Bulk upsert — much faster than calling upsert_object in a loop."""
    rows = [(
        o.id, o.obj_type,
        o.r[0], o.r[1], o.r[2],
        o.v[0], o.v[1], o.v[2],
        o.mass_kg, o.fuel_kg, o.status,
        o.nominal_r[0], o.nominal_r[1], o.nominal_r[2],
        o.last_burn_time_s, o.collision_count,
        o.uptime_s, o.outage_s,
    ) for o in obj_list]

    with get_conn() as conn:
        conn.executemany("""
            INSERT INTO objects (
                id, obj_type,
                rx, ry, rz, vx, vy, vz,
                mass_kg, fuel_kg, status,
                nominal_rx, nominal_ry, nominal_rz,
                last_burn_time_s, collision_count,
                uptime_s, outage_s, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
            ON CONFLICT(id) DO UPDATE SET
                rx=excluded.rx, ry=excluded.ry, rz=excluded.rz,
                vx=excluded.vx, vy=excluded.vy, vz=excluded.vz,
                mass_kg=excluded.mass_kg, fuel_kg=excluded.fuel_kg,
                status=excluded.status,
                nominal_rx=excluded.nominal_rx,
                nominal_ry=excluded.nominal_ry,
                nominal_rz=excluded.nominal_rz,
                last_burn_time_s=excluded.last_burn_time_s,
                collision_count=excluded.collision_count,
                uptime_s=excluded.uptime_s,
                outage_s=excluded.outage_s,
                updated_at=datetime('now')
        """, rows)


def load_all_objects() -> List[dict]:
    """Return all rows from objects table as list of dicts."""
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM objects").fetchall()
    return [dict(r) for r in rows]


def delete_object(obj_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM objects WHERE id=?", (obj_id,))


def save_maneuver_queue(satellite_id: str, queue: List[dict]) -> None:
    """
    Overwrite the queued burns for one satellite.
    Deletes existing rows then inserts current queue.
    """
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM maneuver_queue WHERE satellite_id=?", (satellite_id,)
        )
        if queue:
            conn.executemany("""
                INSERT OR IGNORE INTO maneuver_queue
                    (satellite_id, burn_id, burn_time_abs, dv_x, dv_y, dv_z, fuel_kg)
                VALUES (?,?,?,?,?,?,?)
            """, [(
                satellite_id,
                b["burn_id"],
                b["burn_time_abs"],
                b["dv"][0], b["dv"][1], b["dv"][2],
                b["fuel_kg"],
            ) for b in queue])


def load_maneuver_queue(satellite_id: str) -> List[dict]:
    """Load pending burns for a satellite, ordered by burn_time_abs."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT burn_id, burn_time_abs, dv_x, dv_y, dv_z, fuel_kg
            FROM maneuver_queue
            WHERE satellite_id=?
            ORDER BY burn_time_abs ASC
        """, (satellite_id,)).fetchall()
    return [{
        "burn_id":       r["burn_id"],
        "burn_time_abs": r["burn_time_abs"],
        "dv":            [r["dv_x"], r["dv_y"], r["dv_z"]],
        "fuel_kg":       r["fuel_kg"],
    } for r in rows]


def delete_queued_burn(satellite_id: str, burn_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM maneuver_queue WHERE satellite_id=? AND burn_id=?",
            (satellite_id, burn_id),
        )



def log_cdm_event(satellite_id: str, debris_id: str,
                  distance_km: float, status: str,
                  sim_time: datetime) -> None:
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO cdm_events
                (satellite_id, debris_id, distance_km, status, sim_time)
            VALUES (?,?,?,?,?)
        """, (satellite_id, debris_id, round(distance_km, 6),
              status, sim_time.isoformat()))


def get_cdm_history(satellite_id: Optional[str] = None,
                    limit: int = 100) -> List[dict]:
    """Retrieve CDM event log, optionally filtered by satellite."""
    with get_conn() as conn:
        if satellite_id:
            rows = conn.execute("""
                SELECT * FROM cdm_events
                WHERE satellite_id=?
                ORDER BY recorded_at DESC LIMIT ?
            """, (satellite_id, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM cdm_events
                ORDER BY recorded_at DESC LIMIT ?
            """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def log_executed_burn(satellite_id: str, burn_id: str,
                      burn_time_abs: float, dv: List[float],
                      dv_mag_ms: float, fuel_consumed_kg: float,
                      mass_after_kg: float, fuel_after_kg: float,
                      sim_time: datetime) -> None:
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO burn_log (
                satellite_id, burn_id, burn_time_abs,
                dv_x, dv_y, dv_z, dv_mag_ms,
                fuel_consumed_kg, mass_after_kg, fuel_after_kg,
                sim_time
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            satellite_id, burn_id, burn_time_abs,
            dv[0], dv[1], dv[2], dv_mag_ms,
            fuel_consumed_kg, mass_after_kg, fuel_after_kg,
            sim_time.isoformat(),
        ))


def get_burn_history(satellite_id: Optional[str] = None,
                     limit: int = 100) -> List[dict]:
    """Retrieve burn audit log, optionally filtered by satellite."""
    with get_conn() as conn:
        if satellite_id:
            rows = conn.execute("""
                SELECT * FROM burn_log
                WHERE satellite_id=?
                ORDER BY executed_at DESC LIMIT ?
            """, (satellite_id, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM burn_log
                ORDER BY executed_at DESC LIMIT ?
            """, (limit,)).fetchall()
    return [dict(r) for r in rows]