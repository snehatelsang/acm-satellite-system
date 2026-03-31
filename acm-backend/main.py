from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .physics import (
    LATENCY_S, MAX_DV_KMS, COOLDOWN_S, INIT_FUEL_KG,
    SIM, ObjState,
    propagate_obj,
    batch_cdm_check,
    conjunction_status,
    plan_avoidance_maneuver,
    execute_burn,
    tsiolkovsky,
    eci_to_lla,
    check_eol,
    update_station_keeping,
    mag,
    vadd,
    vsub,
)

from .database import (
    init_db,
    save_sim_meta,
    load_sim_meta,
    upsert_object,
    upsert_objects_batch,
    load_all_objects,
    save_maneuver_queue,
    load_maneuver_queue,
    delete_queued_burn,
    log_cdm_event,
    log_executed_burn,
    get_cdm_history,
    get_burn_history,
)

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("ACM.api")

app = FastAPI(title="Autonomous Constellation Manager", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
   
    init_db()

    # Restore simulation meta
    meta = load_sim_meta()
    if meta:
        SIM.sim_time                  = datetime.fromisoformat(meta["sim_time"])
        SIM.total_collisions          = int(meta["total_collisions"])
        SIM.total_maneuvers_executed  = int(meta["total_maneuvers"])
        SIM.active_cdm_warnings       = int(meta["active_cdm_warnings"])
        log.info(f"Restored sim_time={SIM.sim_time.isoformat()}")
    else:
        log.info("No prior sim state found -- starting fresh")

    db_rows = load_all_objects()
    for row in db_rows:
        obj = ObjState.from_db_row(row)
        if obj.obj_type == "SATELLITE":
            obj.maneuver_queue = load_maneuver_queue(obj.id)
        SIM.objects[obj.id] = obj

    log.info(
        f"Loaded {sum(1 for o in SIM.objects.values() if o.obj_type=='SATELLITE')} satellites "
        f"and {sum(1 for o in SIM.objects.values() if o.obj_type=='DEBRIS')} debris from DB"
    )


class Vec3(BaseModel):
    x: float
    y: float
    z: float

class TelemetryObject(BaseModel):
    id:   str
    type: str
    r:    Vec3
    v:    Vec3

class TelemetryRequest(BaseModel):
    timestamp: str
    objects:   List[TelemetryObject]

class BurnCommand(BaseModel):
    burn_id:       str
    burnTime:      str
    deltaV_vector: Vec3

class ManeuverRequest(BaseModel):
    satelliteId:       str
    maneuver_sequence: List[BurnCommand]

class StepRequest(BaseModel):
    step_seconds: float



def _queue_auto_burns(sat: ObjState, plan: dict) -> None:
    """Append auto-planned evasion + recovery burns (deduped), persist to DB."""
    now_s   = SIM.sim_time.timestamp()
    ev_abs  = now_s + plan["evasion_burn"]["time_offset_s"]
    rec_abs = now_s + plan["recovery_burn"]["time_offset_s"]

    existing = {b["burn_id"] for b in sat.maneuver_queue}
    ev_id    = f"AUTO_EVASION_{sat.id}"
    rec_id   = f"AUTO_RECOVERY_{sat.id}"

    changed = False
    if ev_id not in existing:
        sat.maneuver_queue.append({
            "burn_id":       ev_id,
            "burn_time_abs": ev_abs,
            "dv":            plan["evasion_burn"]["dv_vector_kms"],
            "fuel_kg":       plan["evasion_burn"]["fuel_kg"],
        })
        changed = True

    if rec_id not in existing:
        sat.maneuver_queue.append({
            "burn_id":       rec_id,
            "burn_time_abs": rec_abs,
            "dv":            plan["recovery_burn"]["dv_vector_kms"],
            "fuel_kg":       plan["recovery_burn"]["fuel_kg"],
        })
        changed = True

    if changed:
        sat.maneuver_queue.sort(key=lambda b: b["burn_time_abs"])
        save_maneuver_queue(sat.id, sat.maneuver_queue)   # persist
        log.info(f"Auto-queued evasion + recovery for {sat.id}")


def _persist_sim_state() -> None:
    """Save sim clock + counters to DB."""
    save_sim_meta(
        SIM.sim_time,
        SIM.total_collisions,
        SIM.total_maneuvers_executed,
        SIM.active_cdm_warnings,
    )


@app.post("/api/telemetry")
async def ingest_telemetry(req: TelemetryRequest):
    """
    Ingest state-vector batch.
    - Upserts each object in SIM.objects and in the DB.
    - Scans satellites for CRITICAL conjunctions; auto-queues avoidance burns.
    - Logs WARNING/CRITICAL CDM events to the DB.
    """
    processed = 0
    new_objs  = []

    for obj_data in req.objects:
        r = [obj_data.r.x, obj_data.r.y, obj_data.r.z]
        v = [obj_data.v.x, obj_data.v.y, obj_data.v.z]

        if obj_data.id in SIM.objects:
            o = SIM.objects[obj_data.id]
            o.r = r
            o.v = v
        else:
            o = ObjState(obj_data.id, obj_data.type, r, v)
            SIM.objects[obj_data.id] = o

        new_objs.append(o)
        processed += 1

    upsert_objects_batch(new_objs)

    cdm_count = 0
    for sat in list(SIM.objects.values()):
        if sat.obj_type != "SATELLITE":
            continue
        warnings = batch_cdm_check(sat)
        cdm_count += len(warnings)

        for w in warnings:

            log_cdm_event(
                satellite_id=sat.id,
                debris_id=w["debris_id"],
                distance_km=w["distance_km"],
                status=w["status"],
                sim_time=SIM.sim_time,
            )
            if w["status"] == "CRITICAL":
                deb = SIM.objects.get(w["debris_id"])
                if deb:
                    plan = plan_avoidance_maneuver(sat, deb, tca_s=300.0)
                    if plan:
                        _queue_auto_burns(sat, plan)

    SIM.active_cdm_warnings = cdm_count
    _persist_sim_state()

    log.info(f"Telemetry ACK: processed={processed}  cdm_warnings={cdm_count}")
    return {
        "status":              "ACK",
        "processed_count":     processed,
        "active_cdm_warnings": cdm_count,
    }


@app.post("/api/maneuver/schedule")
async def schedule_maneuver(req: ManeuverRequest):

    sat = SIM.objects.get(req.satelliteId)
    if sat is None or sat.obj_type != "SATELLITE":
        raise HTTPException(404, f"Satellite '{req.satelliteId}' not found.")

    now_s           = SIM.sim_time.timestamp()
    cumulative_fuel = 0.0
    current_mass    = sat.mass_kg

    for burn in req.maneuver_sequence:
        burn_dt  = datetime.fromisoformat(burn.burnTime.replace("Z", "+00:00"))
        burn_abs = burn_dt.timestamp()

        if burn_abs < now_s + LATENCY_S:
            raise HTTPException(
                400,
                f"Burn '{burn.burn_id}' scheduled too soon "
                f"(must be >= {LATENCY_S} s from now).",
            )

        dv     = [burn.deltaV_vector.x, burn.deltaV_vector.y, burn.deltaV_vector.z]
        dv_mag = mag(dv)
        if dv_mag > MAX_DV_KMS:
            raise HTTPException(
                400,
                f"Burn '{burn.burn_id}': |dv|={dv_mag*1000:.2f} m/s exceeds 15 m/s limit.",
            )

        fuel_needed      = tsiolkovsky(current_mass, dv_mag)
        cumulative_fuel += fuel_needed
        current_mass    -= fuel_needed

        sat.maneuver_queue.append({
            "burn_id":       burn.burn_id,
            "burn_time_abs": burn_abs,
            "dv":            dv,
            "fuel_kg":       round(fuel_needed, 6),
        })

    sat.maneuver_queue.sort(key=lambda b: b["burn_time_abs"])

    save_maneuver_queue(sat.id, sat.maneuver_queue)

    sufficient = sat.fuel_kg >= cumulative_fuel
    proj_mass  = round(sat.mass_kg - cumulative_fuel, 4)

    log.info(
        f"Scheduled {len(req.maneuver_sequence)} burn(s) for {req.satelliteId}  "
        f"fuel_needed={cumulative_fuel:.4f} kg  sufficient={sufficient}"
    )

    return {
        "status": "SCHEDULED",
        "validation": {
            "ground_station_los":          True,
            "sufficient_fuel":             sufficient,
            "projected_mass_remaining_kg": proj_mass,
        },
    }


@app.post("/api/simulate/step")
async def simulate_step(req: StepRequest):
    dt       = req.step_seconds
    start_ts = SIM.sim_time.timestamp()
    end_ts   = start_ts + dt
    collisions = 0
    maneuvers  = 0

    for sat in list(SIM.objects.values()):
        if sat.obj_type != "SATELLITE":
            continue

        burns_this_tick = [b for b in sat.maneuver_queue
                           if b["burn_time_abs"] <= end_ts]

        if burns_this_tick:
            cursor_ts = start_ts
            for burn in sorted(burns_this_tick, key=lambda b: b["burn_time_abs"]):

                # cooldown guard
                if burn["burn_time_abs"] - sat.last_burn_time_s < COOLDOWN_S:
                    log.warning(f"Skipping '{burn['burn_id']}' -- thruster cooldown")
                    sat.maneuver_queue.remove(burn)
                    delete_queued_burn(sat.id, burn["burn_id"])
                    continue

                # fuel guard
                dv_mag      = mag(burn["dv"])
                fuel_needed = tsiolkovsky(sat.mass_kg, dv_mag)
                if fuel_needed > sat.fuel_kg:
                    log.warning(f"Skipping '{burn['burn_id']}' -- insufficient fuel")
                    sat.maneuver_queue.remove(burn)
                    delete_queued_burn(sat.id, burn["burn_id"])
                    continue

                # propagate to burn epoch
                dt_to_burn = burn["burn_time_abs"] - cursor_ts
                if dt_to_burn > 0:
                    propagate_obj(sat, dt_to_burn)

                # apply burn -- execute_burn writes to burn_log table internally
                execute_burn(
                    sat,
                    burn["dv"],
                    burn["burn_time_abs"],
                    burn_id=burn["burn_id"],
                    sim_time=SIM.sim_time,
                )

                sat.maneuver_queue.remove(burn)
                delete_queued_burn(sat.id, burn["burn_id"])
                cursor_ts = burn["burn_time_abs"]
                maneuvers += 1

            # propagate remainder after last burn
            dt_remaining = end_ts - cursor_ts
            if dt_remaining > 0:
                propagate_obj(sat, dt_remaining)
        else:
            propagate_obj(sat, dt)

        check_eol(sat)
        update_station_keeping(sat, dt)

    for obj in SIM.objects.values():
        if obj.obj_type == "DEBRIS":
            propagate_obj(obj, dt)

    sat_list = [o for o in SIM.objects.values() if o.obj_type == "SATELLITE"]
    deb_list = [o for o in SIM.objects.values() if o.obj_type == "DEBRIS"]

    for sat in sat_list:
        for deb in deb_list:
            d, status = conjunction_status(sat.r, deb.r)
            if status == "CRITICAL":
                collisions += 1
                sat.collision_count += 1
                log.error(f"COLLISION: {sat.id} <-> {deb.id}  d={d*1000:.1f} m")
                log_cdm_event(sat.id, deb.id, d, "CRITICAL", SIM.sim_time)

    SIM.total_collisions         += collisions
    SIM.total_maneuvers_executed += maneuvers
    SIM.sim_time = datetime.fromtimestamp(end_ts, tz=timezone.utc)

    upsert_objects_batch(list(SIM.objects.values()))
    _persist_sim_state()

    return {
        "status":              "STEP_COMPLETE",
        "new_timestamp":       SIM.sim_time.isoformat().replace("+00:00", "Z"),
        "collisions_detected": collisions,
        "maneuvers_executed":  maneuvers,
    }


@app.get("/api/visualization/snapshot")
async def visualization_snapshot():
    satellites   = []
    debris_cloud = []

    for obj in SIM.objects.values():
        lat, lon, alt = eci_to_lla(obj.r, SIM.sim_time)
        if obj.obj_type == "SATELLITE":
            satellites.append({
                "id":      obj.id,
                "lat":     lat,
                "lon":     lon,
                "alt_km":  alt,
                "fuel_kg": round(obj.fuel_kg, 3),
                "status":  obj.status,
            })
        else:
            debris_cloud.append([obj.id, lat, lon, alt])

    return {
        "timestamp":    SIM.sim_time.isoformat().replace("+00:00", "Z"),
        "satellites":   satellites,
        "debris_cloud": debris_cloud,
    }



@app.get("/api/status")
async def api_status():
    """Fleet-wide health summary."""
    sats = sum(1 for o in SIM.objects.values() if o.obj_type == "SATELLITE")
    debs = sum(1 for o in SIM.objects.values() if o.obj_type == "DEBRIS")
    return {
        "sim_time":            SIM.sim_time.isoformat(),
        "tracked_satellites":  sats,
        "tracked_debris":      debs,
        "total_collisions":    SIM.total_collisions,
        "total_maneuvers":     SIM.total_maneuvers_executed,
        "active_cdm_warnings": SIM.active_cdm_warnings,
    }



@app.get("/api/history/cdm")
async def cdm_history(
    satellite_id: Optional[str] = Query(None, description="Filter by satellite ID"),
    limit:        int            = Query(100,  ge=1, le=1000),
):
    """
    Return CDM event history from the database.
    Optionally filter by satellite_id.
    """
    rows = get_cdm_history(satellite_id=satellite_id, limit=limit)
    return {"count": len(rows), "events": rows}



@app.get("/api/history/burns")
async def burn_history(
    satellite_id: Optional[str] = Query(None, description="Filter by satellite ID"),
    limit:        int            = Query(100,  ge=1, le=1000),
):
    """
    Return executed-burn audit trail from the database.
    Optionally filter by satellite_id.
    """
    rows = get_burn_history(satellite_id=satellite_id, limit=limit)
    return {"count": len(rows), "burns": rows}




if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)