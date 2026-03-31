from __future__ import annotations

import math
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("ACM.physics")

MU             = 398_600.4418   # Earth gravitational parameter   [km^3 s^-2]
RE             = 6_378.137      # Earth equatorial radius         [km]
J2             = 1.08263e-3     # J2 oblateness coefficient

CRITICAL_KM    = 0.100          # Collision threshold  100 m      [km]
WARNING_KM     = 5.000          # Close-approach threshold        [km]

DRY_MASS_KG    = 500.0          # Satellite dry mass              [kg]
INIT_FUEL_KG   = 50.0           # Initial propellant mass         [kg]
WET_MASS_KG    = DRY_MASS_KG + INIT_FUEL_KG

ISP            = 300.0          # Specific impulse (monoprop)     [s]
G0_SI          = 9.80665        # Standard gravity                [m s^-2]
G0             = G0_SI / 1000.0 # Standard gravity                [km s^-2]

MAX_DV_KMS     = 0.015          # Max single-burn delta-v 15 m/s  [km/s]
COOLDOWN_S     = 600.0          # Thruster cooldown period        [s]
STATION_BOX_KM = 10.0           # Station-keeping radius          [km]
LATENCY_S      = 10.0           # Min command lead-time           [s]
EOL_FUEL_FRAC  = 0.05           # End-of-life fuel threshold (5%)


Vec = List[float]   # 3-element [x, y, z] float list


def mag(v: Vec) -> float:
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def vadd(a: Vec, b: Vec) -> Vec:
    return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]

def vsub(a: Vec, b: Vec) -> Vec:
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

def vscale(v: Vec, s: float) -> Vec:
    return [v[0]*s, v[1]*s, v[2]*s]

def vdot(a: Vec, b: Vec) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def vcross(a: Vec, b: Vec) -> Vec:
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]

def vnorm(v: Vec) -> Vec:
    m = mag(v)
    if m == 0.0:
        raise ValueError("Cannot normalise a zero vector.")
    return vscale(v, 1.0 / m)


class ObjState:

    __slots__ = (
        "id", "obj_type", "r", "v",
        "mass_kg", "fuel_kg", "status",
        "nominal_r", "maneuver_queue",
        "last_burn_time_s", "collision_count",
        "uptime_s", "outage_s",
    )

    def __init__(self, obj_id: str, obj_type: str, r: Vec, v: Vec):
        self.id                = obj_id
        self.obj_type          = obj_type.upper()
        self.r                 = list(r)
        self.v                 = list(v)
        self.mass_kg           = WET_MASS_KG
        self.fuel_kg           = INIT_FUEL_KG
        self.status            = "NOMINAL"
        self.nominal_r         = list(r)
        self.maneuver_queue: List[dict] = []
        self.last_burn_time_s: float    = -1e9
        self.collision_count: int       = 0
        self.uptime_s: float            = 0.0
        self.outage_s: float            = 0.0

    @classmethod
    def from_db_row(cls, row: dict) -> "ObjState":
        """Reconstruct an ObjState from a database row dict."""
        obj = cls.__new__(cls)
        obj.id               = row["id"]
        obj.obj_type         = row["obj_type"]
        obj.r                = [row["rx"], row["ry"], row["rz"]]
        obj.v                = [row["vx"], row["vy"], row["vz"]]
        obj.mass_kg          = row["mass_kg"]
        obj.fuel_kg          = row["fuel_kg"]
        obj.status           = row["status"]
        obj.nominal_r        = [row["nominal_rx"], row["nominal_ry"], row["nominal_rz"]]
        obj.last_burn_time_s = row["last_burn_time_s"]
        obj.collision_count  = row["collision_count"]
        obj.uptime_s         = row["uptime_s"]
        obj.outage_s         = row["outage_s"]
        obj.maneuver_queue   = []   # reloaded separately from maneuver_queue table
        return obj


class Sim:
    def __init__(self):
        self.objects:   Dict[str, ObjState] = {}
        self.sim_time:  datetime = datetime.now(timezone.utc)
        self.wall_time: float    = time.time()
        self.total_collisions:         int = 0
        self.total_maneuvers_executed: int = 0
        self.active_cdm_warnings:      int = 0


# Module-level singleton
SIM = Sim()


def _j2_accel(r: Vec) -> Vec:
    x, y, z = r
    rm  = mag(r)
    f   = 1.5 * J2 * MU * RE**2 / rm**5
    zr2 = (z / rm) ** 2
    return [
        f * x * (5 * zr2 - 1),
        f * y * (5 * zr2 - 1),
        f * z * (5 * zr2 - 3),
    ]


def _state_derivatives(r: Vec, v: Vec) -> Tuple[Vec, Vec]:
    rm     = mag(r)
    a_grav = vscale(r, -MU / rm**3)
    a_j2   = _j2_accel(r)
    a_tot  = vadd(a_grav, a_j2)
    return v, a_tot


def rk4_step(r: Vec, v: Vec, dt: float) -> Tuple[Vec, Vec]:
    """4th-order Runge-Kutta step."""
    dr1, dv1 = _state_derivatives(r, v)
    r2 = vadd(r, vscale(dr1, dt/2)); v2 = vadd(v, vscale(dv1, dt/2))
    dr2, dv2 = _state_derivatives(r2, v2)
    r3 = vadd(r, vscale(dr2, dt/2)); v3 = vadd(v, vscale(dv2, dt/2))
    dr3, dv3 = _state_derivatives(r3, v3)
    r4 = vadd(r, vscale(dr3, dt));   v4 = vadd(v, vscale(dv3, dt))
    dr4, dv4 = _state_derivatives(r4, v4)
    r_new = [r[i] + dt/6*(dr1[i]+2*dr2[i]+2*dr3[i]+dr4[i]) for i in range(3)]
    v_new = [v[i] + dt/6*(dv1[i]+2*dv2[i]+2*dv3[i]+dv4[i]) for i in range(3)]
    return r_new, v_new


def propagate_obj(obj: ObjState, dt: float, sub_step: float = 10.0) -> None:
    """Propagate ObjState in-place for dt seconds using RK4 sub-steps."""
    remaining = dt
    while remaining > 0.0:
        s = min(sub_step, remaining)
        obj.r, obj.v = rk4_step(obj.r, obj.v, s)
        remaining -= s


def propagate_vectors(r: Vec, v: Vec, dt: float,
                      sub_step: float = 10.0) -> Tuple[Vec, Vec]:
    """Stateless propagation — returns new (r, v) without touching any ObjState."""
    r_cur, v_cur = list(r), list(v)
    remaining = dt
    while remaining > 0.0:
        s = min(sub_step, remaining)
        r_cur, v_cur = rk4_step(r_cur, v_cur, s)
        remaining -= s
    return r_cur, v_cur


def separation(r_sat: Vec, r_deb: Vec) -> float:
    return mag(vsub(r_sat, r_deb))


def conjunction_status(r_sat: Vec, r_deb: Vec) -> Tuple[float, str]:
    d = separation(r_sat, r_deb)
    if d < CRITICAL_KM:
        return d, "CRITICAL"
    if d < WARNING_KM:
        return d, "WARNING"
    return d, "SAFE"


def conjunction_report(r_sat: Vec, r_deb: Vec) -> dict:
    d, status = conjunction_status(r_sat, r_deb)
    if status == "CRITICAL":
        msg = f"Collision imminent! d={d:.4f} km  (< {CRITICAL_KM} km limit)"
    elif status == "WARNING":
        msg = f"Close approach detected. d={d:.4f} km  (< {WARNING_KM} km limit)"
    else:
        msg = f"No collision risk. d={d:.4f} km  (>= {WARNING_KM} km limit)"
    return {"distance_km": round(d, 6), "status": status, "message": msg}


def batch_cdm_check(sat: ObjState) -> List[dict]:
    """Scan all debris in SIM and return non-SAFE conjunctions sorted by severity."""
    results = []
    for obj in SIM.objects.values():
        if obj.obj_type != "DEBRIS":
            continue
        d, status = conjunction_status(sat.r, obj.r)
        if status != "SAFE":
            results.append({
                "debris_id":   obj.id,
                "distance_km": round(d, 6),
                "status":      status,
            })
    results.sort(key=lambda x: (0 if x["status"] == "CRITICAL" else 1,
                                 x["distance_km"]))
    return results


def rtn_frame(r: Vec, v: Vec) -> Tuple[Vec, Vec, Vec]:
    R_hat = vnorm(r)
    N_hat = vnorm(vcross(r, v))
    T_hat = vcross(N_hat, R_hat)
    return R_hat, T_hat, N_hat


def tsiolkovsky(mass_kg: float, dv_kms: float) -> float:
    """Propellant mass consumed: dm = m*(1 - exp(-dv/(Isp*g0)))"""
    return mass_kg * (1.0 - math.exp(-dv_kms / (ISP * G0)))


def plan_avoidance_maneuver(
    sat      : ObjState,
    deb      : ObjState,
    tca_s    : float,
    margin_s : float = 120.0,
) -> Optional[dict]:

    d, status = conjunction_status(sat.r, deb.r)
    if status != "CRITICAL":
        log.debug(f"plan_avoidance: {sat.id} vs {deb.id} not CRITICAL (status={status})")
        return None

    _R, T_hat, _N = rtn_frame(sat.r, sat.v)

    r_diff = vsub(sat.r, deb.r)
    dv_dir = T_hat if vdot(r_diff, T_hat) >= 0 else vscale(T_hat, -1.0)

    dv_mag = max(0.001, CRITICAL_KM - d + 0.050)
    dv_mag = min(dv_mag, MAX_DV_KMS)

    fuel_ev  = tsiolkovsky(sat.mass_kg, dv_mag)
    fuel_rec = tsiolkovsky(sat.mass_kg - fuel_ev, dv_mag)
    fuel_tot = fuel_ev + fuel_rec
    fuel_ok  = sat.fuel_kg >= fuel_tot

    if not fuel_ok:
        half_fuel = min(sat.fuel_kg / 2.0, sat.mass_kg * 0.9)
        dv_mag    = -ISP * G0 * math.log(1.0 - half_fuel / sat.mass_kg)
        dv_mag    = min(dv_mag, MAX_DV_KMS)
        fuel_ev   = tsiolkovsky(sat.mass_kg, dv_mag)
        fuel_rec  = tsiolkovsky(sat.mass_kg - fuel_ev, dv_mag)
        fuel_tot  = fuel_ev + fuel_rec
        log.warning(f"Fuel-limited maneuver for {sat.id}: dv scaled to {dv_mag*1000:.2f} m/s")

    ev_vec       = vscale(dv_dir,  dv_mag)
    rec_vec      = vscale(dv_dir, -dv_mag)
    ev_offset_s  = max(0.0, tca_s - margin_s)
    rec_offset_s = ev_offset_s + COOLDOWN_S

    return {
        "evasion_burn": {
            "time_offset_s": round(ev_offset_s, 2),
            "dv_vector_kms": [round(c, 9) for c in ev_vec],
            "dv_mag_ms":     round(dv_mag * 1000, 4),
            "fuel_kg":       round(fuel_ev, 6),
        },
        "recovery_burn": {
            "time_offset_s": round(rec_offset_s, 2),
            "dv_vector_kms": [round(c, 9) for c in rec_vec],
            "dv_mag_ms":     round(dv_mag * 1000, 4),
            "fuel_kg":       round(fuel_rec, 6),
        },
        "total_fuel_kg":     round(fuel_tot, 6),
        "fuel_remaining_kg": round(sat.fuel_kg - fuel_tot, 6),
        "fuel_sufficient":   fuel_ok,
    }


def execute_burn(sat: ObjState, dv: Vec, burn_time_abs: float,
                 burn_id: str = "MANUAL",
                 sim_time: Optional[datetime] = None) -> dict:
    dv_mag    = mag(dv)
    fuel_used = tsiolkovsky(sat.mass_kg, dv_mag)

    sat.v                = vadd(sat.v, dv)
    sat.mass_kg         -= fuel_used
    sat.fuel_kg         -= fuel_used
    sat.last_burn_time_s = burn_time_abs

    result = {
        "dv_mag_ms":         round(dv_mag * 1000, 4),
        "fuel_consumed_kg":  round(fuel_used, 6),
        "mass_remaining_kg": round(sat.mass_kg, 4),
        "fuel_remaining_kg": round(sat.fuel_kg, 4),
    }

    log.info(
        f"Burn [{burn_id}] on {sat.id}: "
        f"dv={dv_mag*1000:.3f} m/s  "
        f"fuel_used={fuel_used:.4f} kg  "
        f"fuel_left={sat.fuel_kg:.3f} kg"
    )

    try:
        from database import log_executed_burn
        _t = sim_time or datetime.now(timezone.utc)
        log_executed_burn(
            satellite_id     = sat.id,
            burn_id          = burn_id,
            burn_time_abs    = burn_time_abs,
            dv               = dv,
            dv_mag_ms        = result["dv_mag_ms"],
            fuel_consumed_kg = result["fuel_consumed_kg"],
            mass_after_kg    = result["mass_remaining_kg"],
            fuel_after_kg    = result["fuel_remaining_kg"],
            sim_time         = _t,
        )
    except Exception as exc:
        log.warning(f"burn_log DB write failed: {exc}")

    return result



def eci_to_lla(r: Vec, t: datetime) -> Tuple[float, float, float]:
    """ECI position -> approximate (lat_deg, lon_deg, alt_km) via GST rotation."""
    x, y, z = r
    j2000_s = (t - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)).total_seconds()
    gst_deg = (280.46061837 + 360.98564736629 * j2000_s / 86400.0) % 360.0
    gst_rad = math.radians(gst_deg)
    cos_g, sin_g = math.cos(gst_rad), math.sin(gst_rad)
    xe =  x * cos_g + y * sin_g
    ye = -x * sin_g + y * cos_g
    ze = z
    r_mag = math.sqrt(xe**2 + ye**2 + ze**2)
    lat   = math.degrees(math.asin(ze / r_mag))
    lon   = math.degrees(math.atan2(ye, xe))
    alt   = r_mag - RE
    return round(lat, 4), round(lon, 4), round(alt, 2)


def check_eol(sat: ObjState) -> bool:
    """Mark satellite EOL if fuel <= 5% of initial propellant."""
    if sat.fuel_kg <= EOL_FUEL_FRAC * INIT_FUEL_KG and sat.status != "EOL":
        sat.status = "EOL"
        log.warning(f"{sat.id} fuel critical ({sat.fuel_kg:.2f} kg) -> EOL")
        return True
    return False


def update_station_keeping(sat: ObjState, dt: float) -> None:
    """Increment uptime_s or outage_s based on slot proximity."""
    slot_err = mag(vsub(sat.r, sat.nominal_r))
    if slot_err <= STATION_BOX_KM:
        sat.uptime_s += dt
    else:
        sat.outage_s += dt