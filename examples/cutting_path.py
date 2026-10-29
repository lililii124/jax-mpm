"""Event windows computed from actual accumulated State.time; no clock snapping."""
import math
import numpy as np


def clock_tolerance(time):
    # At most 500 nominal additions occur between cache reset clocks; this
    # bound covers their accumulated rounding, with no change to State.time.
    return 1024 * np.finfo(float).eps * max(abs(float(time)), 1e-12)


def velocity_clock_tolerance(time):
    # Strictly inside MovingWedgeTool.interval_is_linear's 64eps bound.
    return 16 * np.finfo(float).eps * max(abs(float(time)), 1e-12)


def next_window(actual_time, event_time, nominal_dt, capacity, nominal_phase=False, velocity_event=False):
    remaining = float(event_time) - float(actual_time)
    tolerance = velocity_clock_tolerance(event_time) if velocity_event else clock_tolerance(event_time)
    assert not (velocity_event and nominal_phase)
    if abs(remaining) <= tolerance:
        return {"done": True, "endpoint_offset_s": -remaining,
                "roundoff_tolerance_s": tolerance}
    if remaining < 0:
        raise RuntimeError("State.time passed a velocity event beyond roundoff")
    if nominal_phase:
        nearest=int(round(remaining/nominal_dt))
        if nearest>0 and abs(remaining-nearest*nominal_dt)<=tolerance:
            return {"done":False,"steps":min(capacity,nearest),"dt_s":float(nominal_dt),
                "remaining_time_before_window_s":remaining,"event_time_s":float(event_time),
                "final_remaining_time_window":False,"nominal_cache_phase_roundoff":True}
    # Leave at least one nominal step before the event, so accumulated clock
    # roundoff cannot make a full batch cross a velocity kink.
    safe_count = max(0, int(math.floor(remaining / nominal_dt)) - 1)
    if safe_count:
        count = min(capacity, safe_count)
        dt = float(nominal_dt)
    else:
        count = max(1, int(math.ceil(remaining / nominal_dt)))
        dt = remaining / count
    assert 0 < dt <= nominal_dt * (1 + 8 * np.finfo(float).eps)
    return {"done": False, "steps": count, "dt_s": dt,
            "remaining_time_before_window_s": remaining,
            "event_time_s": float(event_time),
            "final_remaining_time_window": safe_count == 0}
