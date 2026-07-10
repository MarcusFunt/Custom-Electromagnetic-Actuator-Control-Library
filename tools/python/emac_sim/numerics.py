"""Numerical helper routines shared by the host physics engines.

These are deliberately small and dependency-free: the same ideas should be easy to port
into the later embedded C++ reference model if needed.
"""

from __future__ import annotations



def hermite_event_fraction(
    y0: float,
    v0: float,
    y1: float,
    v1: float,
    dt: float,
    y_event: float = 0.0,
) -> tuple[float, float]:
    """Return the fractional event time and velocity from a cubic Hermite segment.

    ``y0``/``v0`` and ``y1``/``v1`` are the position-like coordinate and velocity-like
    derivative at the beginning and end of a step of length ``dt``.  The function solves
    for ``y(t) == y_event`` on ``0 <= t <= dt`` using the cubic Hermite interpolant that
    matches both endpoint values and both endpoint slopes.  This is a better crossing-time
    estimate than plain linear interpolation because it uses the velocities the integrator
    already computed instead of pretending the coordinate moved at constant speed through
    the whole tick.

    The caller should only use this after it already knows the event is bracketed.  If the
    bracket is numerically degenerate, the function falls back to a clamped linear fraction
    rather than raising; event detection should be robust even for tiny steps near a gate.
    """
    if dt <= 0.0:
        return 0.0, v0

    z0 = y0 - y_event
    z1 = y1 - y_event
    if z0 == 0.0:
        return 0.0, v0
    if z1 == 0.0:
        return 1.0, v1

    m0 = v0 * dt
    m1 = v1 * dt

    def h(s: float) -> float:
        s2 = s * s
        s3 = s2 * s
        return (
            (2.0 * s3 - 3.0 * s2 + 1.0) * z0
            + (s3 - 2.0 * s2 + s) * m0
            + (-2.0 * s3 + 3.0 * s2) * z1
            + (s3 - s2) * m1
        )

    def dh_dt(s: float) -> float:
        s2 = s * s
        dh_ds = (
            (6.0 * s2 - 6.0 * s) * z0
            + (3.0 * s2 - 4.0 * s + 1.0) * m0
            + (-6.0 * s2 + 6.0 * s) * z1
            + (3.0 * s2 - 2.0 * s) * m1
        )
        return dh_ds / dt

    # Robust bracketed bisection.  Linear interpolation is only used as a fallback if the
    # endpoint signs no longer bracket because of roundoff or an accidental caller error.
    if z0 * z1 > 0.0:
        denom = y0 - y1
        s = (y0 - y_event) / denom if denom != 0.0 else 0.0
        s = max(0.0, min(1.0, s))
        return s, dh_dt(s)

    lo, hi = 0.0, 1.0
    f_lo = z0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = h(mid)
        if f_mid == 0.0:
            lo = hi = mid
            break
        if f_lo * f_mid <= 0.0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid

    s = 0.5 * (lo + hi)
    return s, dh_dt(s)
