import pytest

from emac_sim.linear_plant import CoilStation, LinearActuatorParams, net_force


class ConstantLut:
    """A trivial force_lut stand-in -- net_force only needs something callable as
    (offset_m, current_a) -> force_n, not a real ForceLUT."""

    def __init__(self, value: float):
        self.value = value
        self.calls = []

    def __call__(self, offset_m: float, current_a: float) -> float:
        self.calls.append((offset_m, current_a))
        return self.value


def test_net_force_uses_force_lut_when_present_instead_of_analytic_lobe():
    lut = ConstantLut(2.5)
    coil = CoilStation(position_m=0.0, x_c=0.02, Cmag=0.4, k_a=0.3, force_lut=lut)
    p = LinearActuatorParams(coils=(coil,))

    force = net_force(0.01, [3.0], p)

    assert force == pytest.approx(2.5)
    assert lut.calls == [(0.01, 3.0)]


def test_net_force_without_force_lut_is_unaffected():
    """Regression guard: force_lut defaults to None, so every existing config (which never
    sets it) must produce bit-identical net_force output to before this field existed."""
    coil = CoilStation(position_m=0.0, x_c=0.02, Cmag=0.4, k_a=0.3)
    p = LinearActuatorParams(coils=(coil,))
    assert coil.force_lut is None
    force = net_force(0.01, [3.0], p)
    assert force != 0.0  # sanity: the analytic lobe is actually doing something here


def test_net_force_mixes_lut_and_analytic_coils():
    lut = ConstantLut(1.0)
    lut_coil = CoilStation(position_m=0.0, force_lut=lut)
    analytic_coil = CoilStation(position_m=0.05, x_c=0.02, Cmag=0.0, k_a=0.2)
    p = LinearActuatorParams(coils=(lut_coil, analytic_coil))

    only_lut = net_force(0.0, [3.0, 0.0], p)
    only_analytic = net_force(0.0, [0.0, 3.0], p)
    both = net_force(0.0, [3.0, 3.0], p)

    assert only_lut == pytest.approx(1.0)
    assert both == pytest.approx(only_lut + only_analytic)


def test_net_force_skips_lut_call_for_zero_current_coil():
    """net_force's existing `if i_k == 0.0: continue` short-circuit should still apply to
    a LUT coil -- no reason to call into the LUT for a coil that isn't energized."""
    lut = ConstantLut(99.0)
    coil = CoilStation(position_m=0.0, force_lut=lut)
    p = LinearActuatorParams(coils=(coil,))

    force = net_force(0.01, [0.0], p)

    assert force == pytest.approx(0.0)
    assert lut.calls == []
