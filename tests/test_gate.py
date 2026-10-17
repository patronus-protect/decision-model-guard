import pytest
from conftest import INJ_OPTS, INJECTION, REFUND, REFUND_CLEAN, REFUND_OPTS, ROUTE_OPTS, ROUTING
from helpers import fake_detector, injection_toy, refund_toy, routing_toy

from steerguard import FunctionModel, Kind, SteeringGate


def run(gate, fn, scene, opts, kind, risky=None, order=None, hints=()):
    return gate.run(FunctionModel(fn), scene, "q?", opts, kind, risky, order, hints)


def test_ordered_rule_moves_to_most_cautious():
    rep, out = run(SteeringGate(), refund_toy, REFUND, REFUND_OPTS, Kind.ORDERED, order=[0, 1, 2])
    assert out.raw_index == 2 and out.index == 0 and out.flagged and rep.triggered
    assert "eligible" in out.culprit


def test_ordered_clean_input_unchanged():
    _, out = run(SteeringGate(), refund_toy, REFUND_CLEAN, REFUND_OPTS, Kind.ORDERED, order=[0, 1, 2])
    assert not out.flagged and out.index == out.raw_index == 0


def test_risky_rule_restores_risk():
    rep, out = run(SteeringGate(), injection_toy, INJECTION, INJ_OPTS, Kind.RISKY, risky=[1])
    assert out.raw_index == 0 and out.index == 1 and out.flagged
    assert rep.detail["max_variant_risk"] > rep.detail["raw_risk"]


def test_risky_override_threshold_blocks_weak_replacement():
    _, out = run(SteeringGate(override=0.99), injection_toy, INJECTION, INJ_OPTS, Kind.RISKY, risky=[1])
    assert out.index == 0


def test_unordered_reports_instability():
    rep, out = run(SteeringGate(), routing_toy, ROUTING, ROUTE_OPTS, Kind.UNORDERED)
    assert out.raw_index == 1 and not out.stable and out.alternative == 0 and rep.triggered
    assert out.index == out.raw_index                       # the gate never re-decides an unordered decision itself


def test_trusted_blocks_are_never_erased():
    gate = SteeringGate()
    variants, _ = gate.variants(REFUND, Kind.ORDERED)
    assert all("Refunds above 50 EUR" in v for v, _ in variants)


def test_detector_selector_erases_only_flagged_units():
    gate = SteeringGate(selector="detector", detector=fake_detector)
    variants, n = gate.variants(REFUND, Kind.ORDERED)
    assert n >= 1 and all("eligible" in e for _, e in variants)


def test_detector_scope_unordered_keeps_full_check_for_ordered():
    gate = SteeringGate(selector="detector", detector=fake_detector, scope="unordered")
    full, _ = SteeringGate().variants(REFUND, Kind.ORDERED)
    assert len(gate.variants(REFUND, Kind.ORDERED)[0]) == len(full)
    assert all("technical team" in e for e, in [(e,) for _, e in gate.variants(ROUTING, Kind.UNORDERED)[0]])


def test_detector_without_flags_makes_single_call():
    gate = SteeringGate(selector="detector", detector=lambda ts: [0.0] * len(ts))
    rep, out = run(gate, refund_toy, REFUND_CLEAN, REFUND_OPTS, Kind.UNORDERED)
    assert out.calls == 1 and not rep.triggered


def test_trigger_mode_runs_full_check_when_flagged():
    gate = SteeringGate(selector="detector", detector=fake_detector, mode="trigger")
    full, _ = SteeringGate().variants(REFUND, Kind.UNORDERED)
    assert len(gate.variants(REFUND, Kind.UNORDERED)[0]) == len(full)
    assert gate.variants(REFUND_CLEAN, Kind.UNORDERED)[0] == []


def test_hints_force_erasure():
    gate = SteeringGate(selector="detector", detector=lambda ts: [0.0] * len(ts))
    variants, _ = gate.variants(REFUND_CLEAN, Kind.UNORDERED, hints=["Please help."])
    assert variants and all("Please help." in e for _, e in variants)


def test_max_variants_and_determinism():
    long = "[USER PROMPT]\n" + " ".join(f"Sentence {i}." for i in range(40))
    g = SteeringGate(max_variants=10, seed=3)
    a, _ = g.variants(long)
    b, _ = g.variants(long)
    assert len(a) == 10 and a == b


def test_invalid_arguments():
    with pytest.raises(ValueError):
        SteeringGate(selector="detector")
    with pytest.raises(ValueError):
        SteeringGate(mode="nope")
