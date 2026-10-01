"""Measurement states. Only FULL is eligible for the purchase gate.

FULL         real workload, all stages representative, evidence complete
PARTIAL      real workload but a required stage is missing/degraded (never gate-eligible)
FALLBACK     informational run on a substitute workload (never gate-eligible)
UNSUPPORTED  workload cannot run here (missing weights/dep/hardware)
FAILED       workload errored
INCONCLUSIVE evidence insufficient to decide (fail-closed: not a pass)
ABORTED      supervisor stopped the run (timeout/safety/cancel)
NOT_RUN      scheduled but never executed
"""
FULL = "FULL"
PARTIAL = "PARTIAL"
FALLBACK = "FALLBACK"
UNSUPPORTED = "UNSUPPORTED"
FAILED = "FAILED"
INCONCLUSIVE = "INCONCLUSIVE"
ABORTED = "ABORTED"
NOT_RUN = "NOT_RUN"

GATE_ELIGIBLE = {FULL}

# Purchase-gate verdicts (module/workload level and overall).
PASS = "PASS"
PASS_WITH_HEADROOM = "PASS_WITH_HEADROOM"
LIMITED = "LIMITED"
FAIL = "FAIL"

def gate_counts(results):
    """How many modules reached each state — missing evidence is visible, not silent."""
    from collections import Counter
    mods = [v for k, v in results.items()
            if not k.startswith("_") and isinstance(v, dict)]
    return dict(Counter(v.get("status", NOT_RUN) for v in mods))
