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

STOP_FILES = ("STOP", "STOP_NEXUS", "STOP.txt")

def stop_requested():
    """File-based stop: creating a file named STOP next to the launcher aborts
    the run (for when Ctrl+C is impractical). Checked between modules and
    inside long loops."""
    import os
    return any(os.path.exists(f) for f in STOP_FILES)

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
