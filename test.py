# --- Safety patch for DSPy proposer outputs (demo_candidates + instruction_candidates) ---
import sys, traceback

# Try both possible module paths depending on your DSPy version
try:
    import dspy.propose.grounded_proposer as gp
except ImportError:
    import dspy.teleprompt.grounded_proposer as gp

# Keep original method
_OrigPropose = gp.GroundedProposer.propose_instructions_for_program

def _to_list_of_lists(x, predictors):
    """Coerce x into a list-of-lists aligned with predictor order."""
    n = len(predictors)
    # Start with n empty lists
    lol = [[] for _ in range(n)]

    if x is None:
        return lol

    # If already a list/tuple of length n, clean inner None -> []
    if isinstance(x, (list, tuple)):
        if len(x) == n:
            return [([] if xi is None else list(xi)) for xi in x]
        # If list but wrong length, truncate/pad
        y = list(x)[:n]
        while len(y) < n:
            y.append([])
        return [([] if yi is None else list(yi)) for yi in y]

    # If dict-like, try keys by index, then predictor object, then signature/name
    if isinstance(x, dict):
        for i, pred in enumerate(predictors):
            val = None
            if i in x:
                val = x.get(i)
            elif pred in x:
                val = x.get(pred)
            else:
                # common alt keys
                key_candidates = [
                    getattr(pred, "name", None),
                    getattr(pred, "__name__", None),
                    getattr(pred, "signature", None),
                    str(pred),
                ]
                for k in key_candidates:
                    if k in x:
                        val = x[k]; break
            lol[i] = [] if (val is None) else list(val)
        return lol

    # Fallback: unknown type → give empties
    return lol

def _normalize_demo_candidates(demo_candidates, predictors):
    # Ensure a list-of-lists for demo candidates too
    return _to_list_of_lists(demo_candidates, predictors)

def _patched(self, *, trainset, program, demo_candidates=None, **kw):
    predictors = program.predictors() if hasattr(program, "predictors") else []
    demo_candidates = _normalize_demo_candidates(demo_candidates, predictors)
    # Call original
    instr = _OrigPropose(self, trainset=trainset, program=program,
                         demo_candidates=demo_candidates, **kw)
    # Normalize instruction candidates shape
    instr = _to_list_of_lists(instr, predictors)
    return instr

gp.GroundedProposer.propose_instructions_for_program = _patched
# ---------------------------------------------------------------------
