# --- Monkey-patch to guard against None demo_candidates ---
try:
    import dspy.propose.grounded_proposer as gp
except ImportError:
    import dspy.teleprompt.grounded_proposer as gp

_OrigPropose = gp.GroundedProposer.propose_instructions_for_program

def _normalize_demo_candidates(demo_candidates, predictors):
    if demo_candidates is None:
        return [[] for _ in predictors]
    fixed = []
    for i in range(len(predictors)):
        if isinstance(demo_candidates, (list, tuple)) and i < len(demo_candidates):
            fixed.append(demo_candidates[i] if demo_candidates[i] is not None else [])
        else:
            fixed.append([])
    return fixed

def _patched(self, *, trainset, program, demo_candidates=None, **kw):
    predictors = program.predictors() if hasattr(program, "predictors") else []
    demo_candidates = _normalize_demo_candidates(demo_candidates, predictors)
    return _OrigPropose(self, trainset=trainset, program=program,
                        demo_candidates=demo_candidates, **kw)

gp.GroundedProposer.propose_instructions_for_program = _patched
# ----------------------------------------------------------

