import inspect

def compile_with_compat(
    teleprompter,
    student,
    trainset,
    devset=None,
    *,
    seed=None,
    minibatch=None,                 # True/False or None to leave default
    minibatch_size=None,            # int or None
    require_permission=False,       # set True if you want interactive prompts
    verbose=False,
    track_stats=True,
):
    """
    Compatibility wrapper for teleprompter.compile across DSPy optimizers.
    - Accepts minibatch/minibatch_size and applies them via kwargs or attributes.
    - Caps minibatch_size to len(devset) when provided.
    """

    sig = inspect.signature(teleprompter.compile)
    params = set(sig.parameters.keys())
    kwargs = {}

    # Required/common params
    if 'student'  in params: kwargs['student']  = student
    if 'trainset' in params: kwargs['trainset'] = trainset

    # Figure out which dev/val name to use
    if devset is not None:
        if 'valset' in params:
            kwargs['valset'] = devset
        elif 'devset' in params:
            kwargs['devset'] = devset
        elif 'validset' in params:
            kwargs['validset'] = devset

    # Optional seed
    if seed is not None and 'seed' in params:
        kwargs['seed'] = seed

    # Optional flags if supported
    if 'requires_permission_to_run' in params:
        kwargs['requires_permission_to_run'] = bool(require_permission)
    if 'verbose' in params:
        kwargs['verbose'] = bool(verbose)
    if 'track_stats' in params:
        kwargs['track_stats'] = bool(track_stats)

    # ---- Handle minibatch / minibatch_size safely ----
    # Cap size to len(devset) if we have one
    if minibatch_size is not None and devset is not None:
        minibatch_size = min(int(minibatch_size), len(devset))

    forwarded_minibatch = False
    forwarded_mbsize = False

    # Prefer passing via compile kwargs when available
    if 'minibatch' in params and minibatch is not None:
        kwargs['minibatch'] = bool(minibatch)
        forwarded_minibatch = True
    if 'minibatch_size' in params and minibatch_size is not None:
        kwargs['minibatch_size'] = int(minibatch_size)
        forwarded_mbsize = True

    # Otherwise, fall back to setting teleprompter attributes if present
    if not forwarded_minibatch and minibatch is not None and hasattr(teleprompter, 'minibatch'):
        setattr(teleprompter, 'minibatch', bool(minibatch))
    if not forwarded_mbsize and minibatch_size is not None and hasattr(teleprompter, 'minibatch_size'):
        setattr(teleprompter, 'minibatch_size', int(minibatch_size))

    # Finally call compile with only supported kwargs
    return teleprompter.compile(**kwargs)
