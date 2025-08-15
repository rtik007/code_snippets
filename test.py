import inspect
import random
import numpy as np
# import torch  # if you use it elsewhere and want deterministic runs

# ---------- Safe compile wrapper ----------
def compile_with_compat(
    teleprompter,
    student,
    trainset,
    devset=None,
    *,
    seed=None,
    minibatch=None,          # True/False/None (None => do not touch)
    minibatch_size=16,       # auto-capped to len(devset) if forwarded
    require_permission=False,
    verbose=False,
    track_stats=True,
):
    """
    Calls teleprompter.compile() with only the kwargs it supports.
    Also handles minibatch/minibatch_size safely if present.
    """
    sig = inspect.signature(teleprompter.compile)
    params = set(sig.parameters.keys())
    kwargs = {}

    # Required/common
    if 'student'  in params: kwargs['student']  = student
    if 'trainset' in params: kwargs['trainset'] = trainset

    # dev/val argument name varies by version
    if devset is not None:
        if 'valset' in params:
            kwargs['valset'] = devset
        elif 'devset' in params:
            kwargs['devset'] = devset
        elif 'validset' in params:
            kwargs['validset'] = devset

    # optional controls (only if supported)
    if seed is not None and 'seed' in params:
        kwargs['seed'] = seed
    if 'requires_permission_to_run' in params:
        kwargs['requires_permission_to_run'] = bool(require_permission)
    if 'verbose' in params:
        kwargs['verbose'] = bool(verbose)
    if 'track_stats' in params:
        kwargs['track_stats'] = bool(track_stats)

    # minibatch handling if compile() supports it
    if 'minibatch' in params and minibatch is not None:
        kwargs['minibatch'] = bool(minibatch)
    if 'minibatch_size' in params and minibatch is not None and minibatch:
        safe_size = len(devset) if devset is not None else minibatch_size
        kwargs['minibatch_size'] = min(int(minibatch_size), int(safe_size))

    return teleprompter.compile(**kwargs)


# ---------- Main training helper ----------
def train_with_optimizer(
    optimizer_name: str,
    metric_fn,
    trainset,
    devset,
    *,
    seed: int | None = None,
    temperature: float = 0.8,   # kept <= 1.0 for providers that enforce it
    num_candidates: int = 8,
):
    """
    Train EvidenceModule with a chosen DSPy optimizer.
    Returns: (program, teleprompter)
    """
    # Reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        # torch.manual_seed(seed)

    # Some providers reject temp > 1.0
    temperature = float(min(1.0, max(0.0, temperature)))

    name = optimizer_name.lower()

    if name == "bootstrap":
        teleprompter = BootstrapFewShot(
            metric=metric_fn,
            max_bootstrapped_demos=6,
            max_labeled_demos=5,
            max_rounds=5,
            teacher_settings=dict(temperature=temperature),
        )
        program = compile_with_compat(
            teleprompter, EvidenceModule(), trainset, devset,
            seed=seed, minibatch=None  # BootstrapFewShot.compile usually has no minibatch
        )

    elif name == "copro":
        teleprompter = COPRO(
            metric=metric_fn,
            breadth=8,
            depth=5,
            init_temperature=temperature,
            track_stats=True,
            verbose=False,
        )
        program = compile_with_compat(
            teleprompter, EvidenceModule(), trainset, devset,
            seed=seed, minibatch=None
        )

    elif name == "miprov2":
        # Do NOT pass unsupported kwargs; keep it minimal and safe
        teleprompter = MIPROv2(
            metric=metric_fn,
            num_candidates=num_candidates,
            init_temperature=temperature,
            num_threads=8,
            max_bootstrapped_demos=6,
            max_labeled_demos=6,
            seed=seed,          # supported in many builds; harmless if ignored
        )
        # MIPROv2 defaults to minibatch=True and size ~35; disable to avoid valset-size error
        program = compile_with_compat(
            teleprompter, EvidenceModule(), trainset, devset,
            seed=seed, minibatch=False
        )

    elif name == "simba":
        teleprompter = SIMBA(
            metric=metric_fn,
            bsize=5,
            max_iters=50,
            num_trials=3,
            init_temperature=temperature,
            seed=seed,
        )
        # SIMBA may accept minibatch; wrapper will only forward if supported
        program = compile_with_compat(
            teleprompter, EvidenceModule(), trainset, devset,
            seed=seed, minibatch=False
        )

    else:
        raise ValueError("Unknown optimizer. Choose from: 'bootstrap', 'copro', 'miprov2', 'simba'.")

    return program, teleprompter
