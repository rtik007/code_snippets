import random
import numpy as np
# If you use PyTorch for anything else in the pipeline, import torch and set its seed as well.

def train_with_optimizer(
    optimizer_name: str,
    metric_fn,
    trainset,
    devset,
    seed: int | None = None,
):
    """
    Trains the EvidenceModule with the specified optimizer.

    Args:
        optimizer_name (str): Name of the optimizer ('bootstrap', 'copro', 'miprov2', 'simba')
        metric_fn: Metric function for evaluation
        trainset: Training dataset
        devset: Development/validation dataset
        seed (int | None): Random seed for reproducibility. If None, uses random behaviour.

    Returns:
        tuple: (compiled_program, teleprompter)
    """
    # Set Python-level seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        # torch.manual_seed(seed)  # Uncomment if you use torch elsewhere

    # Choose and configure the optimizer/teleprompter
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "bootstrap":
        # BootstrapFewShot doesn’t currently expose a seed parameter, but setting
        # random and NumPy seeds above will make its sampling deterministic.
        teleprompter = BootstrapFewShot(
            metric=metric_fn,
            max_bootstrapped_demos=6,
            max_labeled_demos=5,
            max_rounds=5,
        )
        program = compile_with_copmat(teleprompter, EvidenceModule(), trainset, devset)

    elif optimizer_name == "copro":
        teleprompter = COPRO(
            metric=metric_fn,
            breadth=8,
            depth=5,
            init_temperature=2.0,
            track_stats=True,
            verbose=True,
        )
        eval_kwargs = {"num_threads": 1, "display_progress": True}
        program = teleprompter.compile(
            student=EvidenceModule(),
            trainset=trainset,
            devset=devset,
            eval_kwargs=eval_kwargs,
        )

    elif optimizer_name == "miprov2":
        # MIPROv2 supports a seed both in its constructor and in its compile() call.
        teleprompter = MIPROv2(
            metric=metric_fn,
            num_candidates=8,
            init_temperature=2.0,
            num_threads=8,
            max_bootstrapped_demos=10,
            max_labeled_demos=6,
            seed=seed,  # set seed at construction
        )
        program = teleprompter.compile(
            student=EvidenceModule(),
            trainset=trainset,
            valset=devset,
            seed=seed,  # override seed for this compile run
            minibatch=False,
            requires_permission_to_run=False,
        )

    elif optimizer_name == "simba":
        teleprompter = SIMBA(
            metric=metric_fn,
            bsize=5,
            max_iters=50,
            num_trials=3,
            init_temperature=0.8,
            # SIMBA may or may not accept a seed; if it does, you can pass seed=seed here
        )
        program = teleprompter.compile(
            student=EvidenceModule(),
            trainset=trainset,
            valset=devset,
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Choose from 'bootstrap', 'copro', 'miprov2', 'simba'.")

    return program, teleprompter
