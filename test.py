def compile_with_compat(teleprompter, student, trainset, devset):
    sig = inspect.signature(teleprompter.compile)
    params = set(sig.parameters.keys())

    # Safe minibatch size for small dev/val sets
    safe_minibatch = min(len(devset), 1) or 1

    if "valset" in params:
        return teleprompter.compile(
            student=student,
            trainset=trainset,
            valset=devset,
            minibatch_size=safe_minibatch
        )
    if "devset" in params:
        return teleprompter.compile(
            student=student,
            trainset=trainset,
            devset=devset,
            minibatch_size=safe_minibatch
        )
    if "validset" in params:
        return teleprompter.compile(
            student=student,
            trainset=trainset,
            validset=devset,
            minibatch_size=safe_minibatch
        )
    return teleprompter.compile(student=student, trainset=trainset)
