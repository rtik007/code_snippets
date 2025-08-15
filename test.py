import traceback, sys

try:
    teleprompter.compile(
        student=evidenceModule(),
        trainset=trainset,
        valset=devset,
        num_candidates=8,
        init_temperature=1.8,
        # ...
        requires_permission_to_run=False,
        verbose=True,
    )
except TypeError as e:
    print("TYPEERROR:", e)
    print("--- likely None detected. Quick dump ---")
    print("trainset is None?", trainset is None, "len:", len(trainset) if trainset else 0)
    print("devset is None?", devset is None, "len:", len(devset) if devset else 0)
    # If you can import the proposer/module, also print any cached candidates here
    traceback.print_exc(file=sys.stdout)
    raise
