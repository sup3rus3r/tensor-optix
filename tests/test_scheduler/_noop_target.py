"""A standalone module with zero heavy imports, so spawning a process
targeting its function doesn't pay tensor_optix's/TF's import cost -- used
only to test process shutdown/exitcode mechanics in isolation from
training-stack import overhead."""


def noop_process_target():
    pass
