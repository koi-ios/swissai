"""Names shared by the zygote parent (zygote.py) and the forkserver side (zygote_preload.py)."""
CONFIG_ENV = "PYSTARTUP_ZYGOTE_CONFIG"   # path of the JSON config the forkserver reads
CHILD_ENV = "PYSTARTUP_ZYGOTE_CHILD"     # set in every worker forked from the zygote (value: zygote pid)
ENV_KEY = "pystartup_environ"            # key of the parent's environment in the preparation data


def in_forkserver_process(orig_argv) -> bool:
    """True only in the interpreter multiprocessing started as its fork server."""
    return any("multiprocessing.forkserver import main" in a for a in (orig_argv or []))
