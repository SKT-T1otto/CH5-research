"""Extend the accepted audit state projection to the executor's post-handoff cache."""
from copy import copy

from tools.command_control_trace import fingerprint as native_fingerprint


def fingerprint(runtime):
    # The D2 intercept key contains exact position/velocity bytes after handoff.
    # Build an observation-only view. The live planner retains its native cache key.
    if all(local.intercept_key is None for local in runtime.locals):
        return native_fingerprint(runtime)
    view = copy(runtime)
    view.locals = []
    for local in runtime.locals:
        snapshot = copy(local)
        if local.intercept_key is not None:
            snapshot.intercept_key = tuple({'audit_bytes_hex': v.hex()} if isinstance(v, bytes) else v
                                           for v in local.intercept_key)
        view.locals.append(snapshot)
    return native_fingerprint(view)
