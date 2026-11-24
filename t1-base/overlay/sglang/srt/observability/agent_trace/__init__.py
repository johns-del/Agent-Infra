"""Optional native observation hooks. Importing this module does not load torch."""
from __future__ import annotations
from urllib.parse import unquote
from .runtime import enabled, get_writer


def api_link(obj, raw_request=None, parent_rid=None):
    """Observe *already assigned* RIDs; never replace RID or alter IPC payloads."""
    try:
        w=get_writer("api")
        if w is None:return
        headers=getattr(raw_request,"headers",{}) if raw_request is not None else {}
        fields={dst:unquote(str(headers[src])) for src,dst in (
            ("x-orvix-user-id","client_user_id"),
            ("x-orvix-session-id","session_id"),
            ("x-orvix-tool-use-id","tool_use_id"),
            ("x-orvix-attempt-id","client_attempt_id"),
            ("x-trace-purpose","purpose"),
            ("x-trace-origin-operation-id","origin_operation_id"),
        ) if src in headers}
        ids={dst:str(headers[src]) for src,dst in (
            ("x-request-id","attempt_id"),("x-trace-forward-id","forward_id"),
            ("x-trace-gateway-id","gw_id"),("x-trace-parent-forward-id","parent_forward_id"),
        ) if src in headers}
        rids=obj.rid if isinstance(obj.rid,list) else [obj.rid]
        for rid in rids:
            from .admission import link
            link(rid, {**fields, **ids})
            w.emit("engine.request.link",{**fields,"parent_engine_rid":parent_rid,
                   "attribution":"client_header_not_authorization",
                   "rid_origin":"native_normalize_batch_and_arguments",
                   "transport_mapping":"explicit" if ids or parent_rid else "unavailable"},rid,**ids)
    except Exception:
        if "w" in locals() and w is not None:w.callback_errors+=1


def api_dispatch(obj):
    try:
        from .admission import dispatch
        dispatch(obj)
        w=get_writer("api")
        if w:w.emit("engine.request.dispatched",{"boundary":"before_native_ipc_dispatch"},obj.rid)
    except Exception:
        if "w" in locals() and w is not None:w.callback_errors+=1


def install(scheduler, *, rank=None):
    if not enabled(): return None
    try:
        from .hooks import EngineObserver
        observer=EngineObserver(scheduler,rank=rank)
        observer.attach()
        scheduler._orvix_observer=observer
        return observer
    except Exception:
        # Trace setup must never prevent engine startup.
        w=get_writer("scheduler")
        if w:w.emit("engine.trace.unsupported",{"reason":"observer_installation_failed"})
        return None
