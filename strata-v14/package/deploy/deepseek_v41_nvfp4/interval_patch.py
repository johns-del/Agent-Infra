"""v7 hooks after v5 no-op hooks; no MultiPrefill early-result fence."""
import ast
MARKER='# ORVIX_INTERVAL_CACHE_V7'

def patch_scheduler(text):
    if '# ORVIX_MULTI_PREFILL_V6' in text:
        raise ValueError('v7 uses native single-partial substrate; derive from original/v5, not patched v6')
    edits=[
      ('        self.init_schedule_policy()\n','        self.init_schedule_policy()\n        # ORVIX_INTERVAL_CACHE_V7\n        self._orvix_interval_patch_level = 7\n'),
      ('            chunk_control.on_launch(batch)\n','            chunk_control.on_launch(batch)\n        # ORVIX_INTERVAL_CACHE_V7\n        interval = getattr(self, "_orvix_interval_cache", None)\n        if interval is not None:\n            interval.on_launch(batch)\n'),
      ('            chunk_control.on_result(batch, result)\n','            chunk_control.on_result(batch, result)\n        # ORVIX_INTERVAL_CACHE_V7\n        interval = getattr(self, "_orvix_interval_cache", None)\n        if interval is not None:\n            interval.on_result(batch, result)\n'),
      ('        for req in self.waiting_queue:\n            # A spent pass budget',
       '        for req in self.waiting_queue:\n            # ORVIX_INTERVAL_CACHE_V7\n            interval = getattr(self, "_orvix_interval_cache", None)\n            if interval is not None and not interval.can_admit(req):\n                continue\n            # A spent pass budget'),
    ]
    if MARKER in text:
        if text.count(MARKER)!=len(edits) or any(text.count(b)!=1 for _,b in edits): raise ValueError('partial v7 hooks')
        return text
    for old,new in edits:
        if text.count(old)!=1: raise ValueError('v7 hook mismatch: '+old.strip())
        text=text.replace(old,new)
    ast.parse(text)
    return text
