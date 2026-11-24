"""Reviewable v6 source transformations; no runtime exec of rewritten methods.

The installer requires exact pinned blobs for lifecycle files absent from the
uploaded excerpts. No fuzzy patching, no downloading code during deployment.
"""
from __future__ import annotations
import ast
import hashlib
import textwrap

MARKER='# ORVIX_MULTI_PARTIAL_V6'
PINNED_COMMIT='858073e4f6d246bd6e3baae1d4e07919a578a04f'
BLOBS={
 'sglang/srt/managers/schedule_batch.py':'79f3112ddae99154cd7c47118d485541e5b67b00',
 'sglang/srt/managers/scheduler_components/batch_result_processor.py':'fba0075a96e6c873564034fd59976b8d8bc5ddbe',
 'sglang/srt/model_executor/model_runner_components/ngram_embedding_manager.py':'c525bcc35a932d4de1890bb1cd797d030c5d37cf',
 'sglang/srt/speculative/eagle_utils.py':'c857c78b78085eaf4c7872b3d0647940b70df0cf',
 'sglang/srt/managers/scheduler_components/invariant_checker.py':'7abb28053b1190889d1329872dd2b62ba0efec58',
}


def blob_hash(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def replace_once(s,old,new):
    if s.count(old)!=1:raise ValueError('unqualified v6 anchor: '+old[:110])
    return s.replace(old,new,1)


def add_import(s):
    if MARKER in s:
        raise ValueError("v6 hooks already present; derive from a clean or v5 runtime")
    header = MARKER + "\nfrom sglang.srt.observability.multi_chunk_compat import partials as orvix_partials, is_partial as orvix_is_partial, streamable_reqs as orvix_streamable_reqs, rewrite_partial_tail_tokens as orvix_rewrite_partial_tail_tokens\n"
    module = ast.parse(s)
    imports = [n for n in module.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if not imports:
        raise ValueError("native module has no import anchor")
    futures = [n for n in imports if isinstance(n, ast.ImportFrom) and n.module == '__future__']
    line = max(n.end_lineno for n in futures) if futures else imports[0].lineno - 1
    lines = s.splitlines(keepends=True)
    lines.insert(line, header)
    return ''.join(lines)


def scoped(s,classname,method,transform):
    cls=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name==classname)
    fn=next(n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==method)
    lines=s.splitlines(keepends=True); old=''.join(lines[fn.lineno-1:fn.end_lineno])
    new=transform(old)
    lines[fn.lineno-1:fn.end_lineno]=[new]
    result=''.join(lines);ast.parse(result);return result


def patch_scheduler(s):
    if MARKER in s:raise ValueError('derive v6 from unmodified/v5 source, not another v6 tree')
    s=add_import(s)
    s=replace_once(s,'        self._orvix_chunk_patch_level = 5\n',
                   '        self._orvix_chunk_patch_level = 5\n        self._orvix_multi_patch_level = 6\n')
    def next_batch(f):
        a=f.index('        if self.chunked_req is not None:\n')
        b=f.index('        # HiSparse has its own',a)
        old=f[a:b]
        new='''        multi = getattr(self, "_orvix_multi_control", None)
        if multi is not None and multi.active:
            # Each completed partial was checkpointed after its result fence.
            reqs_to_exclude.update(multi.partial_requests())
        else:
'''+textwrap.indent(old,'    ')
        f=f[:a]+new+f[b:]
        a=f.index('            if last_batch.chunked_req is not None:')
        b=f.index('\n            if self.dllm_config',a)
        f=f[:a]+'''            # Frozen batch membership, not the current registry or scalar slot.
            reqs_to_exclude.update(orvix_partials(last_batch))
'''+f[b:]
        return f
    s=scoped(s,'Scheduler','get_next_batch_to_run',next_batch)
    def raw(f):
        anchor='        if self.enable_priority_preemption or self.is_hybrid_swa:\n'
        f=replace_once(f,anchor,'''        multi = getattr(self, "_orvix_multi_control", None)
        if multi is not None and multi.active:
            multi.before_admission(running_batch)
'''+anchor)
        a=f.index('        if self.chunked_req is not None:\n            self.chunked_req.init_next_round_input()')
        b=f.index('        self.running_reqs.leave(adder.preempt_list)',a)
        old=f[a:b]
        f=f[:a]+'''        if multi is not None and multi.active:
            multi.admit_pass(adder, running_batch)
        else:
'''+textwrap.indent(old,'    ')+f[b:]
        old='''        if adder.new_chunked_req is not None:
            # Update chunked prefill
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req
'''
        f=replace_once(f,old,'''        if multi is not None and multi.active:
            multi.commit_admissions(adder)
        else:
'''+textwrap.indent(old,'    '))
        f=replace_once(f,'            chunked_req=self.chunked_req,\n',
                        '            chunked_req=None if multi is not None and multi.active else self.chunked_req,\n')
        old='''        # A parked chunked request runs no chunk in this batch.
        new_batch.contains_last_prefill_chunk = len(can_run_list) != int(
            self.chunked_req in can_run_set
        )
'''
        f=replace_once(f,old,'''        if multi is not None and multi.active:
            multi.annotate_batch(new_batch)
        else:
'''+textwrap.indent(old,'    '))
        f=replace_once(f,'                    self.chunked_req.extend_len if self.chunked_req is not None else 0\n',
                       '                    sum(r.extend_len for r in orvix_partials(new_batch))\n')
        return f
    s=scoped(s,'Scheduler','_get_new_batch_prefill_raw',raw)
    s=replace_once(s,'                if r is not batch.chunked_req and r not in self.running_reqs\n',
                   '                if not orvix_is_partial(batch, r) and r not in self.running_reqs\n')
    s=replace_once(s,'            chunk_control.on_launch(batch)\n',
                   '            chunk_control.on_launch(batch)\n        multi = getattr(self, "_orvix_multi_control", None)\n        if multi is not None:\n            multi.on_launch(batch)\n')
    s=replace_once(s,'            chunk_control.on_result(batch, result)\n',
                   '            chunk_control.on_result(batch, result)\n        multi = getattr(self, "_orvix_multi_control", None)\n        if multi is not None:\n            multi.on_result(batch, result)\n')
    def abort(f):
        a=f.index('        if (chunked_req := self.chunked_req) is not None:')
        f=f[:a]+'''        multi = getattr(self, "_orvix_multi_control", None)
        if multi is not None and multi.active:
            multi.mark_abort(recv_req)
            return
'''+f[a:]
        return f
    s=scoped(s,'Scheduler','_abort_chunked',abort)
    def process_abort(f):
        return replace_once(f,'        req = self._pending_chunked_abort_req\n','''        multi = getattr(self, "_orvix_multi_control", None)
        if multi is not None and multi.active:
            multi.process_aborts()
            return
        req = self._pending_chunked_abort_req
''')
    s=scoped(s,'Scheduler','process_pending_chunked_abort',process_abort)
    def pause(f):
        anchor='        if self.disaggregation_mode == DisaggregationMode.DECODE:\n'
        # Inject all parked states before native retract_all; native call still
        # owns lock/storage release, abort handling and requeue semantics.
        return replace_once(f,anchor,'''        multi = getattr(self, "_orvix_multi_control", None)
        if multi is not None and multi.active:
            retract_reqs += [r for r in multi.partial_requests() if not r.finished()]
            retract_reqs = list({id(r): r for r in retract_reqs}.values())
'''+anchor)
    s=scoped(s,'Scheduler','pause_generation',pause)
    # Scalar writes with a request would drop the other partials. Catch any
    # unconverted path rather than silently corrupting the state registry.
    old='''        self.chunked_reqs.clear()
        if req is not None:
            self.chunked_reqs.join([req])'''
    new='''        multi = getattr(self, "_orvix_multi_control", None)
        if req is not None and multi is not None and multi.active:
            raise RuntimeError("v6 forbids assigning a single chunked_req")
        self.chunked_reqs.clear()
        if req is not None:
            self.chunked_reqs.join([req])'''
    s=replace_once(s,old,new)
    ast.parse(s);return s


def patch_batch(s):
    s=add_import(s)
    s=replace_once(s,'    chunked_req_next_prompt_token: Optional[int] = None\n',
        '    chunked_req_next_prompt_token: Optional[int] = None\n'
        '    orvix_partial_reqs: Optional[Tuple[Req, ...]] = None\n'
        '    orvix_next_prompt_tokens: Optional[Tuple[Tuple[Req, int], ...]] = None\n')
    s=scoped(s,'ScheduleBatch','copy',lambda f: replace_once(f,'            chunked_req=self.chunked_req,\n',
        '            chunked_req=self.chunked_req,\n            orvix_partial_reqs=self.orvix_partial_reqs,\n            orvix_next_prompt_tokens=self.orvix_next_prompt_tokens,\n'))
    # Old extend batch may become the running batch after partials are filtered.
    s=scoped(s,'ScheduleBatch','prepare_for_decode',lambda f:replace_once(f,'        self.forward_mode = ForwardMode.DECODE\n',
        '        self.forward_mode = ForwardMode.DECODE\n        self.orvix_partial_reqs = None\n        self.orvix_next_prompt_tokens = None\n'))
    ast.parse(s);return s


def patch_result(s):
    s=add_import(s)
    if 'is_middle_chunk = req is batch.chunked_req' not in s:raise ValueError('unknown result processor')
    s=s.replace('req is not batch.chunked_req','not orvix_is_partial(batch, req)')
    s=s.replace('req is batch.chunked_req','orvix_is_partial(batch, req)')
    s=replace_once(s,'            batch.reqs, batch.return_logprob, skip_stream_req\n',
                   '            orvix_streamable_reqs(batch), batch.return_logprob, skip_stream_req\n')
    ast.parse(s);return s


def patch_ngram(s):
    s=add_import(s)
    s=replace_once(s,'            if chunked_req is not None:\n                skip_token_table_update = [req is chunked_req for req in batch.reqs]\n',
        '            if chunked_req is not None or batch.orvix_partial_reqs is not None:\n'
        '                skip_token_table_update = [orvix_is_partial(batch, req) for req in batch.reqs]\n')
    ast.parse(s);return s


def patch_eagle(s):
    s=add_import(s)
    s=replace_once(s,'    next_prompt_token = batch.chunked_req_next_prompt_token\n',
        '    if batch.orvix_next_prompt_tokens is not None:\n        return orvix_rewrite_partial_tail_tokens(batch, tail_tokens)\n'
        '    next_prompt_token = batch.chunked_req_next_prompt_token\n')
    ast.parse(s);return s


def patch_invariants(s):
    s=add_import(s)
    def f(text):
        text=replace_once(text,'        chunked_req = self.get_chunked_req()\n',
                         '        chunked_reqs = set(lists.get(ListKind.CHUNKED, ()))\n')
        return replace_once(text,'if not req.finished() and req is not chunked_req',
                            'if not req.finished() and req not in chunked_reqs')
    s=scoped(s,'SchedulerInvariantChecker','maybe_check_request_lists',f)
    ast.parse(s);return s

TRANSFORMS={
 'sglang/srt/managers/schedule_batch.py':patch_batch,
 'sglang/srt/managers/scheduler_components/batch_result_processor.py':patch_result,
 'sglang/srt/model_executor/model_runner_components/ngram_embedding_manager.py':patch_ngram,
 'sglang/srt/speculative/eagle_utils.py':patch_eagle,
 'sglang/srt/managers/scheduler_components/invariant_checker.py':patch_invariants,
}
