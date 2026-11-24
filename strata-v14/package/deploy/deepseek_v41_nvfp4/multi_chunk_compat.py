"""Frozen per-batch partial membership, shared by native result/token paths."""
from __future__ import annotations


def partials(batch):
    if batch is None: return ()
    rows=getattr(batch,'orvix_partial_reqs',None)
    if rows is not None: return rows
    r=getattr(batch,'chunked_req',None)
    return () if r is None else (r,)


def is_partial(batch,req):
    return any(r is req for r in partials(batch))


def streamable_reqs(batch):
    return [r for r in batch.reqs if not is_partial(batch,r)]


def rewrite_partial_tail_tokens(batch,tokens):
    pairs=getattr(batch,'orvix_next_prompt_tokens',None)
    if pairs is None: return tokens
    updates=[(i,token) for i,r in enumerate(batch.reqs) for req,token in pairs if req is r]
    if not updates: return tokens
    result=tokens.clone()
    for i,token in updates:
        result[i:i+1].fill_(token)
    return result


def annotate(batch,reqs,vocab_size):
    """Snapshot BEFORE next-round request mutations; values survive batch.copy."""
    middle=tuple(r for r in reqs if r.extend_end<len(r.full_untruncated_fill_ids))
    pairs=[]
    for r in middle:
        ids=r.full_untruncated_fill_ids
        token=int(ids[r.extend_end])
        if not 0<=token<vocab_size: raise RuntimeError('invalid next prompt token')
        pairs.append((r,token))
    batch.orvix_partial_reqs=middle
    batch.orvix_next_prompt_tokens=tuple(pairs)
    batch.chunked_req=None
    batch.chunked_req_next_prompt_token=None
    batch.contains_last_prefill_chunk=len(middle)<len(reqs)
