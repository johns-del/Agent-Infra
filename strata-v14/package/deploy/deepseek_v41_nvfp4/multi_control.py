"""Inspect or activate v6 on a private loopback engine.

Changes require a drained engine. Use ONE clock owner (this OR the ablation
runner); a heartbeat changes only the broadcast logical clock, not policy.
"""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import time

import httpx
from chunk_control import private_endpoint
from multi_chunk_core import MultiConfig
from compare_dev_replay import engine_state, verify_configuration, KEY
from run_unified_campaign import DEFAULT


async def update(client, payload):
    response = await client.post('/set_internal_state', json={'server_args': {}, KEY: payload})
    response.raise_for_status()
    if response.json() != [True]:
        raise RuntimeError('engine rejected v6 control; drain requests and inspect compatibility')


async def run(args):
    headers = {}
    if key := os.environ.get('ORVIX_ENGINE_API_KEY'):
        headers['Authorization'] = 'Bearer ' + key
    async with httpx.AsyncClient(base_url=args.endpoint, headers=headers, trust_env=False, timeout=30.) as client:
        _, state = await engine_state(client)
        if state.get('adapter_version') != 'orvix-dsv41-multi-prefill-v6':
            raise RuntimeError('v6 private runtime required; do not activate on v5')
        identity = state.get('runtime_identity', {})
        if not identity.get('verified'):
            raise RuntimeError('unverified runtime; use v6 prepare-runtime')
        start = time.monotonic()
        base = max(start, float(state.get('clock', state.get('now', 0.))))
        if args.apply:
            config = MultiConfig(**json.loads(args.apply.read_text())).frozen()
            await update(client, {'now': base, 'enabled': False, 'parameters': state['parameters'],
                                  'knobs': DEFAULT, 'cache': {'mode': 'observe'},
                                  'prefill': {'mode': 'off'}, 'chunk_fair': {'mode': 'off'},
                                  'multi_chunk': config})
            _, state = await engine_state(client)
            verify_configuration(state, False, state['parameters'], DEFAULT,
                                 {'mode': 'observe'}, {'mode': 'off'}, {'mode': 'off'}, config)
        print(json.dumps({'adapter_version': state['adapter_version'],
                          'runtime_fingerprint': identity['python_fingerprint'],
                          'multi_chunk': state.get('multi_chunk')}, ensure_ascii=False, indent=2), flush=True)
        if args.heartbeat:
            if state.get('multi_chunk', {}).get('config', {}).get('mode', 'off') == 'off':
                raise ValueError('activate v6 before starting its clock')
            while args.seconds is None or time.monotonic()-start < args.seconds:
                await asyncio.sleep(args.interval)
                await update(client, {'now': base+time.monotonic()-start, 'enabled': False,
                                      'parameters': state['parameters'], 'knobs': DEFAULT})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', type=private_endpoint, default='http://127.0.0.1:32781')
    parser.add_argument('--apply', type=Path, help='policy JSON; omit for read-only inspection')
    parser.add_argument('--heartbeat', action='store_true')
    parser.add_argument('--interval', type=float, default=.25)
    parser.add_argument('--seconds', type=float)
    args = parser.parse_args()
    if not math.isfinite(args.interval) or not .05 <= args.interval <= 10:
        parser.error('interval must be finite and in [0.05,10] seconds')
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds<=0):
        parser.error('seconds must be finite and positive')
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print('Clock stopped. Policy remains active; stop/drain requests before deactivation.')

if __name__ == '__main__':
    main()
