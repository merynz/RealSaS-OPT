"""Forward a process-owned log live without pipe deadlocks or changing execution."""
import codecs
import json
import subprocess
import time


def follow_process_log(process, log_path, *, context, heartbeat_seconds=30.,
                       maximum_seconds=None, poll_seconds=.25):
    if heartbeat_seconds<=0 or poll_seconds<=0:
        raise ValueError('PROCESS_LOG_POSITIVE_INTERVAL_REQUIRED')
    started=time.monotonic(); next_heartbeat=started+heartbeat_seconds
    offset=0; decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')
    pending=''; stage=context.get('stage','STARTING')

    def drain(final=False):
        nonlocal offset,pending,stage
        with log_path.open('rb') as stream:
            stream.seek(offset)
            blocks=0
            while chunk:=stream.read(64<<10):
                offset+=len(chunk); value=decoder.decode(chunk)
                print(value,end='',flush=True); pending+=value
                while '\n' in pending:
                    line,pending=pending.split('\n',1)
                    try: row=json.loads(line)
                    except (ValueError,TypeError): continue
                    if isinstance(row,dict) and row.get('event')=='WORKER_STAGE': stage=row.get('stage',stage)
                # A very long non-JSON line must not grow monitor memory.
                if len(pending)>65536: pending=pending[-65536:]
                blocks+=1
                if not final and blocks>=16: break
        if final: print(decoder.decode(b'',final=True),end='',flush=True)

    while process.poll() is None:
        drain()
        now=time.monotonic()
        if maximum_seconds is not None and now-started>=maximum_seconds:
            raise TimeoutError('PROCESS_LOG_TIME_BUDGET_EXCEEDED')
        if now>=next_heartbeat:
            print(json.dumps({**context,'stage':stage,'elapsed_seconds':round(now-started,3)},sort_keys=True),flush=True)
            next_heartbeat=now+heartbeat_seconds
        try: process.wait(timeout=poll_seconds)
        except subprocess.TimeoutExpired: pass
    drain(final=True)
    return process.returncode
