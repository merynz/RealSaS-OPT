import json
import subprocess
import sys
import pytest
from compiler.realsas_compiler_services.process_log_v1 import follow_process_log


def test_stdout_stderr_and_stage_heartbeat_are_live_and_log_is_preserved(tmp_path,capsys,monkeypatch):
    path=tmp_path/'worker.log'
    program='''import json,sys,time
print('output',flush=True)
print('diagnostic',file=sys.stderr,flush=True)
print(json.dumps({'event':'WORKER_STAGE','stage':'49_SUBJECT_TRUTH_FAMILIES'}),flush=True)
time.sleep(.15)
print('finished',flush=True)
'''
    with path.open('w') as out:
        p=subprocess.Popen([sys.executable,'-u','-c',program],stdout=out,stderr=subprocess.STDOUT)
        import builtins
        original_print=builtins.print; live=[]
        def observe(*args,**kwargs):
            if args and 'output' in str(args[0]): live.append(p.poll() is None)
            return original_print(*args,**kwargs)
        monkeypatch.setattr(builtins,'print',observe)
        code=follow_process_log(p,path,context={'event':'COMPUTE_HEARTBEAT','subject':'s','subject_index':2,'subject_count':3},
                                heartbeat_seconds=.03,poll_seconds=.01)
    assert code==0
    assert any(live)
    console=capsys.readouterr().out
    for text in ['output','diagnostic','finished']: assert text in console and text in path.read_text()
    beats=[json.loads(line) for line in console.splitlines() if line.startswith('{') and 'COMPUTE_HEARTBEAT' in line]
    assert beats and any(b['stage']=='49_SUBJECT_TRUTH_FAMILIES' for b in beats)
    assert all(b['subject_index']==2 and b['subject_count']==3 and b['elapsed_seconds']>0 for b in beats)


def test_monitor_keeps_timeout_and_nonzero_exit_visible(tmp_path):
    path=tmp_path/'worker.log'
    with path.open('w') as out:
        p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)'],stdout=out,stderr=subprocess.STDOUT)
        try:
            with pytest.raises(TimeoutError):
                follow_process_log(p,path,context={},maximum_seconds=.03,poll_seconds=.01)
        finally: p.kill();p.wait()
    with path.open('w') as out:
        p=subprocess.Popen([sys.executable,'-c','raise SystemExit(7)'],stdout=out,stderr=subprocess.STDOUT)
        assert follow_process_log(p,path,context={},poll_seconds=.01)==7
