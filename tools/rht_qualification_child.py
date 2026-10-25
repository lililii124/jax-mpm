"""Isolated frozen-source runner; timing trace observes original executable lines."""
import builtins
import json
from pathlib import Path
import runpy
import sys
import time

source=Path(sys.argv[1]).resolve()
timing=Path(sys.argv[2]).resolve()
sys.path.insert(0,str(source))
sys.argv=['examples.rht_pointed_impact',*sys.argv[3:]]
start=time.perf_counter()
common=str(source/'examples/common.py')
last=None
first_run=True
function_starts={}
import_depth=0
import_seconds=0.0
saved_import=builtins.__import__
def timed_import(*args,**kwargs):
    global import_depth, import_seconds
    outer=import_depth==0
    before=time.perf_counter()
    import_depth+=1
    try:
        return saved_import(*args,**kwargs)
    finally:
        import_depth-=1
        if outer:
            import_seconds+=time.perf_counter()-before
builtins.__import__=timed_import
# Each duration is from a source line until the next line event. Batch+blocking
# spans are reported separately, so asynchronous dispatch is never called compute.
with timing.open('x',buffering=1) as stream:
    def trace(frame,event,arg):
        global last, first_run
        if frame.f_code.co_filename==common and frame.f_code.co_name=='run' and event in ('line','return'):
            now=time.perf_counter()
            if first_run:
                stream.write(json.dumps({'kind':'imports_and_case_construction','seconds':now-start})+'\n')
                first_run=False
            if last is not None:
                stream.write(json.dumps({'kind':'common.run.line','line':last[0],
                    'seconds':now-last[1],'next_line':frame.f_lineno,'event':event})+'\n')
            last=(frame.f_lineno,now)
        return trace if frame.f_code.co_filename==common else None
    def profile(frame,event,arg):
        if frame.f_code.co_name in ('make_case','_load_state'):
            if event=='call':
                function_starts[id(frame)]=time.perf_counter()
            elif event=='return' and id(frame) in function_starts:
                stream.write(json.dumps({'kind':frame.f_code.co_name,'seconds':time.perf_counter()-function_starts.pop(id(frame))})+'\n')
    sys.setprofile(profile)
    sys.settrace(trace)
    try:
        before=time.perf_counter()
        # Loading the module executes original imports and the exact original CLI.
        # First run event minus start reports imports+case construction separately.
        runpy.run_module('examples.rht_pointed_impact',run_name='__main__')
    finally:
        sys.settrace(None)
        sys.setprofile(None)
        builtins.__import__=saved_import
        stream.write(json.dumps({'kind':'outer_import_calls','seconds':import_seconds})+'\n')
        stream.write(json.dumps({'kind':'process_total','seconds':time.perf_counter()-start})+'\n')
