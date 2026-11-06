"""Explicit phase routing and bounded, opt-in in-process task scheduling.

Cancellation is cooperative: an in-flight request may complete or be billed;
its result is not published after cancellation. No implicit provider fallback.
"""
from __future__ import annotations
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
import threading
import time
STAGES={'analysis','chunk','synthesis','follow_up','verification'}
_CANCEL=ContextVar('astrograph_cancel',default=None)
_MODEL=ContextVar('astrograph_phase_model',default=None)
class AnalysisCancelled(RuntimeError):
    def __init__(self):super().__init__('ANALYSIS_CANCELLED')
def check_cancelled():
    event=_CANCEL.get()
    if event is not None and event.is_set():raise AnalysisCancelled()
@contextmanager
def cancellation_scope(event):
    token=_CANCEL.set(event)
    try:yield
    finally:_CANCEL.reset(token)
def phase_model():return _MODEL.get()
class StageRouter:
    def __init__(self,backends):
        if 'analysis' not in backends or set(backends)-STAGES:raise ValueError('INVALID_PHASE_MODELS')
        self.backends=dict(backends);self.model_id=getattr(backends['analysis'],'model_id','')
    def has_stage(self,stage):
        return stage in self.backends
    def for_stage(self,stage):
        if stage not in STAGES:raise ValueError('INVALID_PHASE')
        return _SelectedBackend(self.backends.get(stage,self.backends['analysis']))
    def generate_structured(self,**kwargs):return self.for_stage('analysis').generate_structured(**kwargs)
class _SelectedBackend:
    def __init__(self,backend):self.backend=backend;self.model_id=getattr(backend,'model_id','')
    def generate_structured(self,**kwargs):
        check_cancelled();token=_MODEL.set(self.model_id)
        try:
            value=self.backend.generate_structured(**kwargs);check_cancelled();return value
        finally:_MODEL.reset(token)
def for_stage(backend,stage):
    if stage not in STAGES:raise ValueError('INVALID_PHASE')
    return backend.for_stage(stage) if isinstance(backend,StageRouter) else _SelectedBackend(backend)

class BoundedExecutor:
    def __init__(self,max_waiting=8):
        if type(max_waiting)is not int or max_waiting<0:raise ValueError('INVALID_QUEUE_CAPACITY')
        self.max_waiting=max_waiting;self.lock=threading.RLock();self.condition=threading.Condition(self.lock)
        self.pending=deque();self.items={};self.running=None;self.thread=None
    def busy(self):
        with self.lock:return bool(self.items)
    @contextmanager
    def idle_guard(self):
        with self.lock:
            if self.items:raise ValueError('BUSY')
            yield
    def submit(self,key,callback, *,persist,allow_queue=False):
        with self.lock:
            if key in self.items:raise ValueError('TASK_ALREADY_ACTIVE')
            if self.items and not allow_queue:raise ValueError('BUSY')
            if len(self.items)>=self.max_waiting+1:raise ValueError('QUEUE_FULL')
            state='QUEUED' if self.items else 'RUNNING'
            persist(key,state) # durable admission before callbacks/credentials enter queue
            self.items[key]={'callback':callback,'persist':persist,'token':threading.Event(),'published':False}
            self.pending.append(key)
            if self.thread is None:
                self.thread=threading.Thread(target=self._drive,daemon=True,name='astrograph-user-queue')
                try:self.thread.start()
                except BaseException:
                    self.thread=None;self.pending.remove(key);self.items.pop(key,None);persist(key,'INTERRUPTED');raise
            return state
    def cancel(self,key):
        with self.lock:
            item=self.items.get(key)
            if item is None or item['published']:raise ValueError('TASK_NOT_CANCELLABLE')
            status='CANCEL_REQUESTED' if self.running==key else 'CANCELLED'
            item['persist'](key,status);item['token'].set()
            if status=='CANCELLED':self.pending.remove(key);self.items.pop(key);self.condition.notify_all()
            return status
    @contextmanager
    def publication(self,key):
        with self.lock:
            item=self.items.get(key)
            if item is None or item['token'].is_set():raise AnalysisCancelled()
            yield
            item['published']=True
    def _drive(self):
        while True:
            with self.lock:
                if not self.pending:
                    self.thread=None;self.condition.notify_all();return
                key=self.pending.popleft();item=self.items[key];self.running=key
            try:
                with self.lock:
                    if item['token'].is_set():raise AnalysisCancelled()
                    item['persist'](key,'RUNNING')
                with cancellation_scope(item['token']):
                    check_cancelled();item['callback'](item['token'])
            except AnalysisCancelled:
                try:item['persist'](key,'CANCELLED')
                except Exception:pass # storage failure cannot cause request replay
            except BaseException:
                try:item['persist'](key,'FAILED')
                except Exception:pass
            finally:
                with self.lock:
                    self.items.pop(key,None);self.running=None;self.condition.notify_all()
    def wait_idle(self,timeout=30):
        end=time.monotonic()+timeout
        with self.condition:
            while self.items:
                remaining=end-time.monotonic()
                if remaining<=0:return False
                self.condition.wait(remaining)
            return True
