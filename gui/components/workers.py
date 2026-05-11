import inspect

from PySide6.QtCore import QThread, Signal


class StreamingWorker(QThread):
    on_item = Signal(object)
    error = Signal(str)

    def __init__(self, generator_fn, *args, **kwargs):
        super().__init__()
        self._gen_fn = generator_fn
        self._args = args
        self._kwargs = kwargs
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            for item in self._gen_fn(*self._args, **self._kwargs):
                if not self._running:
                    break
                self.on_item.emit(item)
        except Exception as e:
            self.error.emit(str(e))


class TaskWorker(QThread):
    progress_text = Signal(str)
    result = Signal(object)
    error = Signal(str)

    def __init__(self, task_fn, *args, **kwargs):
        super().__init__()
        self._task_fn = task_fn
        self._args = args
        self._kwargs = kwargs
        self._running = True

    def stop(self):
        self._running = False

    def _build_kwargs(self):
        kwargs = dict(self._kwargs)
        try:
            params = inspect.signature(self._task_fn).parameters
        except (TypeError, ValueError):
            params = {}
        if "progress_cb" in params and "progress_cb" not in kwargs:
            kwargs["progress_cb"] = self.progress_text.emit
        if "should_run" in params and "should_run" not in kwargs:
            kwargs["should_run"] = lambda: self._running
        return kwargs

    def run(self):
        try:
            value = self._task_fn(*self._args, **self._build_kwargs())
            if self._running:
                self.result.emit(value)
        except Exception as e:
            self.error.emit(str(e))
