from PySide6.QtCore import QThread, Signal


class StreamingWorker(QThread):
    on_item = Signal(object)
    finished = Signal()
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
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
