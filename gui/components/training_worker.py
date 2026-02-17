from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QThread, Signal


class TrainingWorker(QThread):
    """Univerzální worker pro trénování modelu.
    Zachovává signály z původního řešení v Tab 2:
      - progress(idx, total, params, mean_f1, std_f1)
      - phase(str)
      - finished(path_to_model)
      - error(str)
    Použití: předat `train_fn`, který *uvnitř* volá poskytnuté callbacky `progress_cb` a `phase_cb`.
    """
    progress = Signal(int, int, dict, float, float)  # idx, total, params, mean_f1, std_f1
    phase = Signal(str)
    finished = Signal(str)  # path to saved/best model
    error = Signal(str)

    def __init__(self, train_fn: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self._train_fn = train_fn
        self._args = args
        self._kwargs = kwargs
        self._running = True

    # Volitelně může GUI požadovat stop; train_fn by měl pravidelně kontrolovat tento stav.
    def stop(self):
        self._running = False

    # Tyto callbacky předáme do train_fn, aby mohl reportovat stav zpět do GUI.
    def _progress_cb(self, idx: int, total: int, params: dict, mean_f1: float, std_f1: float):
        self.progress.emit(idx, total, params, mean_f1, std_f1)

    def _phase_cb(self, phase: str):
        self.phase.emit(phase)

    def run(self):
        try:
            # train_fn má za úkol vrátit cestu k uloženému (nejlepšímu) modelu
            result = self._train_fn(
                *self._args,
                progress_cb=self._progress_cb,
                phase_cb=self._phase_cb,
                should_run=lambda: self._running,
                **self._kwargs
            )
            self.finished.emit(result if isinstance(result, str) else str(result))
        except Exception as e:
            self.error.emit(str(e))
