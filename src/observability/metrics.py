import time
import tracemalloc
# from contextlib import contextmanager
from src.core.models import StepMetrics


class Timer:
    """
    Context manager for timing + memory measurement.

    Usage:
        with Timer("hybrid_search") as t:
            results = lance_store.hybrid_search(...)
        metrics = t.to_metrics(input_count=5, output_count=len(results))

    WHY __enter__ / __exit__ over try/finally?
    try/finally: you write the timing boilerplate every time.
    Context manager: timing logic defined once, used everywhere.
    Also: __exit__ receives the exception if one occurs —
    you can still record partial metrics even on failure.
    """

    def __init__(self, step_name: str, track_memory: bool = True):
        self.step_name    = step_name
        self.track_memory = track_memory
        self.latency_ms   = 0.0
        self.memory_delta_mb = 0.0
        self._start_time  = 0.0

    def __enter__(self) -> "Timer":
        self._start_time = time.perf_counter()
        if self.track_memory:
            tracemalloc.start()
        return self
        # WHY return self?
        # The `as t` in `with Timer(...) as t` binds to
        # whatever __enter__ returns. We want the Timer
        # instance so caller can read .latency_ms after.

    def __exit__(
        self,
        exc_type,   # None if no exception
        exc_val,    # exception instance or None
        exc_tb,     # traceback or None
    ) -> bool:
        self.latency_ms = (time.perf_counter() - self._start_time) * 1000

        if self.track_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.memory_delta_mb = peak / (1024 * 1024)

        # WHY return False (implicit)?
        # Returning True suppresses the exception.
        # We want exceptions to propagate — timer should not
        # swallow errors. Return False = let exception bubble up.
        # We still recorded the timing before it propagated.
        return False

    def to_step_metrics(
        self,
        input_count:  int = 0,
        output_count: int = 0,
        **extra,
    ) -> StepMetrics:
        return StepMetrics(
            step_name=self.step_name,
            latency_ms=round(self.latency_ms, 2),
            input_count=input_count,
            output_count=output_count,
            memory_delta_mb=round(self.memory_delta_mb, 3),
            extra=extra,
        )