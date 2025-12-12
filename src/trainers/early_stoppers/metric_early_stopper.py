from callbacks.base.callback_base import CallbackBase
from utils.infer_higher_is_better import higher_is_better_from_metric_key
from .base.early_stopper_base import EarlyStopperBase


class MetricEarlyStopper(EarlyStopperBase):
    def __init__(self, metric_key, tolerance, **kwargs):
        super().__init__(**kwargs)
        self.metric_key = metric_key
        self.higher_is_better = higher_is_better_from_metric_key(self.metric_key)
        assert tolerance is not None and tolerance >= 1, "tolerance has to be >= 1"
        self.tolerance = tolerance
        self.tolerance_counter = 0
        self.best_metric = -float("inf") if self.higher_is_better else float("inf")

    def _metric_improved(self, cur_metric):
        if self.higher_is_better:
            return cur_metric > self.best_metric
        return cur_metric < self.best_metric

    def _should_stop(self):
        writer = CallbackBase.log_writer_singleton
        assert writer is not None
        # Try metric key with epoch suffix first (as logged by UpdateOutputCallback)
        metric_key_with_suffix = f"{self.metric_key}/{self.to_short_interval_string()}"
        if metric_key_with_suffix in writer.log_cache:
            metric_key_to_use = metric_key_with_suffix
        elif self.metric_key in writer.log_cache:
            metric_key_to_use = self.metric_key
        else:
            assert False, (
                f"couldn't find metric_key {self.metric_key} or {metric_key_with_suffix} "
                f"(valid metric_keys={writer.log_cache.keys()}) -> "
                "make sure every_n_epochs/every_n_updates/every_n_samples is aligned with the corresponding callback"
            )
        cur_metric = writer.log_cache[metric_key_to_use]

        if self._metric_improved(cur_metric):
            self.logger.info(f"{self.metric_key} improved: {self.best_metric} --> {cur_metric}")
            self.best_metric = cur_metric
            self.tolerance_counter = 0
        else:
            self.tolerance_counter += 1
            cmp_str = "<=" if self.higher_is_better else ">="
            stop_training_str = " --> stop training" if self.tolerance_counter >= self.tolerance else ""
            self.logger.info(
                f"{self.metric_key} stagnated: {self.best_metric} {cmp_str} {cur_metric} "
                f"({self.tolerance_counter}/{self.tolerance}){stop_training_str}"
            )

        return self.tolerance_counter >= self.tolerance
