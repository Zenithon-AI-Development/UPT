import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

import torch

from callbacks.base.periodic_callback import PeriodicCallback
from distributed.config import is_rank0
from distributed.gather import all_reduce_mean_grad


class TimingCallback(PeriodicCallback):
    def __init__(
            self,
            enabled: bool = False,
            sample_every_n_updates: int = 50,
            sample_first_n_per_epoch: int = 3,
            reduce: str = "mean",
            file_format: str = "pkl",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.enabled = bool(enabled)
        self.sample_every_n_updates = int(sample_every_n_updates or 0)
        self.sample_first_n_per_epoch = int(sample_first_n_per_epoch or 0)
        assert reduce in ["mean"], f"unsupported reduce: {reduce}"
        self.reduce = reduce
        assert file_format in ["pkl"], f"unsupported file_format: {file_format}"
        self.file_format = file_format

        # buffers
        self._microstep_timings: List[Dict[str, float]] = []
        self._per_update_records: List[Dict[str, Any]] = []

        # IO
        self._profiling_dir: Path = self.path_provider.stage_output_path / "profiling"
        self._timings_dir: Path = self._profiling_dir / "timings"
        if is_rank0():
            self._timings_dir.mkdir(parents=True, exist_ok=True)

    def _to_string(self):
        return (
            f", enabled={self.enabled}, k={self.sample_every_n_updates}, "
            f"first_n={self.sample_first_n_per_epoch}, reduce={self.reduce}, fmt={self.file_format}"
        )

    # Accumulate per-microstep timings (forward/backward/inner model), optional at this stage
    def _track_after_accumulation_step(self, update_outputs=None, timings: Dict[str, float] = None, **_):
        if not self.enabled:
            return
        if timings is None:
            return
        # store a shallow copy to avoid accidental mutation
        self._microstep_timings.append({k: float(v) for k, v in timings.items()})

    # Receive once-per-effective-update timings (data/update_total/optim_step)
    def _track_after_update_step(self, update_counter, trainer, model, times: Dict[str, float] = None, **_):
        if not self.enabled:
            return
        # sampling policy: keep first N each epoch, otherwise every Kth completed update
        updates_per_epoch = update_counter.updates_per_epoch
        completed_update_idx = (update_counter.update - 1) % updates_per_epoch  # just-completed
        should_keep = completed_update_idx < self.sample_first_n_per_epoch
        if not should_keep and self.sample_every_n_updates:
            should_keep = (completed_update_idx % self.sample_every_n_updates) == 0
        if not should_keep:
            # clear microstep buffer regardless to avoid unbounded growth
            self._microstep_timings.clear()
            return

        timings: Dict[str, float] = {}
        # aggregate microsteps (sum) if present
        if len(self._microstep_timings) > 0:
            agg = defaultdict(float)
            for t in self._microstep_timings:
                for k, v in t.items():
                    agg[k] += float(v)
            timings.update(agg)
        # attach per-update outer timings if provided
        if times is not None:
            for k, v in times.items():
                timings[f"time/{k}"] = float(v)
        # standard top-level keys expected later
        for key in [
            "time/update_total",
            "time/data_loading",
            "time/forward",
            "time/backward",
            "time/optim_step",
            "time/model/conditioner",
            "time/model/encoder",
            "time/model/latent",
            "time/model/decoder",
        ]:
            timings.setdefault(key, 0.0)

        record = dict(
            meta=dict(
                epoch=update_counter.epoch,
                global_update=update_counter.update,
                accumulation_steps=getattr(trainer, "accumulation_steps", None),
                batch_size=getattr(trainer, "effective_batch_size", None),
                rank=int(os.environ.get("RANK", "0")),
            ),
            timings=timings,
        )
        self._per_update_records.append(record)
        self._microstep_timings.clear()

    def _periodic_callback(self, interval_type, **_):
        if not self.enabled:
            return
        # compute per-epoch means (reduce across ranks)
        if len(self._per_update_records) == 0:
            return
        keys = sorted(self._per_update_records[0]["timings"].keys())
        sums = {k: 0.0 for k in keys}
        for rec in self._per_update_records:
            for k in keys:
                sums[k] += float(rec["timings"].get(k, 0.0))
        means = {k: (sums[k] / len(self._per_update_records)) for k in keys}

        # DDP mean reduction
        reduced_means = {k: all_reduce_mean_grad(torch.tensor(v)).item() for k, v in means.items()}

        # Log reduced means (single series) using interval_type postfix
        for k, v in reduced_means.items():
            # Avoid slashes at start; writer handles postfix internally if used
            self.writer.add_scalar(key=f"{k}/{interval_type}", value=v)

        # Write epoch pkl (rank 0 only)
        if is_rank0() and self.file_format == "pkl":
            epoch = self.update_counter.epoch
            uri = self._timings_dir / f"updates_epoch{epoch}_time.pkl"
            with open(uri, "wb") as f:
                pickle.dump(self._per_update_records, f)

        # clear for next epoch interval
        self._per_update_records.clear()


