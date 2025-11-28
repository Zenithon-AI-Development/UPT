import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

import torch

from callbacks.base.periodic_callback import PeriodicCallback
from distributed.config import is_rank0, is_distributed
import torch.distributed as dist


class MemoryCallback(PeriodicCallback):
    def __init__(
            self,
            enabled: bool = False,
            sample_every_n_updates: int = 50,
            sample_first_n_per_epoch: int = 3,
            reduce: str = "max",
            file_format: str = "pkl",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.enabled = bool(enabled)
        self.sample_every_n_updates = int(sample_every_n_updates or 0)
        self.sample_first_n_per_epoch = int(sample_first_n_per_epoch or 0)
        assert reduce in ["max"], f"unsupported reduce: {reduce}"
        self.reduce = reduce
        assert file_format in ["pkl"], f"unsupported file_format: {file_format}"
        self.file_format = file_format

        # buffers
        self._microstep_memories: List[Dict[str, float]] = []
        self._per_update_records: List[Dict[str, Any]] = []

        # IO
        self._profiling_dir: Path = self.path_provider.stage_output_path / "profiling"
        self._memory_dir: Path = self._profiling_dir / "memory"
        if is_rank0():
            self._memory_dir.mkdir(parents=True, exist_ok=True)

    def _to_string(self):
        return (
            f", enabled={self.enabled}, k={self.sample_every_n_updates}, "
            f"first_n={self.sample_first_n_per_epoch}, reduce={self.reduce}, fmt={self.file_format}"
        )

    # Accumulate per-microstep memory peaks (forward/backward/inner model)
    def _track_after_accumulation_step(self, update_outputs=None, memories: Dict[str, float] = None, **_):
        if not self.enabled:
            return
        if memories is None:
            return
        # store a shallow copy to avoid accidental mutation
        self._microstep_memories.append({k: float(v) for k, v in memories.items()})

    # Receive once-per-effective-update memory peaks (data_loading/optim_step)
    def _track_after_update_step(self, update_counter, trainer, model, mems: Dict[str, float] = None, **_):
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
            self._microstep_memories.clear()
            return

        memories: Dict[str, float] = {}
        # aggregate microsteps (max) if present
        if len(self._microstep_memories) > 0:
            agg = defaultdict(float)
            for t in self._microstep_memories:
                for k, v in t.items():
                    agg[k] = max(agg[k], float(v))
            memories.update(agg)
        # attach per-update outer mems if provided
        if mems is not None:
            for k, v in mems.items():
                memories[str(k)] = float(v)
        # standard top-level keys expected later (in bytes)
        for key in [
            "mem/data_loading_bytes",
            "mem/forward_bytes",
            "mem/loss_bytes",
            "mem/backward_bytes",
            "mem/optim_step_bytes",
            "mem/model/conditioner_bytes",
            "mem/model/encoder_bytes",
            "mem/model/latent_bytes",
            "mem/model/decoder_bytes",
        ]:
            memories.setdefault(key, 0.0)

        record = dict(
            meta=dict(
                epoch=update_counter.epoch,
                global_update=update_counter.update,
                accumulation_steps=getattr(trainer, "accumulation_steps", None),
                batch_size=getattr(trainer, "effective_batch_size", None),
                rank=int(os.environ.get("RANK", "0")),
            ),
            memory=memories,
        )
        self._per_update_records.append(record)
        self._microstep_memories.clear()

    def _periodic_callback(self, interval_type, **_):
        if not self.enabled:
            return
        # compute per-epoch means of per-update peaks (then DDP-reduce with max)
        if len(self._per_update_records) == 0:
            return
        keys = sorted(self._per_update_records[0]["memory"].keys())
        sums = {k: 0.0 for k in keys}
        for rec in self._per_update_records:
            for k in keys:
                sums[k] += float(rec["memory"].get(k, 0.0))
        means = {k: (sums[k] / len(self._per_update_records)) for k in keys}

        # DDP max reduction across ranks (bytes)
        reduced = {}
        for k, v in means.items():
            t = torch.tensor(v)
            if is_distributed():
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
            reduced[k] = t.item()

        # Log reduced means (bytes) using interval_type postfix
        for k, v in reduced.items():
            self.writer.add_scalar(key=f"{k}/{interval_type}", value=v)

        # Write epoch pkl (rank 0 only)
        if is_rank0() and self.file_format == "pkl":
            epoch = self.update_counter.epoch
            uri = self._memory_dir / f"updates_epoch{epoch}_mem.pkl"
            with open(uri, "wb") as f:
                pickle.dump(self._per_update_records, f)

        # clear for next epoch interval
        self._per_update_records.clear()


