"""
Slot-based UPT: Stage 1 implementation.

Wraps vanilla UPT with fixed [M, N] slot layer for computational efficiency
and time-varying data support.
"""

from .slot_assignment import (
    assign_cells_to_slots_voxel_grid,
    get_slot_positions,
    scatter_slots_to_cells,
)
from .slot_aggregator import (
    SlotAggregatorBase,
    MaskedMeanSlotAggregator,
)
from .slot_splitter import SlotSplitter

__all__ = [
    # Slot assignment
    "assign_cells_to_slots_voxel_grid",
    "get_slot_positions",
    "scatter_slots_to_cells",
    # Slot aggregator
    "SlotAggregatorBase",
    "MaskedMeanSlotAggregator",
    # Slot splitter
    "SlotSplitter",
]

