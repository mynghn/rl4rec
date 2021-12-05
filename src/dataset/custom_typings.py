from dataclasses import dataclass

from torch import LongTensor, device
from typing_extensions import DataClass


@dataclass
class PaddedNSortedUserHistoryBatch:
    data: LongTensor
    lengths: LongTensor

    def to(self, device: device):
        self.data = self.data.to(device)
        return self
