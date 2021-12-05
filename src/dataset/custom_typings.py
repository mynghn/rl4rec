from dataclasses import dataclass

from torch import LongTensor, device


@dataclass
class PaddedNSortedUserHistoryBatch:
    data: LongTensor
    lengths: LongTensor

    def to(self, device: device):
        self.data = self.data.to(device)
        self.lengths = self.lengths.to(device)
