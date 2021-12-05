from dataclasses import dataclass

from torch import LongTensor, device


@dataclass
class PaddedNSortedUserHistoryBatch:
    data: LongTensor
    lengths: LongTensor

    def to(self, device: device):
        return PaddedNSortedUserHistoryBatch(
            data=self.data.to(device), lengths=self.lengths
        )
