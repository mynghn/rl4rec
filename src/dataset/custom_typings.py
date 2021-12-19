from dataclasses import dataclass

from torch import LongTensor, device


@dataclass
class PaddedNSortedEpisodeBatch:
    data: LongTensor
    lengths: LongTensor

    def to(self, device: device):
        return PaddedNSortedEpisodeBatch(
            data=self.data.to(device), lengths=self.lengths
        )
