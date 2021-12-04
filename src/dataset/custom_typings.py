from dataclasses import dataclass

from torch import LongTensor


@dataclass
class PaddedNSortedUserHistoryBatch:
    data: LongTensor
    lengths: LongTensor
