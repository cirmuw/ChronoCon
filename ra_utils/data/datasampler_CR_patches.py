from collections import defaultdict
import random
from torch.utils.data import Sampler, DataLoader, BatchSampler

class PatientBatchSampler(Sampler):
    """
    Yields lists of indices so that (almost) every batch
    contains samples from a single patient.
    """
    def __init__(self, patient_ids, batch_size, drop_last=False):
        self.batch_size  = batch_size
        self.drop_last   = drop_last

        # Map patient_id → list[index]
        self.patient2idx = defaultdict(list)
        for idx, pid in enumerate(patient_ids):
            self.patient2idx[pid].append(idx)

        self._build_epoch_batches()

    def _build_epoch_batches(self):
        # Shuffle *within* each patient
        for idx_list in self.patient2idx.values():
            random.shuffle(idx_list)

        # Pick a new random order of patients each epoch
        self.epoch_batches = []
        patient_order = list(self.patient2idx.keys())
        random.shuffle(patient_order)

        for pid in patient_order:
            idxs = self.patient2idx[pid]
            # chunk into batch_size pieces
            for i in range(0, len(idxs), self.batch_size):
                chunk = idxs[i : i + self.batch_size]
                if len(chunk) == self.batch_size or not self.drop_last:
                    self.epoch_batches.append(chunk)

    def __iter__(self):
        # Re-shuffle every epoch
        self._build_epoch_batches()
        for batch in self.epoch_batches:
            yield batch

    def __len__(self):
        return len(self.epoch_batches)