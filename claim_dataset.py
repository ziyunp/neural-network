import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class ClaimDataset(Dataset):
    """ Claim dataset. """
    _ATTRIBUTE_NUM = 9
    _LABEL_IDX = 9

    def __init__(self, npdata):
        """
        Args:
            npdata (ndarray): Dataset stored in a numpy array. 
        """
        # drv_age1, vh_age, vh_cyl, vh_din, pol_bonus, vh_sale_begin, vh_sale_end, 
        # vh_value, vh_speed, claim_amount, made_claim
        self.dataset = npdata

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # if the index is a vector, convert it into a list
        if torch.is_tensor(index):
            index = index.tolist()

        attributes = self.dataset[index, :self._ATTRIBUTE_NUM]
        label = self.dataset[index, self._LABEL_IDX:]
        sample = {'attributes' : attributes, 'label' : label}

        return sample
