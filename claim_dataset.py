import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class ClaimDataset(Dataset):
    """ Claim dataset. """
    _ATTRIBUTE_NUM = 9
    _LABEL_IDX = 10

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file of raw data. 
        """
        # drv_age1, vh_age, vh_cyl, vh_din, pol_bonus, vh_sale_begin, vh_sale_end, 
        # vh_value, vh_speed, claim_amount, made_claim
        self.dataset = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        np.random.shuffle(self.dataset)

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

def print_dataset():
    dataset = ClaimDataset('part2_training_data.csv')
    print(dataset[:])

if __name__ == "__main__":
    print_dataset()