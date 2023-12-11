import torch
from torch.utils.data import Dataset


class TicTacToeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp = torch.tensor(self.data[idx].input, dtype=torch.float)
        policy = torch.tensor(self.data[idx].policy, dtype=torch.float)
        value = torch.tensor(self.data[idx].value, dtype=torch.float).view(1,)
        return inp, policy, value


    def as_list(self):
        return self.data