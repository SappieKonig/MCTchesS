import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff1 = nn.Linear(18, 32)
        self.ff_policy = nn.Linear(32, 9)
        self.ff_value = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.ff1(x))
        policy = torch.softmax(self.ff_policy(x), dim=1)
        value = torch.tanh(self.ff_value(x))
        return policy, value

    def as_list(self):
        # get flattened list of weights and biases
        w1 = self.ff1.weight.view(-1,).data.tolist()
        b1 = self.ff1.bias.view(-1,).data.tolist()
        w2 = self.ff_policy.weight.view(-1,).data.tolist()
        b2 = self.ff_policy.bias.view(-1,).data.tolist()
        w3 = self.ff_value.weight.view(-1,).data.tolist()
        b3 = self.ff_value.bias.view(-1,).data.tolist()
        return [w1, b1, w2, b2, w3, b3]
