import mcts_tic_tac_toe as inference_tools

import torch
from torch.utils.data import DataLoader

from network import Net
from dataset import TicTacToeDataset

BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 0.01
CYCLES = 1000


net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

for cycle in range(CYCLES):
    print(f"Cycle {cycle}")
    weights = net.as_list()
    cpp_net = inference_tools.Net(weights)
    inferencer = inference_tools.Inferencer(cpp_net)
    print(f"Results against random: {inferencer.test_against_random(100)}")
    samples = inferencer.get_samples(10_000)
    print(sum(abs(sample.value) for sample in samples))
    dataset = TicTacToeDataset(samples)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        for inp, policy, value in dataloader:
            optimizer.zero_grad()
            policy_pred, value_pred = net(inp)
            loss = torch.nn.functional.mse_loss(value_pred, value) - torch.sum(policy * torch.log(policy_pred))
            loss.backward()
            optimizer.step()

