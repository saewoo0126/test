import torch
import torch.nn as nn

device = torch.device("mps")

model = nn.Linear(1, 1).to(device)

x = torch.tensor(
    [[1.0], [2.0], [3.0], [4.0]]
).to(device)

y = torch.tensor(
    [[3.0], [5.0], [7.0], [9.0]]
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10000):
    prediction = model(x)

    loss = criterion(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model(torch.tensor([[10.0]]).to(device)))
