import torch
import torch.nn as nn
from sklearn.datasets import load_boston

boston = load_boston()
print(boston.data.shape)
from pandas import DataFrame
from numpy import matrix

n_in, n_h, n_out, batch_size = 10, 5, 1, 12

#n_in, batch_size = boston.data.shape

#x = torch.randn(n_in, batch_size)
#y = torch.randn(n_in, 1)

x = torch.randn(n_in, batch_size)
y = torch.randn(n_in, 1)

model = nn.Sequential(nn.Linear(n_in, n_h), nn.ReLU(),
nn.Linear(n_h, n_out), nn.Sigmoid())

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(50):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    # Forward pass: Compute predicted y by passing x to the model y_pred = model(x)
    # Compute and print loss
    print('epoch:', epoch,'loss:', loss.item())
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    # perform a backward pass (backpropagation)
    loss.backward()
    optimizer.step()

pass
