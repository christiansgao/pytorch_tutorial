import torch
import torch.nn as nn
from sklearn.datasets import load_boston
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray

### Random Numbers ###
boston = load_boston()
print(boston.data.shape)
n_h, n_out = 14, 1
batch_size, n_in = boston.data.shape
target = boston.target
x = torch.Tensor(boston.data)
y = torch.Tensor(boston.target).reshape(506, 1)
lr = 0.01

print("X shape: {}".format(x.size()))
print("Y shape: {}".format(y.size()))
model = nn.Sequential(nn.Linear(n_in, n_h), nn.ReLU(), nn.Linear(n_h, n_out))
#model = nn.Sequential(nn.Linear(n_in, n_h), nn.ReLU(),nn.Linear(n_h, n_out), nn.Sigmoid())

criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for ep in range(10000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    # Forward pass: Compute predicted y by passing x to the model y_pred = model(x)
    # Compute and print loss

    if not ep % 500:
        print('Epoch: {} - loss: {}'.format(ep, loss.data))
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    # perform a backward pass (backpropagation)
    loss.backward()
    optimizer.step()

output = y_pred.data.cpu().numpy().flatten() # type: ndarray
expected = np.array(boston.target)

sns.set(style="darkgrid")
g = sns.jointplot(x=output, y=expected, kind="reg", color="m", height=7)
plt.show()