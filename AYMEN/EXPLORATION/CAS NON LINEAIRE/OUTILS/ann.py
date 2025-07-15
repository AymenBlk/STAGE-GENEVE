import numpy as np
import torch

class ANN(torch.nn.Module):
    def __init__(self, layer_sizes, activation=torch.nn.ReLU, last_activation=None, verbose=False):
        super().__init__()

        self.verbose = verbose
        layers = []
        for i in range(len(layer_sizes) - 1):
            in_f, out_f = layer_sizes[i], layer_sizes[i + 1]
            layers.append(torch.nn.Linear(in_f, out_f))
            if i < len(layer_sizes) - 2:
                layers.append(activation())
            elif last_activation is not None:
                layers.append(last_activation())

        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
    def train(self, X, Y, epochs=1000, lr=0.01, optimizer=None, loss_fn=None, batch_size=32, shuffle=True):
        loss_function = loss_fn or torch.nn.MSELoss()
        #optimizer = optimizer or torch.optim.SGD(self.parameters(), lr=lr)
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=lr)

        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        for epoch in range(epochs):
            for batch_X, batch_Y in dataloader:
                optimizer.zero_grad()

                outputs = self.forward(batch_X)
                loss = loss_function(outputs, batch_Y)

                loss.backward()
                optimizer.step()

            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")