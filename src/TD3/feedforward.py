import torch
from torch import nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_sizes : list, output_size, *,
                 activation_func = nn.ReLU(), output_activation = None):
        super().__init__()
        self.input_size     = input_size
        self.hidden_sizes   = hidden_sizes
        self.output_size    = output_size
        self.output_act     = output_activation
        layer_sizes         = [input_size] + self.hidden_sizes
        self.layers = nn.Sequential(
            *[nn.Sequential(nn.Linear(i, o), activation_func)
              for i, o in zip(layer_sizes[:-1], layer_sizes[1:])]
        )
        self.out = nn.Linear(self.hidden_sizes[-1], self.output_size)
        self.output_act = output_activation

    
    def forward(self, x):
        x = self.layers(x)
        x = self.out(x)
        if self.output_act is not None:
            x = self.output_act(x)
        return x
    
    def predict(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x.astype(np.float32)).to(device)
        with torch.no_grad():
            out = self.forward(x)
            return out.cpu().numpy()