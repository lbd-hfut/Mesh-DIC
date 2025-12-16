import torch
import torch.nn as nn
import numpy as np

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.width = layers[1:-1]
        self.num_layers = len(self.width)
        self.input_size = layers[0]
        self.output_size = layers[-1]

        self.normalize_layer = nn.BatchNorm1d(self.input_size, affine=False)
        
        # Define input layer
        self.input_layer = nn.Linear(self.input_size, self.width[0])
        
        # Define hidden layers (MLP)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.width[i], self.width[i+1]) for i in range(self.num_layers-1)]
            )
        
        # Define output layer
        self.output_layer = nn.Linear(self.width[-1], self.output_size)
        
        
        # Define activation function parameter 'a'
        self.a = nn.Parameter(torch.tensor([0.2] * (self.num_layers + 2)))

    def forward(self, x):
        
        x = self.normalize_layer(x)
        # Input layer
        x = self.input_layer(x)
        x = 5 * self.a[0] * x
        x = torch.tanh(x)
        
        # Hidden layers (MLP)
        for i in range(self.num_layers-1):
            x = self.hidden_layers[i](x)
            x = 5 * self.a[i + 1] * x
            x = torch.tanh(x)
        
        # Output layer
        x = 5 * self.a[-1] * x
        x = self.output_layer(x)
        
        return x

class SinActivation(nn.Module):
    def __init__(self):
        super(SinActivation, self).__init__()
    def forward(self, x):
        return torch.sin(1*x)
    
class PhiActivation(nn.Module):
    def __init__(self):
        super(PhiActivation, self).__init__()
        self.relu = nn.ReLU()
    def forward(self, x):
        y = self.relu(x)**2 - 3*self.relu(x-1)**2 + 3*self.relu(x-2)**2 - self.relu(x-3)**2
        return y
    
class SeReLU(nn.Module):
    def __init__(self):
        super(SeReLU, self).__init__()
        self.relu = nn.ReLU()
    def forward(self, x):
        y = self.relu(x) * self.relu(1-x)
        return y
    
class SelfAdaptive(nn.Module):
    def __init__(self):
        super(SelfAdaptive, self).__init__()
        # self.a = nn.Parameter(torch.tensor(0.2))
        self.a = nn.Parameter(torch.tensor([0.5]))
        
    def forward(self, x):
        y = 2 * self.a * x
        return y

# Define activation functions
activation_dict = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'serelu': SeReLU(),
    'softplus': nn.Softplus(),
    'sin': SinActivation(),
    'phi': PhiActivation(),
}

class MscaleDNN(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim, scales, activation):
        super(MscaleDNN, self).__init__()
        self.scales = scales
        self.activation = activation_dict[activation]
        self.subnets = nn.ModuleList()
        self.normalize_layer = nn.BatchNorm1d(input_dim, affine=False)
        
        for scale in scales:
            layers = []
            prev_units = input_dim
            for i, units in enumerate(hidden_units):
                layers.append(nn.Linear(prev_units, units))
                layers.append(SelfAdaptive())
                layers.append(self.activation)
                prev_units = units
            # layers.append(SelfAdaptive())
            layers.append(nn.Linear(prev_units, output_dim))
            self.subnets.append(nn.Sequential(*layers))

    def forward(self, x):
        outputs = []
        x = self.normalize_layer(x)
        for i, scale in enumerate(self.scales):
            scaled_x = x * scale
            outputs.append(self.subnets[i](scaled_x))
        return torch.sum(torch.stack(outputs), dim=0)
    
    # Set the parameters for the early stop method
    def Earlystop_set(self, patience=10, delta=0, path=None):
        self.patience = int(patience)
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    # Logical judgment of executing the early stop method
    def Earlystop(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss < val_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
    # Save the checkpoint of the current optimal model
    def save_checkpoint(self):
        if self.path:
            checkpoint = {
                'model_state_dict': self.state_dict(),
            }
            torch.save(checkpoint, self.path)
            self.best_model = self.state_dict()
            
    # Freeze all trainable parameters  
    def freeze_all_parameters(self):  
        for param in self.parameters():  
            param.requires_grad = False  
  
    # Unfreeze all trainable parameters  
    def unfreeze_all_parameters(self):  
        for param in self.parameters():  
            param.requires_grad = True
    
    def initialize_weights(self, method='xavier'):
        """
        Initialize the weights of the model.
 
        Parameters:
        - method (str): 'xavier' or 'kaiming'. Default is 'xavier'.
        """
        if method == 'xavier':
            initializer = torch.nn.init.xavier_uniform_
        elif method == 'kaiming':
            initializer = torch.nn.init.kaiming_uniform_
        else:
            raise ValueError(
                "Unsupported initialization method. Use 'xavier' or 'kaiming'."
                )
 
        for subnet in self.subnets:
            for layer in subnet:
                if isinstance(layer, nn.Linear):
                    initializer(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                        
                # Initialize the 'SelfAdaptive' parameter 'a'
                if isinstance(layer, SelfAdaptive):
                    # Initialize 'a' to 0.5 (or your desired value)
                    # print(f" before a = {layer.a.item()}")
                    nn.init.constant_(layer.a, 0.5)  
                    # print(f" after a = {layer.a.item()}")
        
               
            
class PositionalEncoding:
    def __init__(self, num_frequencies, input_dims=3):
        self.num_frequencies = num_frequencies
        self.input_dims = input_dims
        self.create_encoding_functions()
        
    def create_encoding_functions(self):
        # Define the frequency bands
        self.frequency_bands = 2 ** torch.linspace(0, self.num_frequencies - 1, self.num_frequencies)
        
        # Create the list of encoding functions
        self.encoding_functions = []
        for freq in self.frequency_bands:
            self.encoding_functions.append(lambda x, freq=freq: torch.sin(2 * np.pi * freq * x))
            self.encoding_functions.append(lambda x, freq=freq: torch.cos(2 * np.pi * freq * x))

    def encode(self, x):
        # x is expected to be of shape (N, input_dims) where N is the batch size
        encodings = [x]  # Start with the original input
        for fn in self.encoding_functions:
            encodings.append(fn(x))
        return torch.cat(encodings, dim=-1)
