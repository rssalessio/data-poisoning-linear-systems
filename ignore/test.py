import torch
from torch import nn

dev = torch.device("cuda:0")

def f_true(x, slope=0.5, bias=0.3):
    '''The true underlying relation.'''
    return slope * x + bias

def get_data(n_points, noise_level=0.1, true_function=f_true, **tf_kwargs):
    '''Generates noisy data from true_function.
    Arguments:
        n_points (int): Number of datapoints to generate
        noise_level (float): Std of gaussian noise to be added
        true_function (callable): The noiseless underlying function
        **function_kwargs: Optional key-word arguments passed to true_function
    '''
    x = 2 * torch.rand(n_points, 1) - 1
    y = true_function(x, **tf_kwargs) + noise_level * torch.randn(n_points, 1)
    return x, y

class LinearModel(nn.Module):
    '''A PyTorch linear regression model.'''
    def __init__(self, in_features, out_features):
        super().__init__()
        # In this function initialize objects that we want to use later
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        # Here we define the forward pass
        # PyTorch will keep track of the computational graph in the background,
        # which means we don't have to worry about implementing the backwards pass
        return self.linear(x)

# Instantiate the model
model = LinearModel(in_features=1, out_features=1)
x, y = get_data(300)
x_gpu = x.to(dev)
y_gpu = y.to(dev)
model = model.to(dev)

def train_gpu(model, loss_function, optimizer, n_epochs=20):
    '''Training the model.'''
    # Notify model to use training settings, used in possible dropout layers etc.
    model.train()
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1:2d}/{n_epochs}", end="")
        
        # Reset optimizer
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        
        print(f"\tLoss {loss:.4g}")
        
        # Backward pass
        loss.backward()
        optimizer.step()

# Specify loss function and link optimizer with model parameters
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.3)

# Start the training
train_gpu(model, loss_function, optimizer)