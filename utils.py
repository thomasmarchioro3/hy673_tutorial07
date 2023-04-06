import torch
import torch.nn as nn

class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class CNNModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn_layers = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=4),
                Swish(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                Swish(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                Swish(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                Swish(),
                nn.Flatten(),
                nn.Linear(256, 64),
                Swish(),
                nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        return x
    

def generate_samples(model, x_hat, steps=60, step_size=10, return_x_steps=False):
    """
    Function for sampling images for a given model.
    Inputs:
        model - Neural network modeling the energy
        x_hat - Images to start from for sampling. To generate new images from scrath, enter noise between -1 and 1.
        steps - Number of iterations in the MCMC algorithm.
        step_size - Multiplicative factor for each step in the MCMC algorithm.
        return_x_steps - If True, we return the sample at every iteration of the MCMC
    """
    is_training = model.training  # check if the model is in training mode (to restore it later)

    # Before MCMC, set model parameters to "required_grad=False"
    # since we are only interested in the gradients w.r.t. x
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    x_hat.requires_grad = True

    # allocate a tensor to generate noise each loop iteration
    # more efficient than creating a new tensor every iteration
    noise = torch.randn(x_hat.shape, device=x_hat.device)

    x_steps = []

    # iterative MCMC sampling
    for _ in range(steps):
        # add noise to the input.
        noise.normal_(0, 0.005)
        x_hat = torch.clamp(x_hat + noise, -1, 1)

        # calculate gradients w.r.t. input.
        out = model(x_hat).sum()
        grad, = torch.autograd.grad(out, x_hat, only_inputs=True)

        # for stability (limits gradient value)
        grad = torch.clamp(grad, -0.03, 0.03)

        # MCMC step
        x_hat = x_hat + step_size * grad

        # reset gradients in x_hat
        x_hat = torch.clamp(x_hat, -1, 1)

        if return_x_steps:
            x_steps.append(x_hat.clone().detach())

    # Reactivate gradients for the model parameters
    for p in model.parameters():
        p.requires_grad = True
    model.train(is_training)

    if return_x_steps:
        return torch.stack(x_steps, dim=0)
    else:
        return x_hat