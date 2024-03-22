
import torch
import torch.nn as nn


class ConstantStd(nn.Module): 
    def __init__(self, std=0.015, min_std=1e-4, max_std=0.25, trainable=True):
        super(ConstantStd, self).__init__()
        self.trainable = trainable
        requires_grad = trainable
        self.std = nn.Parameter(torch.tensor(std), requires_grad=requires_grad)
        self.max_std = max_std
        self.min_std = min_std

    def rsample(self, shape=[], device=None):
        return torch.randn(shape, device=device) * self.std.clamp(self.min_std, self.max_std)


class UniformStd(nn.Module): 
    def __init__(self, min_std=1e-4, max_std=0.1, trainable=True):
        super(UniformStd, self).__init__()
        self.trainable = trainable
        requires_grad = trainable
        self.max_std = nn.Parameter(torch.tensor(max_std), requires_grad=requires_grad)
        self.min_std = nn.Parameter(torch.tensor(min_std), requires_grad=requires_grad)
        self.dist = torch.distributions.Uniform(self.min_std, self.max_std)

    def rsample(self, shape=[], device=None):
        return torch.randn(shape, device=device) * self.dist.rsample(shape).to(device)


class NormalStd(nn.Module): 
    def __init__(self, mean=0.015, std=0.25, trainable=True):
        super(NormalStd, self).__init__()
        self.trainable = trainable
        requires_grad = trainable
        self.mean_std = nn.Parameter(torch.tensor(mean), requires_grad=requires_grad)
        self.std_std = nn.Parameter(torch.tensor(std), requires_grad=requires_grad)
        self.dist = torch.distributions.Normal(self.mean_std, self.std_std)

    def rsample(self, shape=[], device=None):
        return torch.randn(shape, device=device) * self.dist.rsample(shape).to(device)


class Uniform(nn.Module):
    def __init__(self, low=-0.5, high=0.5, trainable=True):
        super(Uniform, self).__init__()
        self.trainable = trainable
        requires_grad = trainable
        self.low = nn.Parameter(torch.tensor(low), requires_grad=requires_grad)
        self.high = nn.Parameter(torch.tensor(high), requires_grad=requires_grad)
        self.dist = torch.distributions.Uniform(self.low, self.high)

    def rsample(self, shape=[], device=None):
        return self.dist.rsample(shape).to(device)


class NoiseDistribution(nn.Module):
    '''
    Kwarg:
        method (str): How the noise is generated.
            'uniform': noise is sampled from a uniform distribution
            The noise is sampled from Normal Distribution with different kind of standard deviation:
                'uniformstd': Standard deviation is sampled from a uniform distribution 
                'constant': Standard deviation is set to a constant value
                'normal': Standard deviation is sampled from a normal distribution. The mean of standard deviation is set to `std`

        std (float): Standard deviation.

        upper_std (float): Upper bound of standard deviation.
        lower_std (float): Lower bound of standard deviation.
            if method = 'uniform': initial noise boundary will be [-0.5, 0.5]
            if method = 'uniformstd': initial std boundary will be [1e-4, 0.1]
            if method = 'normal': initial std boundary will be [1e-4, 0.25]
        
        trainable (bool): Whether the standard deviation is trainable.
            if trainable:
                method = 'uniform': The upper and lower bounds are trained.
                method = 'uniformstd': The upper and lower  bounds of std are trained.
                method = 'constant': The standard deviation is trained.
                method = 'normal': The mean and std of standard deviation is trained.
    '''
    def __init__(self, method='uniformstd', trainable=True, **kwargs) -> None:
        super().__init__()
        self.trainable = trainable

        if method == 'uniform':
            high = kwargs.get('high', 0.5)
            low = kwargs.get('low', -0.5)
            self.dist = Uniform(low, high, trainable=trainable)
        elif method == 'uniformstd':
            max_std = kwargs.get('max_std', 0.1)
            min_std = kwargs.get('min_std', 1e-4)
            self.dist = UniformStd(min_std, max_std, trainable=trainable)
        elif method == 'normal':
            mean_std = kwargs.get('mean_std', 1e-4)
            std_std = kwargs.get('std_std', 0.25)
            self.dist = NormalStd(mean_std, std_std, trainable=trainable)
        elif method == 'constant':
            max_std = kwargs.get('max_std', 0.25)
            min_std = kwargs.get('min_std', 1e-4)
            std = kwargs.get('std', 0.015)
            self.dist = ConstantStd(std, min_std, max_std, trainable=trainable)

    def rsample(self, shape, device=None):
        return self.dist.rsample(shape, device)

