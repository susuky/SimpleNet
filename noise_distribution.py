
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
        self.max_std = nn.Parameter(torch.tensor(max_std),
                                    requires_grad=requires_grad)
        self.min_std = nn.Parameter(torch.tensor(min_std),
                                    requires_grad=requires_grad)
        self.dist = torch.distributions.Uniform(self.min_std, self.max_std)

    def rsample(self, shape=[], device=None):
        return torch.randn(shape, device=device) * self.dist.rsample(shape).to(device)


class NormalStd(nn.Module):
    def __init__(self, mean=0.0, std=0.015, trainable=True):
        super(NormalStd, self).__init__()
        self.trainable = trainable
        requires_grad = trainable
        self.mean_std = nn.Parameter(torch.tensor(mean),
                                     requires_grad=requires_grad)
        self.std_std = nn.Parameter(torch.tensor(std),
                                    requires_grad=requires_grad)
        self.dist = torch.distributions.Normal(self.mean_std, self.std_std)

    def rsample(self, shape=[], device=None):
        return torch.randn(shape, device=device) * self.dist.rsample(shape).to(device)


class Uniform(nn.Module):
    def __init__(self, low=-0.5, high=0.5, trainable=True):
        super(Uniform, self).__init__()
        self.trainable = trainable
        requires_grad = trainable
        self.low = nn.Parameter(torch.tensor(low),
                                requires_grad=requires_grad)
        self.high = nn.Parameter(torch.tensor(high),
                                 requires_grad=requires_grad)
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
                'constantstd': Standard deviation is set to a constant value
                'normalstd': Standard deviation is sampled from a normal distribution. The mean of standard deviation is set to `std`

        1. method='uniform':
            high (float): Upper bound of uniform distribution.  Default: 0.5
            low (float): Lower bound of uniform distribution. Default: -0.5 
        2. method='uniformstd':
            max_std (float): Upper bound of std from uniform distribution.  Default: 0.1
            min_std (float): Lower bound of std from uniform distribution. Default: 1e-4       
        3. method='normalstd':
            mean_std (float): Mean of std from normal distribution. Default: 0.0
            std_std (float): std of std from normal distribution. Default: 0.015
        4. method='constantstd':
            std (float): Standard deviation. Default: 0.015
            max_std (float): Upper bound of std, if std is learnable. Default: 0.25
            min_std (float): Lower bound of std, if std is learnable. Default: 1e-4
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
        elif method == 'normalstd':
            mean_std = kwargs.get('mean_std', 1e-4)
            std_std = kwargs.get('std_std', 0.015)
            self.dist = NormalStd(mean_std, std_std, trainable=trainable)
        elif method == 'constantstd':
            max_std = kwargs.get('max_std', 0.25)
            min_std = kwargs.get('min_std', 1e-4)
            std = kwargs.get('std', 0.015)
            self.dist = ConstantStd(std, min_std, max_std, trainable=trainable)

    def rsample(self, shape, device=None):
        return self.dist.rsample(shape, device)
