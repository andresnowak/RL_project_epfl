from torch.distributions import TransformedDistribution, TanhTransform
import torch.distributions as dist
import torch

class TanhNormal(dist.Distribution):
    def __init__(self, loc, scale, eps=1e-6):
        super().__init__()
        self.normal = dist.Normal(loc, scale)
        self.eps = eps  # Numerical stability

    def sample(self, sample_shape=torch.Size()):
        x = self.normal.sample(sample_shape)
        return torch.tanh(x)

    def rsample(self, sample_shape=torch.Size()):
        x = self.normal.rsample(sample_shape)  # Differentiable
        return torch.tanh(x)

    def log_prob(self, value):
        # Inverse tanh (atanh) + correction term
        y = torch.clamp(value, -1 + self.eps, 1 - self.eps)
        x = 0.5 * (torch.log1p(y) - torch.log1p(-y))  # atanh(y)
        log_prob = self.normal.log_prob(x) - torch.log(1 - y.pow(2) + self.eps)
        return log_prob
    
    def entropy(self):
        # H(base_normal) - E[log(1 - tanh(x)^2)]
        base_entropy = self.normal.entropy()
        correction = torch.log(torch.tensor(4.0))  # log(4) ≈ 1.386 (empirically derived)
        return base_entropy - correction

