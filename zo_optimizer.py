import torch
import torch.nn as nn
from typing import Callable

class ZeroOrderOptimizer:
    def __init__(self, model: nn.Module, lr=1e-3, eps=1e-3):
        self.model = model
        self.lr = lr
        self.eps = eps
        self.layer_names = ["fc.bias"]
        
        self.m = {} # Adam first moment
        self.v = {} # Adam second moment
        self.t = 0
        self.beta1, self.beta2 = 0.9, 0.999

    def _sample_u(self, param):
        # Rademacher (+1/-1) better gaussians for SPSA
        return torch.randint(0, 2, param.shape, device=param.device) * 2.0 - 1.0

    def step(self, loss_fn: Callable[[], float]) -> float:
        self.t += 1
        params = {n: p for n, p in self.model.named_parameters() if n in self.layer_names}
        
        with torch.no_grad():
            u = {n: self._sample_u(p) for n, p in params.items()}
            
            # f(theta + eps*u)
            for n, p in params.items(): p.data.add_(u[n], alpha=self.eps)
            f_plus = loss_fn()
            
            # f(theta - eps*u)
            for n, p in params.items(): p.data.sub_(u[n], alpha=2*self.eps)
            f_minus = loss_fn()
            
            # Recovery
            for n, p in params.items(): p.data.add_(u[n], alpha=self.eps)
            
            # Gradient estimation
            g_shared = (f_plus - f_minus) / (2 * self.eps)
            
            for n, p in params.items():
                grad = g_shared * u[n]
                
                # Adam update
                if n not in self.m:
                    self.m[n] = torch.zeros_like(p)
                    self.v[n] = torch.zeros_like(p)
                
                self.m[n] = self.beta1 * self.m[n] + (1 - self.beta1) * grad
                self.v[n] = self.beta2 * self.v[n] + (1 - self.beta2) * (grad**2)
                
                m_hat = self.m[n] / (1 - self.beta1**self.t)
                v_hat = self.v[n] / (1 - self.beta2**self.t)
                
                p.data.sub_(self.lr * m_hat / (torch.sqrt(v_hat) + 1e-8))
                
        return (f_plus + f_minus) / 2