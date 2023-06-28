from torch.optim import Optimizer
from utils.consts import *

"""

Estimates the value of the Hessian for Pre-conditioning

"""


class Estimator:

    def __init__(self) -> None: pass

    def compute(self, params: List[Tensor], loss: Tensor, batch: Tensor = null) -> List[Tensor]:
        pass


class Hutchinson(Estimator):

    def __init__(self) -> None:
        super().__init__()

    """
    
    Computes the Diagonal of Hessian
    
    :param p - parameter
    :param g - parameter gradient
    :param batch - mini batch
    
    """

    def compute(self, params: List[Tensor], loss: Tensor, batch: Tensor = null) -> List[Tensor]:

        u: List[Tensor] = list([torch.randn_like(p) for p in params])  # noise matrices

        J: Tuple[Tensor] = torch.autograd.grad(loss, params, create_graph=true)  # compute jacobian

        gradu: List[Tensor] = []  # dot products with gradient and noise

        for i, g in enumerate(J):
            gradu.append((g * u[i]).sum())

        hvp: Tuple[Tensor] = torch.autograd.grad(gradu, params,
                                                 retain_graph=true)  # compute hessian vector product

        hessian: List[Tensor] = []

        for i, grad in enumerate(hvp):
            hessian.append(grad * u[i])
        return hessian


# class GaussNewtonBartlett(Estimator):
#
#     def __init__(self):
#         super().__init__()
#
#     def compute(self, params: List[Tensor], loss: Tensor, batch: Tensor = null):


class Sophia(Optimizer):

    def __init__(self, params, estimator: Estimator, lr: float = 8e-4, betas: tuple[float, float] = (.965, .99),
                 eps: float = 1e-8, k: int = 10, weight_decay: float = 1e-1,
                 rho: float = .022) -> None:

        self.params = params

        self.estimator = estimator
        self.hessian = []

        self.s = 0
        self.k = k

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, rho=rho, k=k, est=estimator)
        super().__init__(params, defaults)

    def estimate_hessian(self, loss: Tensor = null, batch: Tensor = null, params: List[Tensor] = null) -> None:
        if self.s % self.k == 1:
            self.hessian = self.estimator.compute(params, loss, batch)

    def step(self, loss: Tensor = null, batch: Tensor = null) -> None:
        for group in self.param_groups:
            self.estimate_hessian(loss, batch, group["params"])
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                g = p.grad  # parameters' gradient

                state = self.state[p]

                # init state
                if len(state) == 0:
                    state["step"] = 0
                    state["ema"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)  # EMA Momentum
                    state["h"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)  # Hessian

                step = state["step"]

                b1, b2 = group["betas"]
                m: Tensor = state["ema"]  # momentum
                h: Tensor = state["h"]  # hessian

                lr = group["lr"]  # learning rate

                if step % group["k"] == 1:
                    h_hat = self.hessian[i]  # get computed hessian
                    h.mul_(b2).add_(h_hat, alpha=1 - b2)

                m.mul_(b1).add_(g, alpha=1 - b1)
                
                # Update Parameters
                with torch.no_grad():
                    p.mul_(1 - lr * group["weight_decay"])

                    m_sign = m.sign()

                    dp = (m.abs() / (h.clamp(min=group["eps"]))).clamp(
                        max=group["rho"])

                    p.addcmul_(value=-lr, tensor1=m_sign, tensor2=dp)

                state["step"] += 1

            self.s += 1
