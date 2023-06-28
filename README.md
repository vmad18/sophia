# sophia

Implementation of Sophia: **S**econd-**o**rder Cli**p**ped Stoc**h**astic Opt**i**miz**a**tion

Sophia works by pre-conditioning the loss landscape by the Hessian. 
By pre-conditioning on the local curvature, Sophia can converges ~2x faster than Adam (on GPT-2). 

* Currently implemented Sophia-H optimizer as proposed in the paper. 
* Will implement Sophia-G later.

# Algorithm
* Compute EMA of gradient (m)
* Compute diagonal hessian via an Estimator
* Compute EMA of hessian (h)
* clip(m/h, rho)
* update param

The clip is necessary in case of a flat dimension where the hessian is close to 0.

The authors of sophia purpose 2 estimators for the diagonal of the hessian: **Hutchinson** and **Gauss-Netwon-Bartlett**
Hence two optimizers: 
* Sophia-H
* Sophia-G


Hutchinson Algorithm:
```python
class Hutchinson(Estimator):

    def __init__(self) -> None:
        super().__init__()

    """
    
    Computes the Diagonal of Hessian
    
    :param params - model parameters
    :param loss - model loss
    :param batch - mini batch
    
    """

    def compute(self, params: List[Tensor], loss: Tensor, batch: Tensor = null) -> List[Tensor]:

        u: List[Tensor] = list([torch.randn_like(p) for p in params])

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
```



# Paper

[Sophia Paper](https://arxiv.org/pdf/2305.14342.pdf)
