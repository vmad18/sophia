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

# Paper

[Sophia Paper](https://arxiv.org/pdf/2305.14342.pdf)
