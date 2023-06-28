# sophia

Implementation of Sophia: **S**econd-**o**rder Cli**p**ped Stoc**h**astic Opt**i**miz**a**tion

Sophia works by pre-conditioning the loss landscape by the Hessian. 
By pre-conditioning on the local curvature, Sophia can converges ~2x faster than Adam (on GPT-2). 

* Currently implemented SophiaH optimizer as proposed in the paper. 

# The Algorithm:
* Compute EMA of gradient (m)
* Compute diagonal Hessian via an Estimator
* Compute EMA of Hessian (h)
* clip(m/h, rho)
* update param

The clip is necessary in case of a flat dimension. 


# Paper

[Sophia Paper](https://arxiv.org/pdf/2305.14342.pdf)
