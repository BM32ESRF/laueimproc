#!/usr/bin/env python3

r"""Gaussian mixture model.

Scalar Terminology
------------------
* \(D\): The space dimension, same as the number of variables. Index \(d \in [\![1;D]\!]\).
* \(K\): The number of gaussians. Index \(j \in [\![1;K]\!]\).
* \(N\): The number of observations. Index \(i \in [\![1;N]\!]\).

Tensor Terminology
------------------
* \(
    \mathbf{\mu}_j =
    \begin{pmatrix}
        \mu_1  \\
        \vdots \\
        \mu_D  \\
    \end{pmatrix}_j
\): The center of the gaussian \(j\).
* \(
    \mathbf{\Sigma}_j =
    \begin{pmatrix}
        \sigma_{1,1} & \dots  & \sigma_{1,D} \\
        \vdots       & \ddots & \vdots       \\
        \sigma_{D,1} & \dots  & \sigma_{d,D} \\
    \end{pmatrix}_j
\): The full symetric positive covariance matrix of the gaussian \(j\).
* \(\eta_j\), the relative mass of the gaussian \(j\). We have \(\sum\limits_{j=1}^K \eta_j = 1\).
* \(\alpha_i\): The weights has the relative number of time the individual has been drawn.
* \(\omega_i\): The weights has the inverse of the relative covariance of each individual.
* \(
    \mathbf{x}_i =
    \begin{pmatrix}
        x_1    \\
        \vdots \\
        x_D    \\
    \end{pmatrix}_j
\): The observation \(i\).

Calculus Terminology
--------------------
* \(
    \mathcal{N}_{\mathbf{\mu}_j, \mathbf{\Sigma}_j}(\mathbf{x}_i) =
    \frac
    {
        e^{
            -\frac{1}{2}
            (\mathbf{x}_i-\mathbf{\mu}_j)^\intercal
            \mathbf{\Sigma}_j^{-1}
            (\mathbf{x}_i-\mathbf{\mu}_j)
        }
    }
    {\sqrt{(2\pi)^D |\mathbf{\Sigma}_j}|}
\): The multidimensional gaussian probability density.
* \(
    \Gamma(\mathbf{x}_i) =
    \sum\limits_{j=1}^K \eta_j \mathcal{N}_{\mathbf{\mu}_j, \mathbf{\Sigma}_j}(\mathbf{x}_i)
\): The total probability density of the observation \(\mathbf{x}_i\).
"""
