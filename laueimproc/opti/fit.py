#!/usr/bin/env python3

"""Implement the optimal gradient decrease."""

import torch


def find_optimal_step(
    step: torch.Tensor,
    cost_dif: torch.Tensor,
    grad: torch.Tensor,
    *,
    _check: bool = True
):
    r"""Estimate the best step to minimize the cost, based on order 2 estimation.

    Parameters
    ----------
    step : torch.Tensor
        \(\overrightarrow{ab}\) is the difference
        between the previous \(\overrightarrow{ob}\)
        and the current \(\overrightarrow{oa}\) position of shape (..., *s).
    cost_dif : torch.Tensor
        \(\gamma\) is the difference of the scalar loss value
        of the previous and current step of shape (...,).
    grad : torch.Tensor
        The current gradient \(\overrightarrow{g}\) value of shape (..., *s)
        at the position \(\overrightarrow{ob}\).

    Returns
    -------
    delta : torch.Tensor
        An estimation of the optimal step \(\delta\) of shape (...,) such as:
        \(\begin{cases}
            f = \alpha.\lambda^2 + \beta.\lambda + cst \\
            f(-|\overrightarrow{ab}|) - f(0) = \gamma \\
            f^{\prime}(0)
            = <\frac{\overrightarrow{ab}}{|\overrightarrow{ab}|}, \overrightarrow{g}> \\
            f^{\prime}(\delta.|\overrightarrow{g}|) = 0;
        \end{cases}\)
        then \(\overrightarrow{oc} = \overrightarrow{ob} - \delta.\overrightarrow{g}\)

    Examples
    --------
    >>> import torch
    >>> from laueimproc.opti.fit import find_optimal_step
    >>> loss = lambda xy: xy[..., 0]**2+xy[..., 1]**2/2
    >>> prev_pos, curr_pos = torch.tensor([1.1, 1.1]), torch.tensor([1.0, 1.0])
    >>> prev_cost, curr_cost = loss(prev_pos), loss(curr_pos)
    >>> for _ in range(10):
    ...     print(f"loss {curr_cost}")
    ...     grad = torch.func.grad(loss)(curr_pos)
    ...     delta = find_optimal_step(prev_pos-curr_pos, prev_cost-curr_cost, grad)
    ...     prev_pos, curr_pos = curr_pos, curr_pos - delta*grad
    ...     prev_cost, curr_cost = curr_cost, loss(curr_pos)
    ...
    loss 1.5
    loss 0.1377229541540146
    loss 0.015718786045908928
    loss 0.00806969590485096
    loss 2.0100125766475685e-05
    loss 1.1283106687187683e-05
    loss 2.8506252647275687e-07
    loss 2.648626207246707e-07
    loss 3.2647382575134998e-09
    loss 3.1178359893857532e-09
    >>>
    """
    # import sympy
    # gamma, ab, gp, gn = sympy.symbols("gamma ab gproj gnorm", real=True)
    # alpha, beta, cst, delta = sympy.symbols("alpha beta cst delta", real=True)
    # eq1 = sympy.Eq((alpha*(-ab)**2 + beta*(-ab) + cst) - cst, gamma)
    # eq2 = sympy.Eq(beta, gp)
    # eq3 = sympy.Eq(2*alpha*(delta*gn) + beta, 0)
    # sympy.pprint(eq1)
    # sympy.pprint(eq2)
    # sympy.pprint(eq3)
    # sympy.pprint(sympy.solve([eq1, eq2, eq3], [delta, alpha, beta], dict=True).pop())

    if _check:
        assert isinstance(step, torch.Tensor), step.__class__.__name__
        assert isinstance(cost_dif, torch.Tensor), cost_dif.__class__.__name__
        assert isinstance(grad, torch.Tensor), grad.__class__.__name__
        assert step.shape == grad.shape
        assert step.shape[:len(cost_dif.shape)] == cost_dif.shape

    step_norm_square = torch.sum(step*step, dim=-1)   # (...,)
    step_norm = torch.sqrt(step_norm_square)  # (...,)
    grad_norm_square = torch.sum(grad*grad, dim=-1)  # (...,)
    grad_norm = torch.sqrt(grad_norm_square)   # (...,)

    beta = -grad_norm_square
    beta *= torch.abs(torch.sum(grad * step, dim=-1) / (step_norm * grad_norm))  # angular deviation
    alpha = (cost_dif*grad_norm_square + beta*grad_norm*step_norm) / step_norm_square
    alpha += torch.sign(alpha) * 1e-14  # correction factor for cabbages
    delta = -beta / (2*alpha)

    delta = torch.nan_to_num(delta, nan=1e-2, out=delta)
    delta = torch.abs(delta)
    # delta[alpha <= 0] = 1e-2  # if we jumped over the hill
    return delta
