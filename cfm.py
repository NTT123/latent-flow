import torch
from torchdiffeq import odeint


class CFM:
    """Conditional Flow Matching"""

    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.MSELoss()

    def training_losses(self, flow, x_0, x_1, t, model_kwargs):
        """
        x: (B, T, C)
        t: (B,)
        """
        # unsqueeze t to (B, 1, 1)
        t_unsqueezed = t.reshape(*([-1] + [1] * (x_1.ndim - 1)))
        x_t = (1 - t_unsqueezed) * x_0 + t_unsqueezed * x_1
        dx_t = x_1 - x_0
        loss = self.loss_fn(flow(t=t, x_t=x_t, **model_kwargs), dx_t)
        return {"loss": loss}

    @torch.no_grad()
    def solve_ode(
        self,
        flow,
        x0,
        time_grid: torch.Tensor,
        method: str = "dopri5",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        return_intermediates: bool = False,
        device: torch.device = None,
    ):

        def ode_func(t, x):
            B = x.shape[0]
            # repeat interleaved t
            t_interleaved = torch.repeat_interleave(t[None], B, dim=0)
            return flow(x_t=x, t=t_interleaved)

        ode_opts = {}
        time_grid = time_grid.to(device)

        sol = odeint(
            ode_func,
            x0,
            time_grid,
            method=method,
            options=ode_opts,
            atol=atol,
            rtol=rtol,
        )

        if return_intermediates:
            return sol
        else:
            return sol[-1]


def create_cfm():
    return CFM()
