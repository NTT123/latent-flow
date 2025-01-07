# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from cfm import create_cfm


def sample(
    flow,
    decoder,
    x0,
    input_size=8,
    in_channels=128,
    device="cuda",
):
    cfm = create_cfm()
    # create torch Generator object and use it to generate x0
    generator = torch.Generator(device=device)
    # generator.manual_seed(seed)
    # x0 = torch.randn(
    #     n, input_size, input_size, in_channels, generator=generator, device=device
    # )
    time_grid = torch.linspace(0, 1, 2).to(device)
    x = cfm.solve_ode(
        flow.forward,
        x0,
        device=device,
        time_grid=time_grid,
        method="dopri5",
    )
    # decode to image
    x = x.permute(0, 3, 1, 2)
    x = decoder(x)
    return x
