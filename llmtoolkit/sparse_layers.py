import torch


def poweriter(input, p_buffer, q_buffer, iter):
    for i in range(iter):
        if i == iter - 1:
            p_buffer[0] = torch.linalg.qr(p_buffer[0]).Q
        q_buffer[0] = input @ p_buffer[0]
        if i == iter - 1:
            q_buffer[0] = torch.linalg.qr(q_buffer[0]).Q
        p_buffer[0] = input.permute((0, 1, 3, 2)) @ q_buffer[0]
    # return q_buffer[0] @ p_buffer[0].permute((0, 1, 3, 2))
    return p_buffer, q_buffer


def make_poweriter():
    pass


class PowerLayer:
    def __init__():
        pass


input = torch.rand([64, 32, 512, 512]).requires_grad_()
p_buffer = torch.rand([64, 32, 512, 4])
q_buffer = torch.rand([64, 32, 512, 4])
for i in range(10):
    output = poweriter(input, [p_buffer], [q_buffer], 1)
