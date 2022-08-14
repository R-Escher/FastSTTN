import torch
import torch.nn as nn
from torch.autograd.function import Function


class ReversibleBlock(nn.Module):
    def __init__(self, g, f): 
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, x, **kwargs):
        # input's channel dim is cut in half (256 to 128)
        x1, x2 = torch.chunk(x, 2, dim=1) 
        
        y1, y2 = None, None

        with torch.no_grad():
            # this is where the MultiHeadAttention forward is called (it is called ONLY here and during backprop in function below)
            y1 = x1 + self.g(x2, **kwargs)
            y2 = x2 + self.f(x=y1)

        return torch.cat([y1, y2], dim=1)

    def backward_pass(self, y, dy, **kwargs):
        y1, y2 = torch.chunk(y, 2, dim=1)
        del y
        assert (not y1.requires_grad), "y1 must already be detached" # from revnet
        assert (not y2.requires_grad), "y2 must already be detached" # from revnet

        dy1, dy2 = torch.chunk(dy, 2, dim=1)
        del dy
        assert (not dy1.requires_grad), "dy1 must not require grad" # from revnet
        assert (not dy2.requires_grad), "dy2 must not require grad" # from revnet

        with torch.enable_grad():
            y1.requires_grad_(requires_grad = True)
            fy1 = self.f(y1)
            torch.autograd.backward(fy1, dy2)

        with torch.no_grad():
            x2 = y2 - fy1 # Restore first input of forward()
            del y2, fy1

            # The gradient of x1 is the sum of the gradient of the output
            # y1 as well as the gradient that flows back through G
            # (The gradient that flows back through G is stored in y1.grad)
            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad_(requires_grad = True)
            gx2 = self.g(x=x2, **kwargs)
            torch.autograd.backward(gx2, dx1)

        with torch.no_grad():
            x1 = y1 - gx2
            del y1, gx2

            # The gradient of x2 is the sum of the gradient of the output
            # y2 as well as the gradient that flows back through F
            # (The gradient that flows back through F is stored in x2.grad)
            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            # Undo the channelwise split
            x = torch.cat([x1, x2.detach()], dim=1)
            dx = torch.cat([dx1, dx2], dim=1)

        return x, dx


class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        kwargs = ctx.kwargs
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None


class ReversibleSequence(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList([ReversibleBlock(TBlocks.attention, TBlocks.feed_forward) for TBlocks in blocks])

    def forward(self, x, **kwargs):
        return _ReversibleFunction.apply(x, self.blocks, kwargs)

