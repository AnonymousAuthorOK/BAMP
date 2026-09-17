import time
import torch
from torch import nn
from torch.autograd import Function
import logging
import numpy as np

import dreamplace.configure as configure

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.center_aware_macro_placement.center_aware_macro_placement_cuda as center_aware_macro_placement_cuda

logger = logging.getLogger(__name__)

class CenterAwareMacroPlacementFunction(Function):
    @staticmethod
    def forward(ctx, pos, gamma, xl, xh, yl, yh, num_nodes, num_movable_nodes, node_size_x, node_size_y, movable_macro_mask):
        tt = time.time()
        if pos.is_cuda:
            func = center_aware_macro_placement_cuda.forward
        else:
            # func = center_aware_macro_placement_cpp.forward
            print("no cpp version")
            exit()
        output = func(pos.view(pos.numel()), gamma, xl, xh, yl, yh, num_nodes, num_movable_nodes, node_size_x, node_size_y, movable_macro_mask)
        ctx.pos = pos
        ctx.gamma = gamma
        ctx.xl = xl
        ctx.xh = xh
        ctx.yl = yl
        ctx.yh = yh
        ctx.num_nodes = num_nodes
        ctx.num_movable_nodes = num_movable_nodes
        ctx.node_size_x = node_size_x
        ctx.node_size_y = node_size_y
        ctx.movable_macro_mask = movable_macro_mask
        if pos.is_cuda:
            torch.cuda.synchronize()
        logger.debug("center aware macro placement forward %.3f ms" %
                     ((time.time() - tt) * 1000))
        return output[0]
    
    @staticmethod
    def backward(ctx, grad):
        tt = time.time()
        if grad.is_cuda:
            func = center_aware_macro_placement_cuda.backward
        else:
            # func = center_aware_macro_placement_cpp.backward
            print("no cpp version")
            exit()
        output = func(grad, ctx.pos, ctx.gamma, ctx.xl, ctx.xh, ctx.yl, ctx.yh, ctx.num_nodes, ctx.num_movable_nodes, ctx.node_size_x, ctx.node_size_y, ctx.movable_macro_mask)
        if grad.is_cuda:
            torch.cuda.synchronize()
        logger.debug("center aware macro placement backward %.3f ms" %
                     ((time.time() - tt) * 1000))
        return output, None, None, None, None, None ,None, None, None, None, None
    
class CenterAwareMacroPlacement(nn.Module):
    def __init__(self, xl, xh, yl, yh, num_nodes, num_movable_nodes,node_size_x, node_size_y, movable_macro_mask):
        super(CenterAwareMacroPlacement, self).__init__()
        self.xl = xl
        self.xh = xh
        self.yl = yl
        self.yh = yh
        self.num_nodes = num_nodes
        self.num_movable_nodes = num_movable_nodes
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.movable_macro_mask = movable_macro_mask
    
    def forward(self, pos, gamma):
        return CenterAwareMacroPlacementFunction.apply(pos,
                                                       gamma,
                                                       self.xl,
                                                       self.xh,
                                                       self.yl,
                                                       self.yh,
                                                       self.num_nodes,
                                                       self.num_movable_nodes,
                                                       self.node_size_x,
                                                       self.node_size_y,
                                                       self.movable_macro_mask
                                                       )
    