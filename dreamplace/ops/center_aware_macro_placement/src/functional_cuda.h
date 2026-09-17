#ifndef GPUPLACE_CENTER_AWARE_WA_FUNCTIONAL_H
#define GPUPLACE_CENTER_AWARE_WA_FUNCTIONAL_H

#include <iostream>
#include <cmath>
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

// Forward kernel: Global reduction version for WA-HPWL inverse loss (x+y direction)
template <typename T>
__global__ void centerAwareMacroPlacementKernel(
    const T *pos_x, const T *pos_y,
    const T xl, const T xh, const T yl, const T yh,
    const int num_movable_nodes,
    const T *node_size_x, const T *node_size_y,
    const bool *movable_macro_mask,
    const T gamma,
    T *partial_loss)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_movable_nodes)
        return;

    if (!movable_macro_mask[i])
    {
        partial_loss[i] = 0;
        return;
    }

    T node_cx = pos_x[i] + node_size_x[i] / static_cast<T>(2.0);
    T node_cy = pos_y[i] + node_size_y[i] / static_cast<T>(2.0);

    T d_l = node_cx - xl;
    T d_r = xh - node_cx;
    T d_b = node_cy - yl;
    T d_t = yh - node_cy;

    T d_l2 = d_l * d_l;
    T d_r2 = d_r * d_r;
    T d_b2 = d_b * d_b;
    T d_t2 = d_t * d_t;

    T min_loss = d_l2;
    int dir = 0;
    if (d_r2 < min_loss)
    {
        min_loss = d_r2;
        dir = 1;
    }
    if (d_b2 < min_loss)
    {
        min_loss = d_b2;
        dir = 2;
    }
    if (d_t2 < min_loss)
    {
        min_loss = d_t2;
        dir = 3;
    }

    partial_loss[i] = min_loss;
}

// Backward kernel
template <typename T>
__global__ void centerAwareMacroPlacementBackwardKernel(
    const T *grad_tensor,
    const T *pos_x, const T *pos_y,
    const T xl, const T xh, const T yl, const T yh,
    const int num_movable_nodes,
    const T *node_size_x, const T *node_size_y,
    const bool *movable_macro_mask,
    T gamma,
    T *grad_x, T *grad_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_movable_nodes)
        return;

    if (!movable_macro_mask[i])
    {
        grad_x[i] = 0;
        grad_y[i] = 0;
        return;
    }

    T node_cx = pos_x[i] + node_size_x[i] / static_cast<T>(2.0);
    T node_cy = pos_y[i] + node_size_y[i] / static_cast<T>(2.0);

    T d_l = node_cx - xl;
    T d_r = xh - node_cx;
    T d_b = node_cy - yl;
    T d_t = yh - node_cy;

    T d_l2 = d_l * d_l;
    T d_r2 = d_r * d_r;
    T d_b2 = d_b * d_b;
    T d_t2 = d_t * d_t;

    T min_loss = d_l2;
    int dir = 0;
    if (d_r2 < min_loss)
    {
        min_loss = d_r2;
        dir = 1;
    }
    if (d_b2 < min_loss)
    {
        min_loss = d_b2;
        dir = 2;
    }
    if (d_t2 < min_loss)
    {
        min_loss = d_t2;
        dir = 3;
    }

    T upstream_grad = *grad_tensor;
    T grad_scale = 2 * upstream_grad;

    grad_x[i] = 0;
    grad_y[i] = 0;
    if (dir == 0)
        grad_x[i] = grad_scale * d_l;
    else if (dir == 1)
        grad_x[i] = -grad_scale * d_r;
    else if (dir == 2)
        grad_y[i] = grad_scale * d_b;
    else if (dir == 3)
        grad_y[i] = -grad_scale * d_t;
}

// Reduction kernel
template <typename T>
__global__ void reducePartialLossKernel(const T *partial_loss, int n, T *total_loss)
{
    __shared__ T shared[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    T val = 0;
    if (i < n)
        val = partial_loss[i];
    shared[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(total_loss, shared[0]);
}

DREAMPLACE_END_NAMESPACE
#endif
