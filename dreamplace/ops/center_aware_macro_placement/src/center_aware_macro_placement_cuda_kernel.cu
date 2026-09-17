#include <cfloat>
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"
#include "center_aware_macro_placement/src/functional_cuda.h"

DREAMPLACE_BEGIN_NAMESPACE

// Launcher function

template <typename T>
int centerAwareMacroPlacementCudaLauncher(
    const T *pos_x, const T *pos_y,
    const T gamma,
    const T xl, const T xh, const T yl, const T yh,
    int num_movable_nodes,
    const T *node_size_x, const T *node_size_y,
    const bool *movable_macro_mask,
    T *total_distance_from_center)
{
    const int threads_per_block = 256;
    const int num_blocks = (num_movable_nodes + threads_per_block - 1) / threads_per_block;

    // 分配 GPU 中间 buffer
    T *partial_loss;
    if (cudaMalloc(&partial_loss, sizeof(T) * num_movable_nodes) != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed\n";
        return -1;
    }

    cudaMemset(total_distance_from_center, 0, sizeof(T));

    centerAwareMacroPlacementKernel<<<num_blocks, threads_per_block>>>(
        pos_x, pos_y,
        xl, xh, yl, yh,
        num_movable_nodes,
        node_size_x, node_size_y,
        movable_macro_mask,
        gamma,
        partial_loss);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << "\n";
        cudaFree(partial_loss);
        return -1;
    }

    // 调用归约函数
    reducePartialLossKernel<<<num_blocks, threads_per_block>>>(
        partial_loss, num_movable_nodes, total_distance_from_center);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Reduction kernel error: " << cudaGetErrorString(err) << "\n";
        cudaFree(partial_loss);
        return -1;
    }

    cudaFree(partial_loss);
    return 0;
}

template <typename T>
int centerAwareMacroPlacementBackwardLauncher(
    const T *grad_tensor,
    const T *pos_x, const T *pos_y,
    const T gamma,
    const T xl, const T xh, const T yl, const T yh,
    int num_movable_nodes,
    const T *node_size_x, const T *node_size_y,
    const bool *movable_macro_mask,
    T *grad_x_tensor, T *grad_y_tensor)
{
    const int threads_per_block = 256;
    const int num_blocks = (num_movable_nodes + threads_per_block - 1) / threads_per_block;

    centerAwareMacroPlacementBackwardKernel<<<num_blocks, threads_per_block>>>(
        grad_tensor,
        pos_x, pos_y,
        xl, xh, yl, yh,
        num_movable_nodes,
        node_size_x, node_size_y,
        movable_macro_mask,
        gamma,
        grad_x_tensor, grad_y_tensor);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Backward kernel launch error: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T)                        \
    template int centerAwareMacroPlacementCudaLauncher<T>( \
        const T *pos_x, const T *pos_y,                    \
        const T gamma,                                     \
        const T xl, const T xh, const T yl, const T yh,    \
        int num_movable_nodes,                             \
        const T *node_size_x, const T *node_size_y,        \
        const bool *movable_macro_mask,                    \
        T *total_distance_from_center);
REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

#define REGISTER_KERNEL_LAUNCHER(T)                            \
    template int centerAwareMacroPlacementBackwardLauncher<T>( \
        const T *grad_tensor,                                  \
        const T *pos_x, const T *pos_y,                        \
        const T gamma,                                         \
        const T xl, const T xh, const T yl, const T yh,        \
        int num_movable_nodes,                                 \
        const T *node_size_x, const T *node_size_y,            \
        const bool *movable_macro_mask,                        \
        T *grad_x_tensor, T *grad_y_tensor);
REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE