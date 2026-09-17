#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int centerAwareMacroPlacementCudaLauncher(
    const T *pos_x, const T *pos_y,
    const T gamma,
    const T xl, const T xh, const T yl, const T yh,
    const int num_movable_nodes,
    const T *node_size_x, const T *node_size_y,
    const bool *movable_macro_mask,
    T *total_distance_from_center);
template <typename T>
int centerAwareMacroPlacementBackwardLauncher(
    const T *grad_tensor,
    const T *pos_x, const T *pos_y,
    const T gamma,
    const T xl, const T xh, const T yl, const T yh,
    const int num_movable_nodes,
    const T *node_size_x, const T *node_size_y,
    const bool *movable_macro_mask,
    T *grad_x_tensor, T *grad_y_tensor);

std::vector<at::Tensor> center_aware_macro_placement_forward(
    at::Tensor pos,
    double gamma,
    double xl,
    double xh,
    double yl,
    double yh,
    int num_nodes,
    int num_movable_nodes,
    at::Tensor node_size_x,
    at::Tensor node_size_y,
    at::Tensor movable_macro_mask)
{
    CHECK_FLAT_CUDA(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    CHECK_FLAT_CUDA(movable_macro_mask);
    CHECK_CONTIGUOUS(movable_macro_mask);

    at::Tensor total_distance_from_center = at::zeros({}, pos.options());

    DREAMPLACE_DISPATCH_FLOATING_TYPES(
        pos, "centerAwareMacroPlacementCudaLauncher", [&]
        {
        // template <typename T>
        centerAwareMacroPlacementCudaLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
            gamma,
            xl, xh, yl, yh,
            num_movable_nodes,
            DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(movable_macro_mask, bool),
            // forward output total_distance_from_center
            DREAMPLACE_TENSOR_DATA_PTR(total_distance_from_center, scalar_t)
        ); });

    return {total_distance_from_center};
}

at::Tensor center_aware_macro_placement_backward(
    at::Tensor grad,
    at::Tensor pos,
    double gamma,
    double xl,
    double xh,
    double yl,
    double yh,
    int num_nodes,
    int num_movable_nodes,
    at::Tensor node_size_x,
    at::Tensor node_size_y,
    at::Tensor movable_macro_mask)
{
    CHECK_FLAT_CUDA(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    CHECK_FLAT_CUDA(movable_macro_mask);
    CHECK_CONTIGUOUS(movable_macro_mask);
    at::Tensor grad_out = at::zeros_like(pos);

    DREAMPLACE_DISPATCH_FLOATING_TYPES(
        pos, "centerAwareMacroPlacementBackwardLauncher", [&]
        {
        // template <typename T>
        centerAwareMacroPlacementBackwardLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(grad, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
            gamma,
            xl, xh, yl, yh,
            num_movable_nodes,
            DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(movable_macro_mask, bool),
            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + num_nodes
          ); });
    return grad_out;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &DREAMPLACE_NAMESPACE::center_aware_macro_placement_forward,
          "CenterAwareMacroPlacement forward (CUDA)");
    m.def("backward", &DREAMPLACE_NAMESPACE::center_aware_macro_placement_backward,
          "CenterAwareMacroPlacement backward (CUDA)");
}
