#include "ddcolor.h"

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h>
#define GGML_NN_IMPLEMENTATION
#include <ggml_nn.h>

ggml_tensor_t* ggml_nn_avgpool2d(ggml_context_t *ctx, ggml_tensor_t *x, int kernel_size, int stride_size)
{
    x = ggml_pad(ctx, x, 1, 1, 0, 0); // W+1, H+1, C, B

    int W = (int)x->ne[0];
    int H = (int)x->ne[1];
    int C = (int)x->ne[2];
    int B = (int)x->ne[3];
    x = ggml_cont(ctx, ggml_reshape_3d(ctx, x, W, H, C*B));

    // GGML_API struct ggml_tensor * ggml_pool_2d(
    //         struct ggml_context * ctx,
    //         struct ggml_tensor  * a,
    //         enum ggml_op_pool     op,
    //         int                   k0,
    //         int                   k1,
    //         int                   s0,
    //         int                   s1,
    //         float                 p0,
    //         float                 p1);

    ggml_tensor_t *y = ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, kernel_size, kernel_size, stride_size, stride_size, 1, 1);
    y = ggml_cont(ctx, ggml_reshape_4d(ctx, y, W + 1, H + 1, C, B));

    y = ggml_nn_slice(ctx, y, 0 /*dim*/, 1 /*start*/, W /*stop*/, 1 /*step*/);
    y = ggml_nn_slice(ctx, y, 1 /*dim*/, 1 /*start*/, H /*stop*/, 1 /*step*/);

    return y;
}

