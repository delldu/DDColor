/************************************************************************************
***
*** Copyright 2024 Dell(18588220928g@163.com), All Rights Reserved.
***
*** File Author: Dell, Tue 30 Jan 2024 11:52:34 PM CST
***
************************************************************************************/

#ifndef _GGML_NN_H_
#define _GGML_NN_H_

#include <ggml.h>
#include <vector> // std::vector

#pragma GCC diagnostic ignored "-Wformat-truncation"

typedef struct ggml_tensor ggml_tensor_t;
typedef struct ggml_context ggml_context_t;

ggml_tensor_t* ggml_nn_add(ggml_context_t *ctx, ggml_tensor_t *x, float value);
ggml_tensor_t* ggml_nn_grid_y(ggml_context_t *ctx, ggml_tensor_t *x, int h);
ggml_tensor_t* ggml_nn_grid_x(ggml_context_t *ctx, ggml_tensor_t *x, int w);

// ----------------------------------------------------------------------------------------------------------------------------------------
// https://pytorch.org/docs/stable/generated/torch.nn.Identity.html
ggml_tensor_t* ggml_nn_identity(ggml_context_t* ctx, ggml_tensor_t* x);

struct Identity {
    void create_weight_tensors(ggml_context_t* ctx)
    {
        GGML_UNUSED(ctx);
    }

    void setup_weight_names(const char* prefix)
    {
        GGML_UNUSED(prefix);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        return ggml_nn_identity(ctx, x);
    }
};


// ----------------------------------------------------------------------------------------------------------------------------------------
// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
// class torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)[source]

ggml_tensor_t* ggml_nn_linear(ggml_context_t* ctx, ggml_tensor_t* x, ggml_tensor_t* w, ggml_tensor_t* b);

struct Linear {
    int64_t in_features;
    int64_t out_features;
    bool has_bias = true; // Fixed default

    ggml_tensor_t* weight;
    ggml_tensor_t* bias = NULL;

    void create_weight_tensors(ggml_context_t* ctx, ggml_type wtype = GGML_TYPE_Q8_0)
    {
        weight = ggml_new_tensor_2d(ctx, wtype, in_features, out_features);
        if (has_bias) {
            bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_features);
        }
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(weight, "%s%s", prefix, "weight");
        if (has_bias) {
            ggml_format_name(bias, "%s%s", prefix, "bias");
        }
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        return ggml_nn_linear(ctx, x, weight, bias);
    }
};

// ----------------------------------------------------------------------------------------------------------------------------------------
// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
// class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
//      padding_mode='zeros', device=None, dtype=None)[source]

ggml_tensor_t* ggml_nn_conv_2d(ggml_context_t* ctx, ggml_tensor_t * x, ggml_tensor_t * w,
    ggml_tensor_t * b, int s0 /*=1*/, int s1 /*=1*/, int p0 /*=0*/, int p1 /*=0*/, int d0 /*=1*/, int d1 /*=1*/,
    bool is_depthwise);

struct Conv2d {
    int64_t in_channels;
    int64_t out_channels;
    std::pair<int, int> kernel_size;

    // Fixed defaults ...
    std::pair<int, int> stride = { 1, 1 };
    std::pair<int, int> padding = { 0, 0 };
    std::pair<int, int> dilation = { 1, 1 };
    bool is_depthwise = false;
    bool has_bias = true;

    ggml_tensor_t* weight;
    ggml_tensor_t* bias = NULL;

    void create_weight_tensors(ggml_context_t* ctx, ggml_type wtype=GGML_TYPE_F16)
    {
        if (is_depthwise) {
            weight = ggml_new_tensor_4d(ctx, wtype, kernel_size.second, kernel_size.first, 1 /*in_channels*/, out_channels);
        } else {
            weight = ggml_new_tensor_4d(ctx, wtype, kernel_size.second, kernel_size.first, in_channels, out_channels);
        }
        if (has_bias) {
            bias = ggml_new_tensor_1d(ctx, (wtype == GGML_TYPE_Q8_0)? GGML_TYPE_F16 : GGML_TYPE_F32, out_channels);
        }
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(weight, "%s%s", prefix, "weight");
        if (has_bias) {
            ggml_format_name(bias, "%s%s", prefix, "bias");
        }
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        return ggml_nn_conv_2d(ctx, x, weight, bias, stride.second, stride.first, padding.second, padding.first,
            dilation.second, dilation.first, is_depthwise);
    }
};


// ----------------------------------------------------------------------------------------------------------------------------------------
// https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
// class torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None)[source]

ggml_tensor_t* ggml_nn_layer_norm(ggml_context_t* ctx, ggml_tensor_t* x, ggml_tensor_t* w, ggml_tensor_t* b, float eps);

struct LayerNorm {
    int64_t normalized_shape;
    float eps = 1e-5; // Fixed default values

    ggml_tensor_t* w;
    ggml_tensor_t* b;

    void create_weight_tensors(ggml_context_t* ctx)
    {
        w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape);
        b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape);
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(w, "%s%s", prefix, "weight");
        ggml_format_name(b, "%s%s", prefix, "bias");
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        return ggml_nn_layer_norm(ctx, x, w, b, eps);
    }
};

// -----------------------------------------------------------
// https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
// class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)

ggml_tensor_t* ggml_nn_batch_norm2d(ggml_context_t* ctx, ggml_tensor_t* x, ggml_tensor_t* w, ggml_tensor_t* b, 
    ggml_tensor_t* mean, ggml_tensor_t* var, float eps);


struct BatchNorm2d {
    int64_t num_features;
    float eps = 1e-5; // Fixed default values

    ggml_tensor_t* w;
    ggml_tensor_t* b;
    ggml_tensor_t* running_mean;
    ggml_tensor_t* running_var;
    ggml_tensor_t* num_batches_tracked;

    void create_weight_tensors(ggml_context_t* ctx)
    {
        w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_features);
        b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_features);

        running_mean = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_features);
        running_var = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_features);
        num_batches_tracked = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(w, "%s%s", prefix, "weight");
        ggml_format_name(b, "%s%s", prefix, "bias");

        ggml_format_name(running_mean, "%s%s", prefix, "running_mean");
        ggml_format_name(running_var, "%s%s", prefix, "running_var");
        ggml_format_name(num_batches_tracked, "%s%s", prefix, "num_batches_tracked");
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        return ggml_nn_batch_norm2d(ctx, x, w, b, running_mean, running_var, eps);
    }
};

// ----------------------------------------------------------------------------------------------------------------------------------------
// https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
// class torch.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None)[source]

ggml_tensor_t* ggml_nn_group_norm(ggml_context_t* ctx, ggml_tensor_t* x, ggml_tensor_t* w, ggml_tensor_t* b, int num_groups);

struct GroupNorm {
    int num_groups = 32;
    int64_t num_channels;
    float eps = 1e-5; // Fixed default values

    ggml_tensor_t* weight;
    ggml_tensor_t* bias;

    void create_weight_tensors(ggml_context_t* ctx)
    {
        // norm use GGML_TYPE_F32 !!!
        weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
        bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(weight, "%s%s", prefix, "weight");
        ggml_format_name(bias, "%s%s", prefix, "bias");
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        return ggml_nn_group_norm(ctx, x, weight, bias, num_groups); // hardcoded eps === 1e-6 now
    }
};

// ----------------------------------------------------------------------------------------------------------------------------------------
// class torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, 
//      add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)[source]

ggml_tensor_t* ggml_nn_attention(ggml_context_t* ctx, ggml_tensor_t* q, ggml_tensor_t* k, ggml_tensor_t* v, bool mask);

#if 0
struct MultiheadAttention {
    int64_t embed_dim;
    int64_t n_head;
    bool bias = true;

    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;

    void create_weight_tensors(ggml_context_t* ctx, ggml_type wtype=GGML_TYPE_Q8_0)
    {
        q_proj.in_features = embed_dim;
        q_proj.out_features = embed_dim;
        q_proj.has_bias = bias;
        q_proj.create_weight_tensors(ctx, wtype);

        k_proj.in_features = embed_dim;
        k_proj.out_features = embed_dim;
        k_proj.has_bias = bias;
        k_proj.create_weight_tensors(ctx, wtype);

        v_proj.in_features = embed_dim;
        v_proj.out_features = embed_dim;
        v_proj.has_bias = bias;
        v_proj.create_weight_tensors(ctx, wtype);

        out_proj.in_features = embed_dim;
        out_proj.out_features = embed_dim;
        out_proj.has_bias = bias;
        out_proj.create_weight_tensors(ctx, wtype);
    }

    void setup_weight_names(const char* prefix)
    {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "q_proj.");
        q_proj.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "k_proj.");
        k_proj.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "v_proj.");
        v_proj.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "out_proj.");
        out_proj.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x, bool mask)
    {
        // embed_dim = 768, n_head = 12, bias = 1, mask = 1
        // x:    f32 [768, 77, 1, 1], 
        // embed_dim = 1280, n_head = 20, bias = 1, mask = 1
        // x:    f32 [1280, 77, 1, 1], 

        int64_t N = x->ne[2]; // ==> 1
        int64_t n_token = x->ne[1]; // ==> 77
        int64_t d_head = embed_dim / n_head; // ==> 64

        ggml_tensor_t* q = q_proj.forward(ctx, x);
        q = ggml_reshape_4d(ctx, q, d_head, n_head, n_token, N); // [N, n_token, n_head, d_head]
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [N, n_head, n_token, d_head]
        q = ggml_reshape_3d(ctx, q, d_head, n_token, n_head * N); // [N * n_head, n_token, d_head]

        ggml_tensor_t* k = k_proj.forward(ctx, x);
        k = ggml_reshape_4d(ctx, k, d_head, n_head, n_token, N); // [N, n_token, n_head, d_head]
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [N, n_head, n_token, d_head]
        k = ggml_reshape_3d(ctx, k, d_head, n_token, n_head * N); // [N * n_head, n_token, d_head]

        ggml_tensor_t* v = v_proj.forward(ctx, x);
        v = ggml_reshape_4d(ctx, v, d_head, n_head, n_token, N); // [N, n_token, n_head, d_head]
        v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [N, n_head, d_head, n_token]
        v = ggml_reshape_3d(ctx, v, n_token, d_head, n_head * N); // [N * n_head, d_head, n_token]

        ggml_tensor_t* kqv = ggml_nn_attention(ctx, q, k, v, mask); // [N * n_head, n_token, d_head]

        kqv = ggml_reshape_4d(ctx, kqv, d_head, n_token, n_head, N);
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)); // [N, n_token, n_head, d_head]
        x = ggml_reshape_3d(ctx, kqv, d_head * n_head, n_token, N); // [N, n_token, d_head * n_head]

        x = out_proj.forward(ctx, x); // [N, n_token, embed_dim]
        return x;
    }
};
#endif

// ----------------------------------------------------------------------------------------------------------------------------------------
// https://paperswithcode.com/method/pixelshuffle
// class torch.nn.PixelShuffle(upscale_factor)[source] -- convert x from (∗,C*r*2, H, W) to (∗, C, H*r, W*r)


struct PixelShuffle {
    int upscale_factor;

    void create_weight_tensors(ggml_context_t* ctx)
    {
        GGML_UNUSED(ctx);
    }

    void setup_weight_names(const char* prefix)
    {
        GGML_UNUSED(prefix);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        return ggml_shuffle(ctx, x, upscale_factor);
    }
};

// ----------------------------------------------------------------------------------------------------------------------------------------
// https://pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html
// class torch.nn.PixelUnshuffle(downscale_factor)[source] -- convert x from (B, C, H×r, W×r) to (B, C×r*r, H, W)

ggml_tensor_t* pixel_nn_unshuffle(ggml_context_t *ctx, ggml_tensor_t *x, int downscale_factor);

struct PixelUnshuffle {
    int downscale_factor;

    void create_weight_tensors(ggml_context_t* ctx)
    {
        GGML_UNUSED(ctx);
    }

    void setup_weight_names(const char* prefix)
    {
        GGML_UNUSED(prefix);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        return pixel_nn_unshuffle(ctx, x, downscale_factor);
    }
};

ggml_tensor_t* ggml_nn_normalize(ggml_context_t *ctx, ggml_tensor_t *x, ggml_tensor_t *mean, ggml_tensor_t *std);

// ----------------------------------------------------------------------------------------------------------------------------------------
// https://pytorch.org/docs/stable/generated/torch.mean.html
// torch.mean(input, dim, keepdim=False, *, dtype=None, out=None)

ggml_tensor_t* ggml_nn_mean(ggml_context_t *ctx, ggml_tensor_t *x, int dim);

struct Mean {
    int dim = 2; // mean on channel, keepdim == true

    void create_weight_tensors(ggml_context_t* ctx)
    {
        GGML_UNUSED(ctx);
    }

    void setup_weight_names(const char* prefix)
    {
        GGML_UNUSED(prefix);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        return ggml_nn_mean(ctx, x, dim);
    }
};

ggml_tensor_t* ggml_nn_slice(ggml_context_t *ctx, ggml_tensor_t *x, int dim, int start /*0*/, int stop /*x->ne[dim]*/, int step /*1*/);
std::vector<ggml_tensor_t *> ggml_nn_chunks(ggml_context_t *ctx, ggml_tensor_t *x, int dim, int k);
ggml_tensor_t* ggml_nn_mul_mat(ggml_context_t *ctx, ggml_tensor_t *a, ggml_tensor_t *b);

// ----------------------------------------------------------------------------------------------------------------------------------------
// https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
// class torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

ggml_tensor_t* ggml_nn_avgpool2d(ggml_context_t *ctx, ggml_tensor_t *x, int kernel_size, int stride_size);

struct AvgPool2d {
    int kernel_size = 2;
    int stride_size = 1;

    void create_weight_tensors(ggml_context_t* ctx)
    {
        GGML_UNUSED(ctx);
    }

    void setup_weight_names(const char* prefix)
    {
        GGML_UNUSED(prefix);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        return ggml_nn_avgpool2d(ctx, x, kernel_size, stride_size);
    }
};

#endif // _GGML_NN_H_

#ifdef GGML_NN_IMPLEMENTATION
ggml_tensor_t* ggml_nn_identity(ggml_context_t* ctx, ggml_tensor_t* x)
{
    return ggml_dup_inplace(ctx, x);
}

ggml_tensor_t* ggml_nn_conv_2d(ggml_context_t* ctx, ggml_tensor_t* x, ggml_tensor_t* w,
    ggml_tensor_t* b, int s0 /*=1*/, int s1 /*=1*/, int p0 /*=0*/, int p1 /*=0*/, int d0 /*=1*/, int d1 /*=1*/, 
    bool is_depthwise)
{
    x = (is_depthwise)? ggml_conv_depthwise_2d(ctx, w, x, s0, s1, p0, p1, d0, d1) : ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);

    if (b != NULL) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        x = ggml_add(ctx, x, b);
    }

    return x;
}

ggml_tensor_t* ggml_nn_layer_norm(ggml_context_t* ctx, ggml_tensor_t* x, ggml_tensor_t* w, ggml_tensor_t* b, float eps)
{
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

ggml_tensor_t* ggml_nn_batch_norm2d(ggml_context_t* ctx, ggml_tensor_t* x, ggml_tensor_t* w, ggml_tensor_t* b, 
    ggml_tensor_t* mean, ggml_tensor_t* var, float eps)
{
    int C = x->ne[2];
    w = ggml_repeat(ctx, ggml_reshape_4d(ctx, w, 1, 1, C, 1), x);
    b = ggml_repeat(ctx, ggml_reshape_4d(ctx, b, 1, 1, C, 1), x);
    mean = ggml_repeat(ctx, ggml_reshape_4d(ctx, mean, 1, 1, C, 1), x);
    var = ggml_repeat(ctx, ggml_reshape_4d(ctx, var, 1, 1, C, 1), x);

    // var += eps;
    var = ggml_nn_add(ctx, var, eps);
    var = ggml_sqrt(ctx, var);

    x = ggml_sub(ctx, x, mean);
    x = ggml_div(ctx, x, var);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);

    return x;
}

// q: [N * n_head, n_token, d_head]
// k: [N * n_head, n_k, d_head]
// v: [N * n_head, d_head, n_k]
// return: [N * n_head, n_token, d_head]
ggml_tensor_t* ggml_nn_attention(ggml_context_t* ctx, ggml_tensor_t* q, ggml_tensor_t* k, ggml_tensor_t* v, bool mask /* = false*/)
{
#if defined(SD_USE_FLASH_ATTENTION) && !defined(SD_USE_CUBLAS) && !defined(SD_USE_METAL)
    ggml_tensor_t* kqv = ggml_flash_attn(ctx, q, k, v, false); // [N * n_head, n_token, d_head]
#else
    float d_head = (float)q->ne[0];

    ggml_tensor_t* kq = ggml_mul_mat(ctx, k, q); // [N * n_head, n_token, n_k]
    kq = ggml_scale_inplace(ctx, kq, 1.0f / sqrt(d_head));
    if (mask) {
        kq = ggml_diag_mask_inf_inplace(ctx, kq, 0);
    }
    kq = ggml_soft_max_inplace(ctx, kq);

    ggml_tensor_t* kqv = ggml_mul_mat(ctx, v, kq); // [N * n_head, n_token, d_head]
#endif
    return kqv;
}

ggml_tensor_t* ggml_nn_group_norm(ggml_context_t* ctx, ggml_tensor_t* x, ggml_tensor_t* w, ggml_tensor_t* b, int num_groups)
{
    if (ggml_n_dims(x) >= 3) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], 1);
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    }

    x = ggml_group_norm(ctx, x, num_groups, 1e-6); // TODO: eps is hardcoded to 1e-6 for now
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

ggml_tensor_t* ggml_nn_linear(ggml_context_t* ctx, ggml_tensor_t* x, ggml_tensor_t* w, ggml_tensor_t* b)
{
    x = ggml_mul_mat(ctx, w, x);
    if (b != NULL) {
        x = ggml_add(ctx, x, b);
    }
    return x;
}


// convert x from (B, C, H×r, W×r) to (B, C×r*r, H, W)
ggml_tensor_t* pixel_nn_unshuffle(ggml_context_t *ctx, ggml_tensor_t *x, int downscale_factor)
{
    int C = x->ne[2]; // channel numbers
    int R = downscale_factor;

    ggml_tensor_t *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, R, R, C, C);
    x = ggml_im2col(ctx, a, x, R /*s0*/, R /*s1*/, 0 /*p0*/, 0 /*p1*/, 1 /*d0*/, 1 /*d1*/, true /*is_2D*/, GGML_TYPE_F32);
    x = ggml_permute(ctx, x, 2, 0, 1, 3); // from src index to dst: 0->2, 1->0, 2->1, 3->3
    x = ggml_cont(ctx, x); // !!! import !!!

    return x;
}

// mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) ==> mean = ggml_new_4d(ctx, GGML_TYPE_F32, 1, 1, 3, 1)
// std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) ==>  std = ggml_new_4d(ctx, GGML_TYPE_F32, 1, 1, 3, 1)
ggml_tensor_t* ggml_nn_normalize(ggml_context_t *ctx, ggml_tensor_t *x, ggml_tensor_t *mean, ggml_tensor_t *std)
{
    mean = ggml_repeat(ctx, mean, x);
    std = ggml_repeat(ctx, std, x);
    return ggml_div(ctx, ggml_sub(ctx, x, mean), std);
}


ggml_tensor_t* ggml_nn_mean(ggml_context_t *ctx, ggml_tensor_t *x, int dim)
{
    int dims[4] = {0, 1, 2, 3};

    for (int i = 0; i <= dim; i++) {
        dims[i] = (i < dim) ? i + 1: 0; // [0, 1, ..., dim] --> [1, ..., dim, 0]
    }
    x = ggml_cont(ctx, ggml_permute(ctx, x, dims[0], dims[1], dims[2], dims[3]));
    // ------------------------------------------------------------------------
    // ggml_mean(ctx, x); // mean on dim 0
    float m = (float)x->ne[0];
    x = ggml_sum_rows(ctx, x);
    x = ggml_scale(ctx, x, m);
    // ------------------------------------------------------------------------
    for (int i = 0; i <= dim; i++) {
        dims[i] = (i == 0) ? dim : i - 1; // [0, 1, ..., dim] --> [dim, 0, ..., dim - 1]
    }
    x = ggml_cont(ctx, ggml_permute(ctx, x, dims[0], dims[1], dims[2], dims[3]));

    return x;
}

ggml_tensor_t* ggml_nn_avgpool2d(ggml_context_t *ctx, ggml_tensor_t *x, int kernel_size, int stride_size)
{
    ggml_tensor_t *y;

    x = ggml_cont(ctx, ggml_pad(ctx, x, 1, 1, 0, 0)); // W+1, H+1, C, B
    x = ggml_cont(ctx, x);

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
    y = ggml_cont(ctx, ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, kernel_size, kernel_size, stride_size, stride_size, 1, 1));
    y = ggml_cont(ctx, ggml_reshape_4d(ctx, y, W + 1, H + 1, C, B));
    y = ggml_nn_slice(ctx, y, 0 /*dim*/, 1 /*start*/, W /*stop*/, 1 /*step*/);
    y = ggml_nn_slice(ctx, y, 1 /*dim*/, 1 /*start*/, H /*stop*/, 1 /*step*/);

    return y;
}

// ggml_tensor_t* ggml_nn_slice(ggml_context_t *ctx, ggml_tensor_t *x, int dim, int start /*0*/, int stop /*x->ne[dim]*/, int step /*1*/);
ggml_tensor_t* ggml_nn_slice(ggml_context_t *ctx, ggml_tensor_t *x, int dim, int start, int stop, int step)
{
    size_t starts[4] = {0, 0, 0, 0};
    size_t stops[4] = {0, 0, 0, 0};
    size_t steps[4] = {1, 1, 1, 1};
    size_t shapes[4] = {0, 0, 0, 0};
    size_t strides[4] = {0, 0, 0, 0};
    size_t offset = 0;

    starts[dim] = MAX(start, 0);
    for (int i = 0; i < 4; i++) {
        stops[i] = (size_t)x->ne[i];
    }
    stops[dim] = MIN(stop, stops[dim]); // x->ne[dim]
    steps[dim] = step;

    // ----------------------------------------------------------------------------------------
    for (int i = 0; i < 4; i++) {
        shapes[i] = (stops[i] - starts[i] + steps[i] - 1) / steps[i]; // shaps = (stop - step) // step
    }
    for (int i = 0; i < 4; i++) {
        strides[i] = (size_t)x->nb[i] * steps[i];
    }
    // offset = 0;
    for (int i = 0; i < 4; i++) {
        offset += (size_t)x->nb[i] * starts[i];
    }
    // ----------------------------------------------------------------------------------------

    // GGML_API struct ggml_tensor * ggml_view_4d(
    //         struct ggml_context * ctx,
    //         struct ggml_tensor  * a,
    //         int64_t               ne0,
    //         int64_t               ne1,
    //         int64_t               ne2,
    //         int64_t               ne3,
    //         size_t                nb1, // row   stride in bytes
    //         size_t                nb2, // slice stride in bytes
    //         size_t                nb3,
    //         size_t                offset);
    return ggml_view_4d(ctx, x,
            (int64_t)shapes[0], (int64_t)shapes[1], (int64_t)shapes[2], (int64_t)shapes[3],
            strides[1], strides[2], strides[3],
            offset);
}


std::vector<ggml_tensor_t *> ggml_nn_chunks(ggml_context_t *ctx, ggml_tensor_t *x, int dim, int k)
{
    int B = x->ne[dim];
    int S = (B + k - 1) / k;
    std::vector<ggml_tensor_t *> chunks;

    for (int i = 0; i < k; i++) {
        ggml_tensor_t *c = ggml_nn_slice(ctx, x, dim, i * S, (i + 1)*S, 1);
        chunks.push_back(c);
    }

    return chunks;
}

// def ggml_nn_mul_mat(ctx, g_a, g_b):
//     # a = torch.randn(2, 10, 3, 4)
//     # b = torch.randn(2, 10, 4, 5)
//     # g_a -- 4, 3, 10, 2
//     # g_b -- 5, 4, 10, 2 --> 4, 5, 10, 2
//     # ==> g_c -- 5, 3, 10, 2
//     g_b = ggml.ggml_cont(ctx, ggml.ggml_permute(ctx, g_b, 1, 0, 2, 3))
//     g_y = ggml.ggml_mul_mat(ctx, g_b, g_a)
//     return g_y

ggml_tensor_t* ggml_nn_mul_mat(ggml_context_t *ctx, ggml_tensor_t *a, ggml_tensor_t *b)
{
    b = ggml_cont(ctx, ggml_permute(ctx, b, 1, 0, 2, 3));
    return ggml_mul_mat(ctx, b, a);
}


ggml_tensor_t* ggml_nn_add(ggml_context_t *ctx, ggml_tensor_t *x, float value)
{
    ggml_tensor_t *a = ggml_dup(ctx, x);
    a = ggml_clamp(ctx, a, value, value); // a = value

    return ggml_add(ctx, x, a);
}

ggml_tensor_t* ggml_nn_grid_y(ggml_context_t *ctx, ggml_tensor_t *x, int h)
{
    int w = x->ne[0];
    ggml_tensor_t *a = ggml_new_tensor_2d(ctx,  GGML_TYPE_F32, w, h); // a -- shape template
    ggml_tensor_t *y = ggml_repeat(ctx, x, a); // shape like a
    return ggml_cont(ctx, y);
}

ggml_tensor_t* ggml_nn_grid_x(ggml_context_t *ctx, ggml_tensor_t *x, int w)
{
    ggml_tensor_t *y = ggml_nn_grid_y(ctx, x, w);
    y = ggml_transpose(ctx, y);
    return ggml_cont(ctx, y);
}


#endif // GGML_NN_IMPLEMENTATION