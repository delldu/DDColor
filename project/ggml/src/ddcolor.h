#ifndef __DDCOLOR__H__
#define __DDCOLOR__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>

#pragma GCC diagnostic ignored "-Wformat-truncation"

// def custom_conv_layer(ni, nf,
//     ks = 3,
//     stride = 1,
//     use_activ = True, # True | False
//     extra_bn = False, # True || False
// ):
//     padding = (ks - 1) // 2
//     conv = nn.Conv2d(ni, nf, kernel_size=ks, bias=not extra_bn, stride=stride, padding=padding)
//     conv = nn.utils.spectral_norm(conv)
//     layers = [conv]

//     if use_activ: # True | False
//         layers.append(nn.ReLU(True))
//     if extra_bn: # True | False
//         layers.append(nn.BatchNorm2d(nf))

//     return nn.Sequential(*layers)

struct CustomConv2d {
    int ni;
    int nf;
    int ks = 3;
    bool use_activ = true; // Fixed default ...
    bool extra_bn = false; // Fixed default ...

    struct Conv2d conv;
    struct BatchNorm2d bn;

    void create_weight_tensors(ggml_context_t* ctx) {
        conv.in_channels = ni;
        conv.out_channels = nf;
        conv.kernel_size = {ks, ks};
        // conv.stride = { 1, 1 };
        conv.padding = { (ks - 1)/2, (ks - 1)/2 };
        // conv.dilation = { 1, 1 };
        // conv.is_depthwise = false;
        conv.has_bias = extra_bn ? false: true;
        conv.create_weight_tensors(ctx, GGML_TYPE_F32);

        if (extra_bn) {
            bn.num_features = nf;
            bn.create_weight_tensors(ctx);
        }
    }

    // (conv): Sequential(
    //   (0): Conv2d(448, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    //   (1): ReLU(inplace=True)
    //   (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    // )
    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "0.");
        conv.setup_weight_names(s);

        if (extra_bn) {
            if (use_activ) {
                snprintf(s, sizeof(s), "%s%s", prefix, "2."); // Add ReLU ...
            } else {
                snprintf(s, sizeof(s), "%s%s", prefix, "1.");
            }
            bn.setup_weight_names(s);
        }
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
        x = conv.forward(ctx, x);
        if (use_activ) {
            x = ggml_relu_inplace(ctx, x);
        }

        if (extra_bn) {
            x = bn.forward(ctx, x);
        }
        return x;
    }
};


struct MLP {
    const int num_layers = 3;
    const int color_embed_dim = 256;
    struct Linear layers[3];

    void create_weight_tensors(ggml_context_t* ctx) {
        GGML_ASSERT(ARRAY_SIZE(layers) == num_layers);
        for (int i = 0; i < num_layers; i++) {
            layers[i].in_features = color_embed_dim; // 256;
            layers[i].out_features = color_embed_dim; // 256;
            layers[i].create_weight_tensors(ctx, GGML_TYPE_F32);
        }
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        for (int i = 0; i < num_layers; i++) { // num_layers -- 3
            snprintf(s, sizeof(s), "%slayers.%d.", prefix, i);
            layers[i].setup_weight_names(s);
        }
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
        x = layers[0].forward(ctx, x);
        x = ggml_relu_inplace(ctx, x);

        x = layers[1].forward(ctx, x);
        x = ggml_relu_inplace(ctx, x);

        x = layers[2].forward(ctx, x);

    	return x;
    }
};

struct FFNLayer {
    int d_model = 256;
    int dim_feedforward = 2048;

    // network params
    struct Linear linear1;
    struct Linear linear2;
    struct LayerNorm norm;

    void create_weight_tensors(ggml_context_t* ctx) {
        linear1.in_features = d_model;
        linear1.out_features = dim_feedforward;
        linear1.create_weight_tensors(ctx, GGML_TYPE_F32);

        linear2.in_features = dim_feedforward;
        linear2.out_features = d_model;
        linear2.create_weight_tensors(ctx, GGML_TYPE_F32);

        norm.normalized_shape = d_model;
        norm.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "linear1.");
        linear1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "linear2.");
        linear2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm.");
        norm.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
        // h = self.linear2(F.relu(self.linear1(tgt)))
        auto h = linear1.forward(ctx, x);
        h = ggml_relu_inplace(ctx, h);
        h = linear2.forward(ctx, h);

        x = ggml_add(ctx, x, h); // x = x + h
        x = norm.forward(ctx, x);
    	return x;
    }
};

struct MultiheadAttention {
    // network hparams
    int embed_dim = 256;
    int num_heads = 8;

    struct Linear in_proj;
    struct Linear out_proj;

    void create_weight_tensors(ggml_context_t* ctx) {
        in_proj.in_features = embed_dim;
        in_proj.out_features = 3 * embed_dim;
        in_proj.create_weight_tensors(ctx, GGML_TYPE_F32);

        out_proj.in_features = embed_dim;
        out_proj.out_features = embed_dim;
        out_proj.create_weight_tensors(ctx, GGML_TYPE_F32);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "in_proj_");
        in_proj.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "out_proj.");
        out_proj.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* query, ggml_tensor_t* key, ggml_tensor_t* value) {
        // # (Pdb) query.size() -- [100, 1, 256]
        // # (Pdb) key.size() -- [1024, 1, 256]
        // # (Pdb) value.size() -- [1024, 1, 256]
        GGML_ASSERT(query->ne[0] == 100); // num_queries -- 100
        GGML_ASSERT(key->ne[0] == 1024);
        GGML_ASSERT(value->ne[0] == 1024);

        // in_proj.weight -- [512, 1536]
        // in_proj.bias -- [1536]
        GGML_ASSERT(in_proj.weight->ne[1] == 1536);
        GGML_ASSERT(in_proj.bias->ne[0] == 1536);

        int head_dim = embed_dim / num_heads; // 32
        std::vector<ggml_tensor_t *> w_qkv = ggml_nn_chunks(ctx, in_proj.weight, 1, 3);
        std::vector<ggml_tensor_t *> b_qkv = ggml_nn_chunks(ctx, in_proj.bias, 1, 3);
        ggml_tensor_t *w_q = w_qkv[0];
        ggml_tensor_t *w_k = w_qkv[1];
        ggml_tensor_t *w_v = w_qkv[2];
        ggml_tensor_t *b_q = b_qkv[0];
        ggml_tensor_t *b_k = b_qkv[1];
        ggml_tensor_t *b_v = b_qkv[2];

        ggml_tensor_t *g_q = ggml_nn_linear(ctx, query, w_q, b_q);
        g_q = ggml_cont(ctx, ggml_reshape_4d(ctx, g_q, 32, 8, 100, 1)); // head_dim -- 32
        g_q = ggml_cont(ctx, ggml_permute(ctx, g_q, 0, 2, 1, 3));
        ggml_tensor_t *g_q_scaled = ggml_scale(ctx, g_q, 1.0/sqrtf(32)); // head_dim -- 32
        // # ggml_shape("g_q_scaled", g_q_scaled) # [32, 1024, 8, 1] 

        ggml_tensor_t *g_k = ggml_nn_linear(ctx, key, w_k, b_k);
        g_k = ggml_cont(ctx, ggml_reshape_4d(ctx, g_k, 32, 8, 1024, 1)); // head_dim -- 32
        g_k = ggml_cont(ctx, ggml_permute(ctx, g_k, 0, 2, 1, 3));
        g_k = ggml_cont(ctx, ggml_transpose(ctx, g_k)); //  0, 1
        // ggml_shape("g_k", g_k) # g_k shape:  [1024, 32, 8, 1]

        ggml_tensor_t *g_v = ggml_nn_linear(ctx, value, w_v, b_v);
        g_v = ggml_cont(ctx, ggml_reshape_4d(ctx, g_v, 32, 8, 1024, 1)); // head_dim -- 32
        g_v = ggml_cont(ctx, ggml_permute(ctx, g_v, 0, 2, 1, 3));


        ggml_tensor_t *attn_output_weights = ggml_nn_mul_mat(ctx, g_q_scaled, g_k);
        // ggml_shape("attn_output_weights", attn_output_weights) # [1024, 100, 8, 1]

        // # tensor [attn_output_weights] size: [8, 100, 1024], min: -6.5267, max: 8.720252, mean: 0.013913
        attn_output_weights = ggml_soft_max(ctx, attn_output_weights);
        // ggml_shape("attn_output_weights soft_max", attn_output_weights) # [1024, 100, 8, 1]

        // # attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        // # [8, 100, 32] ==> [100, 8, 32] ==> [100, 256]
        ggml_tensor_t *attn_output = ggml_nn_mul_mat(ctx, attn_output_weights, g_v);
        // ggml_shape("g->attn_output", attn_output) # ggml.shape: [32, 100, 8, 1], torch.shape: [8,100,32]

        attn_output = ggml_cont(ctx, ggml_permute(ctx, attn_output, 0, 2, 1, 3));
        attn_output = ggml_cont(ctx, ggml_reshape_2d(ctx, attn_output, 256, 100));

        // ggml_shape("g->attn_output ==> ", attn_output) # ggml.shape: [256, 100, 1, 1]
        // # 0.003356, max: 0.003356, mean: 0.003356

        // # ********************************************************************************
        // # attn_output shape:  [256, 100, 1, 1]
        // # out_proj.weight shape:  [256, 256, 1, 1]
        // # out_proj.bias shape:  [256, 1, 1, 1]
        // # attn_output shape:  [256, 100, 1, 1]
        // # ********************************************************************************
        attn_output = ggml_nn_linear(ctx, attn_output, out_proj.weight, out_proj.bias);
        attn_output = ggml_cont(ctx, ggml_reshape_3d(ctx, attn_output, 256, 1, 100));
        // ggml_shape("attn_output ==> ", attn_output)
        // # tensor [attn_output] size: [100, 1, 256], min: -2.726733, max: 2.047928, mean: -0.011761

        return attn_output;
    }
};


struct CrossAttentionLayer {
    // network hparams
    int d_model = 256;
    int nhead = 8;

    // network params
    struct MultiheadAttention multihead_attn;
    struct LayerNorm norm;

    ggml_tensor_t* norm_weight;  // torch.float32, [256] 
    ggml_tensor_t* norm_bias;  // torch.float32, [256]

    void create_weight_tensors(ggml_context_t* ctx) {
        multihead_attn.embed_dim = d_model;
        multihead_attn.num_heads = nhead;
        multihead_attn.create_weight_tensors(ctx);

        norm.normalized_shape = d_model;
        // norm.eps = 1e-5;
        norm.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "multihead_attn.");
        multihead_attn.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm.");
        norm.setup_weight_names(s);
    }


    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* tgt, ggml_tensor_t* memory, ggml_tensor_t* pos, ggml_tensor_t* query_pos) {
        ggml_tensor_t *tgt2 = multihead_attn.forward(ctx, 
                ggml_add(ctx, tgt, query_pos),
                ggml_add(ctx, memory, pos),
                memory
            );
        tgt = ggml_add(ctx, tgt, tgt2);
        return norm.forward(ctx, tgt);
    }
};



struct SelfAttentionLayer {
    // network hparams
    int d_model = 256;
    int nhead = 8;

    // network params
    struct MultiheadAttention self_attn;
    struct LayerNorm norm;

    void create_weight_tensors(ggml_context_t* ctx) {
        self_attn.embed_dim = d_model;
        self_attn.num_heads = nhead;
        self_attn.create_weight_tensors(ctx);

        norm.normalized_shape = d_model;
        // norm.eps = 1e-5;
        norm.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "self_attn.");
        self_attn.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm.");
        norm.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* tgt, ggml_tensor_t *query_pos) {
    	// please implement forward by your self, please !!!
        ggml_tensor_t *q = ggml_add(ctx, tgt, query_pos);
        ggml_tensor_t *k = ggml_add(ctx, tgt, query_pos);
        ggml_tensor_t *tgt2 = self_attn.forward(ctx, q, k, tgt);
        tgt = ggml_add(ctx, tgt, tgt2);
        tgt = norm.forward(ctx, tgt);

    	return tgt;
    }
};

static void y_embed_map(struct ggml_tensor * dst , const struct ggml_tensor * src, int ith, int nth, void * userdata)
{
    GGML_ASSERT(userdata == NULL);
    // GGML_ASSERT(ggml_are_same_shape(dst, src));
    // GGML_ASSERT(ggml_is_contiguous(dst));
    // GGML_ASSERT(ggml_is_contiguous(src));

    int N = (int)dst->ne[0];
    int W = (int)dst->ne[1];
    int H = (int)dst->ne[2];
    int B = (int)dst->ne[3];
    int dr = (B + nth - 1)/nth;
    int start = ith * dr;
    int stop = MIN(start + dr, B);

    for (int b_i = 0; b_i < B; b_i++) {
        for (int n_i = 0; n_i < N; n_i++) {
            for (int w_i = 0; w_i < W; w_i++) {
                for (int h_i = 0; h_i < H; h_i++) {
                    ggml_set_f32_nd(dst, n_i, w_i, h_i, b_i, (1.0 + w_i)/H*2.0*M_PI);
                } // h_i
            } // w_i
        } // n_i
    } // b_i
}

static void x_embed_map(struct ggml_tensor * dst , const struct ggml_tensor * src, int ith, int nth, void * userdata)
{
    // N, W, H, B = tensor_out.contents.ne[:4]
    // dr = (B + nth - 1)//nth
    // start = ith * dr
    // stop = min(start + dr, B)

    // print(f"x_embed_map: N = {N}, W = {W}, H = {H}, B = {B}")
    // ggml_shape("tensor_out", tensor_out)

    // # ggml.ggml_set_f32(tensor_out, 2.0)
    // for b_i in range(B): # range(start, stop):
    //     for n_i in range(N):
    //         for w_i in range(W):
    //             for h_i in range(H):
    //                 # ggml.ggml_set_f32_nd(tensor_out, n_i, w_i, h_i, b_i, (1.0 + w_i)/W*2.0*math.pi)
    //                 ggml.ggml_set_f32_nd(tensor_out, n_i, w_i, h_i, b_i, (1.0 + w_i)/W*2.0*math.pi)


    GGML_ASSERT(userdata == NULL);
    // GGML_ASSERT(ggml_are_same_shape(dst, src));
    // GGML_ASSERT(ggml_is_contiguous(dst));
    // GGML_ASSERT(ggml_is_contiguous(src));

    int N = (int)dst->ne[0];
    int W = (int)dst->ne[1];
    int H = (int)dst->ne[2];
    int B = (int)dst->ne[3];
    int dr = (B + nth - 1)/nth;
    int start = ith * dr;
    int stop = MIN(start + dr, B);

    for (int b_i = 0; b_i < B; b_i++) {
        for (int n_i = 0; n_i < N; n_i++) {
            for (int w_i = 0; w_i < W; w_i++) {
                for (int h_i = 0; h_i < H; h_i++) {
                    ggml_set_f32_nd(dst, n_i, w_i, h_i, b_i, (1.0 + w_i)/W*2.0*M_PI);
                } // h_i
            } // w_i
        } // n_i
    } // b_i
}

static void sin_cos_map(struct ggml_tensor * dst , const struct ggml_tensor * src, int ith, int nth, void * userdata)
{
    GGML_ASSERT(userdata == NULL);
    GGML_ASSERT(ggml_are_same_shape(dst, src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src));

    int N = (int)dst->ne[0];
    int W = (int)dst->ne[1];
    int H = (int)dst->ne[2];
    int B = (int)dst->ne[3];
    int dr = (B + nth - 1)/nth;
    int start = ith * dr;
    int stop = MIN(start + dr, B);

    for (int b_i = 0; b_i < B; b_i++) {
        for (int w_i = 0; w_i < W; w_i++) {
            for (int h_i = 0; h_i < H; h_i++) {
                for (int n_i = 0; n_i < N/2; n_i++) {
                    float sin_v = ggml_get_f32_nd(src, 2 * n_i + 0, w_i, h_i, b_i);
                    float cos_v = ggml_get_f32_nd(src, 2 * n_i + 1, w_i, h_i, b_i);
                    // ---------------------------------------------------------------------------
                    ggml_set_f32_nd(dst, 2*n_i + 0, w_i, h_i, b_i, sinf(sin_v));
                    ggml_set_f32_nd(dst, 2*n_i + 1, w_i, h_i, b_i, cosf(cos_v));
                }
            }
        }
    }
}

struct PositionEmbedding {
    // network hparams
    const float scale = 6.283185307179586; // 2 * math.pi
    const int num_pos_feats = 128;

    // network params
    ggml_tensor_t *dim_t;

    void create_weight_tensors(ggml_context_t* ctx) {
        dim_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 1, num_pos_feats);
    }

    void setup_weight_names(const char *prefix) {
        // decoder.color_decoder.pe_layer.dim_t -- 128,     1,     1,     1
        ggml_format_name(dim_t, "%s%s", prefix, "dim_t");
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
        int W = x->ne[0];
        int H = x->ne[1];
        int B = x->ne[3];

        ggml_tensor_t *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, num_pos_feats, W, H, B);

        ggml_tensor_t *y_embed = ggml_map_custom1(ctx, a, y_embed_map, GGML_N_TASKS_MAX, NULL);
        ggml_tensor_t *x_embed = ggml_map_custom1(ctx, a, x_embed_map, GGML_N_TASKS_MAX, NULL);

        ggml_tensor_t *g_dim_t = ggml_repeat(ctx, dim_t, y_embed);

        ggml_tensor_t *pos_y = ggml_div(ctx, y_embed, g_dim_t);
        pos_y = ggml_map_custom1(ctx, pos_y, sin_cos_map, GGML_N_TASKS_MAX, NULL);

        ggml_tensor_t *pos_x = ggml_div(ctx, x_embed, g_dim_t);
        pos_x = ggml_map_custom1(ctx, pos_x, sin_cos_map, GGML_N_TASKS_MAX, NULL);

        ggml_tensor_t *g_pos_embed = ggml_concat(ctx, pos_y, pos_x, 0); 
        g_pos_embed = ggml_cont(ctx, ggml_permute(ctx, g_pos_embed, 2, 0, 1, 3));
        // # ggml.shape: [256, 32, 32, 2] --> [32, 32, 256, 2]

    	return g_pos_embed;
    }
};


// def test_ggml_einsum():
//     a = torch.randn(1, 100, 256)
//     b = torch.randn(1, 256, 512, 512)
//     einsum_x = torch.einsum("bqc,bchw->bqhw", a, b) # [1, 100, 512, 512]

//     m_a = a.reshape(100, 256)
//     m_b = b.reshape(256, 512*512)

//     ctx = ggml_new_ctx(mem=1024)
//     # --------------------------------------------------------------------------------------
//     g_a = ggml_tensor(ctx, m_a) # ggml_shape("g_a", g_a)
//     g_b = ggml_tensor(ctx, m_b) # ggml_shape("g_b", g_b)

//     g_c = ggml_nn_mul_mat(ctx, g_a, g_b)
//     g_c = ggml.ggml_reshape_4d(ctx, g_c, 512, 512, 100, 1) # ggml_shape("g_c", g_c)

//     ggml_compute(ctx, g_c)
//     # --------------------------------------------------------------------------------------

static ggml_tensor_t* ggml_nn_einsum(ggml_context_t *ctx, ggml_tensor_t *a, ggml_tensor_t *b)
{
    //     a = torch.randn(1, 100, 256)
    //     b = torch.randn(1, 256, 512, 512)
    //     einsum_x = torch.einsum("bqc,bchw->bqhw", a, b) # [1, 100, 512, 512]
    GGML_ASSERT(a->ne[0] == 256 && a->ne[1] == 100 && a->ne[2] == 1);
    GGML_ASSERT(b->ne[0] == 512 && b->ne[1] == 512 && b->ne[2] == 256 && b->ne[3] == 1);

    a = ggml_reshape_2d(ctx, a, 256, 100 * 1); // [256, 100, 1] --> [256, 100]
    b = ggml_reshape_2d(ctx, b, 512*512, 256*1); // [512, 512, 256, 1] --> [512*512, 256*1]
    ggml_tensor_t *c = ggml_nn_mul_mat(ctx, a, b);
    c = ggml_cont(ctx, ggml_reshape_4d(ctx, c, 512, 512, 100, 1));

    return c;
}


struct MultiScaleColorDecoder {
    // network hparams
    int hidden_dim = 256;
    int num_queries = 100;
    int nheads = 8;
    int dim_feedforward = 2048;
    int num_scales = 3;
    const int num_feature_levels = 3;
    const int dec_layers = 9;

    // network params
    struct PositionEmbedding pe_layer;
    struct SelfAttentionLayer transformer_self_attention_layers[9]; // dec_layers
    struct CrossAttentionLayer transformer_cross_attention_layers[9]; // dec_layers
    struct FFNLayer transformer_ffn_layers[9]; // dec_layers

    struct LayerNorm decoder_norm;

    ggml_tensor_t* query_feat_weight;  // torch.float32, [100, 256] 
    ggml_tensor_t* query_embed_weight;  // torch.float32, [100, 256] 
    ggml_tensor_t* level_embed_weight;  // torch.float32, [3, 256] 

    // struct Conv2d input_proj_0;
    // struct Conv2d input_proj_1;
    // struct Conv2d input_proj_2;
    struct Conv2d input_proj[3];

    struct MLP color_embed;


    void create_weight_tensors(ggml_context_t* ctx) {
        GGML_ASSERT(ARRAY_SIZE(transformer_self_attention_layers) == dec_layers);
        GGML_ASSERT(ARRAY_SIZE(transformer_cross_attention_layers) == dec_layers);
        GGML_ASSERT(ARRAY_SIZE(transformer_ffn_layers) == dec_layers);

        pe_layer.create_weight_tensors(ctx);

        for (int i = 0; i < dec_layers; i++) {
            transformer_self_attention_layers[i].d_model = hidden_dim;
            transformer_self_attention_layers[i].nhead = nheads;
            transformer_self_attention_layers[i].create_weight_tensors(ctx);

            transformer_cross_attention_layers[i].d_model = hidden_dim;
            transformer_cross_attention_layers[i].nhead = nheads;
            transformer_cross_attention_layers[i].create_weight_tensors(ctx);

            transformer_ffn_layers[i].d_model = hidden_dim;
            transformer_ffn_layers[i].dim_feedforward = dim_feedforward;
            transformer_ffn_layers[i].create_weight_tensors(ctx);
        }

        decoder_norm.normalized_shape = hidden_dim;
        // decoder_norm.eps = 1e-5;
        decoder_norm.create_weight_tensors(ctx);

        query_feat_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 100);
        query_embed_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 100);
        level_embed_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 3); // num_feature_levels

        //  in_channels=[512, 512, 256],
        int in_channels[3] = {512, 512, 256};
        for (int i = 0; i < 3; i++) {
            input_proj[i].in_channels = in_channels[i];
            input_proj[i].out_channels = 256;
            input_proj[i].kernel_size = {1, 1};
            input_proj[i].create_weight_tensors(ctx, GGML_TYPE_F32);
        }

        // input_proj_0.in_channels = 512;
        // input_proj_0.out_channels = 256;
        // input_proj_0.kernel_size = {1, 1};
        // input_proj_0.create_weight_tensors(ctx, GGML_TYPE_F32);

        // input_proj_1.in_channels = 512;
        // input_proj_1.out_channels = 256;
        // input_proj_1.kernel_size = {1, 1};
        // input_proj_1.create_weight_tensors(ctx, GGML_TYPE_F32);

        // input_proj_2.in_channels = 256;
        // input_proj_2.out_channels = 256;
        // input_proj_2.kernel_size = {1, 1};
        // input_proj_2.create_weight_tensors(ctx, GGML_TYPE_F32);

        color_embed.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "pe_layer.");
        pe_layer.setup_weight_names(s);

        for (int i = 0; i < dec_layers; i++) {
            // snprintf(s, sizeof(s), "%s%s", prefix, "transformer_self_attention_layers.0.");
            // transformer_self_attention_layers_0.setup_weight_names(s);

            snprintf(s, sizeof(s), "%stransformer_self_attention_layers.%d.", prefix, i);
            transformer_self_attention_layers[i].setup_weight_names(s);

            // snprintf(s, sizeof(s), "%s%s", prefix, "transformer_cross_attention_layers.0.");
            // transformer_cross_attention_layers_0.setup_weight_names(s);
            snprintf(s, sizeof(s), "%stransformer_cross_attention_layers.%d.", prefix, i);
            transformer_cross_attention_layers[i].setup_weight_names(s);

            // snprintf(s, sizeof(s), "%s%s", prefix, "transformer_ffn_layers.0.");
            // transformer_ffn_layers_0.setup_weight_names(s);

            snprintf(s, sizeof(s), "%stransformer_ffn_layers.%d.", prefix, i);
            transformer_ffn_layers[i].setup_weight_names(s);
        }

        snprintf(s, sizeof(s), "%s%s", prefix, "decoder_norm.");
        decoder_norm.setup_weight_names(s);

        ggml_format_name(query_feat_weight, "%s%s", prefix, "query_feat.weight");
        ggml_format_name(query_embed_weight, "%s%s", prefix, "query_embed.weight");
        ggml_format_name(level_embed_weight, "%s%s", prefix, "level_embed.weight");

        for (int i = 0; i < 3; i++) {
            // snprintf(s, sizeof(s), "%s%s", prefix, "input_proj.0.");
            // input_proj_0.setup_weight_names(s);

            snprintf(s, sizeof(s), "%sinput_proj.%d.", prefix, i);
            input_proj[i].setup_weight_names(s);
        }

        // snprintf(s, sizeof(s), "%s%s", prefix, "input_proj.0.");
        // input_proj_0.setup_weight_names(s);
        // snprintf(s, sizeof(s), "%s%s", prefix, "input_proj.1.");
        // input_proj_1.setup_weight_names(s);
        // snprintf(s, sizeof(s), "%s%s", prefix, "input_proj.2.");
        // input_proj_2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "color_embed.");
        color_embed.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* out0, ggml_tensor_t* out1, ggml_tensor_t* out2, ggml_tensor_t* out3) {
        // xxxx_debug
    	return out0;
    }
};


struct CustomPixelShuffle {
    // network hparams
    int ni = 256;
    int nf = 512;
    int scale = 2;
    bool extra_bn = false;
    
    struct CustomConv2d conv;
    struct PixelShuffle shuf;
    struct AvgPool2d blur;

    void create_weight_tensors(ggml_context_t* ctx) {
        // self.conv = custom_conv_layer(ni, nf * (scale**2), ks=1, use_activ=False, extra_bn=extra_bn)
        conv.ni = ni;
        conv.nf = nf * (scale * scale);
        conv.ks = 1;
        conv.use_activ = false;
        conv.extra_bn = extra_bn;
        conv.create_weight_tensors(ctx);

        shuf.upscale_factor = scale;
        shuf.create_weight_tensors(ctx);

        blur.kernel_size = 2;
        blur.stride_size = 1;
        blur.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "shuf.");
        shuf.setup_weight_names(s);
    }

    //     self.pad = nn.ReplicationPad2d((1, 0, 1, 0)) # left = 1, right = 0, top = 1, bottom = 0
    //     self.blur = nn.AvgPool2d(2, stride=1)
    //     self.relu = nn.ReLU(True)

    // def forward(self, x):
    //     x = self.shuf(self.relu(self.conv(x)))
    //     return self.blur(self.pad(x))

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
        x = conv.forward(ctx, x);
        x = ggml_relu_inplace(ctx, x);
        x = shuf.forward(ctx, x);

        x = ggml_pad(ctx, x, 1, 1, 0, 0); // W+1, H+1, C, B
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
        x = ggml_nn_avgpool2d(ctx, x, 2 /*kernel_size*/, 1 /*stride_size*/);

        // xxxx_debug, pad(x) --> blur(x)
        x = blur.forward(ctx, x);
    	return x;
    }
};


struct UnetBlockWide {
    // network hparams
    
    int up_in_c = 1536;
    int x_in_c = 768;
    int n_out = 512;

    // shuf
    struct CustomPixelShuffle shuf;
    struct BatchNorm2d bn;
    struct CustomConv2d conv;

    void create_weight_tensors(ggml_context_t* ctx) {
        shuf.ni = up_in_c;
        shuf.nf = n_out;
        shuf.scale = 2;
        shuf.extra_bn = true;
        shuf.create_weight_tensors(ctx);

        bn.num_features = x_in_c;
        bn.eps = 1e-5;
        bn.create_weight_tensors(ctx);

        // ni = n_out + x_in_c # 1280
        // self.conv = custom_conv_layer(ni, n_out, ks =3, extra_bn=True)
        conv.ni = n_out + x_in_c;
        conv.nf = n_out;
        conv.ks = 3;
        // conv.use_activ = true;
        conv.extra_bn = true;
        conv.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "shuf.");
        shuf.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "bn.");
        bn.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);
    }

    // def forward(self, up_in, s):
    //     # tensor [up_in] size: [1, 1536, 16, 16], min: -26.544884, max: 16.26297, mean: -0.00155
    //     # tensor [s] size: [1, 768, 32, 32], min: -6.887635, max: 24.325577, mean: 0.001135
    //     up_out = self.shuf(up_in)
    //     # tensor [up_out] size: [1, 512, 32, 32], min: 0.0, max: 4.052812, mean: 0.314582

    //     cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
    //     # tensor [cat_x] size: [1, 1280, 32, 32], min: 0.0, max: 8.217813, mean: 0.350186
    //     # tensor [self.conv(cat_x)] size: [1, 512, 32, 32], min: -2.069555, max: 33.400291, mean: -0.

    //     # --------------------------------------------------------------------------------------------
    //     # ggml_debug
    //     # self.bn -- BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    //     # nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
    //     # self.bn.state_dict() is dict:
    //     # tensor [weight] size: [768], min: 0.780468, max: 1.366361, mean: 0.912081
    //     # tensor [bias] size: [768], min: -0.136402, max: 0.466912, mean: 0.102002
    //     # tensor [running_mean] size: [768], min: -0.962075, max: 16.190048, mean: 0.001125
    //     # tensor [running_var] size: [768], min: 0.0012, max: 51.826649, mean: 0.180806
    //     # tensor [num_batches_tracked] size: [], min: 310000.0, max: 310000.0, mean: 310000.0
    //     # tensor [s] size: [1, 768, 32, 32], min: -6.887635, max: 24.325577, mean: 0.001135
    //     # tensor [self.bn(s)] size: [1, 768, 32, 32], min: -8.827082, max: 8.217813, mean: 0.092955
    //     # --------------------------------------------------------------------------------------------
    //     return self.conv(cat_x)

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* up_in, ggml_tensor_t* s) {
        auto up_out = shuf.forward(ctx, up_in);
        auto cat_x = ggml_concat(ctx, up_out, bn.forward(ctx, s), 2); // dim on channel
        cat_x = ggml_relu_inplace(ctx, cat_x);

    	return conv.forward(ctx, cat_x);
    }
};

struct Decoder {
    // network hparams
    int nf = 512;
    int num_queries = 100;
    int num_scales = 3;
    int dec_layers = 9;

    // network params
    struct UnetBlockWide layers_0;
    struct UnetBlockWide layers_1;
    struct UnetBlockWide layers_2;

    struct CustomPixelShuffle last_shuf;
    struct MultiScaleColorDecoder color_decoder;

    void create_weight_tensors(ggml_context_t* ctx) {
        // m.append(UnetBlockWide(1536, 768, 512))
        layers_0.up_in_c = 1536;
        layers_0.x_in_c = 768;
        layers_0.n_out = 512;
        layers_0.create_weight_tensors(ctx);

        // m.append(UnetBlockWide(512, 384, 512))
        layers_1.up_in_c = 512;
        layers_1.x_in_c = 384;
        layers_1.n_out = 512;
        layers_1.create_weight_tensors(ctx);

        // m.append(UnetBlockWide(512, 192, 256))
        layers_2.up_in_c = 512;
        layers_2.x_in_c = 192;
        layers_2.n_out = 256;
        layers_2.create_weight_tensors(ctx);

        // embed_dim = nf // 2 # --> 256
        // self.last_shuf = CustomPixelShuffle(embed_dim, embed_dim, scale=4)
        last_shuf.ni = nf / 2;
        last_shuf.nf = nf / 2;
        last_shuf.scale = 4;
        last_shuf.extra_bn = false;
        last_shuf.create_weight_tensors(ctx);

        color_decoder.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[512];
        snprintf(s, sizeof(s), "%s%s", prefix, "layers.0.");
        layers_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layers.1.");
        layers_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layers.2.");
        layers_2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "last_shuf.");
        last_shuf.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "color_decoder.");
        color_decoder.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, std::vector<ggml_tensor_t *> encoder_output_layers) {
    	// please implement forward by your self, please !!!
        // (x0, x1, x2, x3) = encoder_output_layers
        // out0 = self.layer_output(0, x3, x2)
        // # tensor [out0] size: [1, 512, 32, 32], min: -2.069555, max: 33.400291, mean: -0.006212

        // out1 = self.layer_output(1, out0, x1)
        // # tensor [out1] size: [1, 512, 64, 64], min: -2.505532, max: 12.87324, mean: -0.048077

        // out2 = self.layer_output(2, out1, x0)
        // # tensor [out2] size: [1, 256, 128, 128], min: -2.547425, max: 12.310425, mean: -0.095538

        // out3 = self.last_shuf(out2) 
        // # tensor [out3] size: [1, 256, 512, 512], min: 0.0, max: 0.285963, mean: 0.009585

        // out = self.color_decoder([out0, out1, out2], out3)
        // # tensor [out] size: [1, 100, 512, 512], min: -14.656635, max: 24.051313, mean: 1.814844

        // return out


        GGML_ASSERT(encoder_output_layers.size() == 4);
        ggml_tensor_t* x0 = encoder_output_layers[0];
        ggml_tensor_t* x1 = encoder_output_layers[0];
        ggml_tensor_t* x2 = encoder_output_layers[0];
        ggml_tensor_t* x3 = encoder_output_layers[0];

        auto out0 = layers_0.forward(ctx, x3, x2);
        auto out1 = layers_0.forward(ctx, out0, x1);
        auto out2 = layers_0.forward(ctx, out1, x0);
        auto out3 = last_shuf.forward(ctx, out2);
        auto out = color_decoder.forward(ctx, out0, out1, out2, out3);

    	return out;
    }
};


struct Block {
    int dim = 1536;

    // network params
    struct Conv2d dwconv;
    struct LayerNorm norm; // LayerNormChannelsLast
    struct Linear pwconv1;
    struct Linear pwconv2;
    ggml_tensor_t *gamma;

    void create_weight_tensors(ggml_context_t* ctx) {
        // kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)
        dwconv.in_channels = dim;
        dwconv.out_channels = dim;
        dwconv.kernel_size = {7, 7};
        dwconv.padding = {3, 3};
        dwconv.is_depthwise = true;
        dwconv.create_weight_tensors(ctx, GGML_TYPE_F32);

        norm.normalized_shape = dim;
        norm.eps = 1e-6;
        norm.create_weight_tensors(ctx);

        pwconv1.in_features = dim;
        pwconv1.out_features = 4*dim;
        pwconv1.create_weight_tensors(ctx, GGML_TYPE_F32);

        pwconv2.in_features = 4*dim;
        pwconv2.out_features = dim;
        pwconv2.create_weight_tensors(ctx, GGML_TYPE_F32);

        gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "dwconv.");
        dwconv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm.");
        norm.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "pwconv1.");
        pwconv1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "pwconv2.");
        pwconv2.setup_weight_names(s);

        ggml_format_name(gamma, "%s%s", prefix, "gamma");
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
        //         input = x
        //         x = self.dwconv(x)
        //         x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        //         x = self.norm(x)
        //         x = self.pwconv1(x)
        //         x = self.act(x)
        //         x = self.pwconv2(x)

        //         x = self.gamma * x
        //         x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        //         x = x + input
        //         return x
        auto input = x;

        x = dwconv.forward(ctx, x);
        x = ggml_permute(ctx, x, 0, 2, 3, 1); // ???

        x = norm.forward(ctx, x);
        x = pwconv1.forward(ctx, x);
        x = ggml_relu_inplace(ctx, x);
        x = pwconv2.forward(ctx, x);

        x = ggml_mul_mat(ctx, gamma, x);
        x = ggml_permute(ctx, x, 0, 3, 1, 2); // ???

        x = ggml_add(ctx, x, input);
    	return x;
    }
};

struct LayerNormChannelsFirst {
    // network hparams
    int normalized_shape = 192;
    float eps = 1e-06;

    // network params
    ggml_tensor_t *w;
    ggml_tensor_t *b;

    void create_weight_tensors(ggml_context_t* ctx) {
        w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape);
        b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape);
    }

    void setup_weight_names(const char *prefix) {
        // char s[GGML_MAX_NAME];
        ggml_format_name(w, "%s%s", prefix, "weight");
        ggml_format_name(b, "%s%s", prefix, "bias");
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
        // int W = (int)x->ne[0];
        // int H = (int)x->ne[1];
        // int C = (int)x->ne[2];
        // int B = (int)x->ne[3];

        ggml_tensor_t *u = ggml_nn_mean(ctx, x, 2); // dim == 2
        u = ggml_repeat(ctx, u, x);

        ggml_tensor_t *d = ggml_sub(ctx, x, u);
        // # g_s = ggml.ggml_pow(ctx, g_d, 2.0)
        ggml_tensor_t *s = ggml_mul(ctx, d, d);
        s = ggml_nn_mean(ctx, s, 2); // # dim = 2
        s = ggml_sqrt(ctx, s);
        s = ggml_repeat(ctx, s, d);
        x = ggml_div(ctx, d, s);

        // # g_y = ggml_nn_linear(ctx, g_x, g_weight, g_bias)
        // g_weight = ggml_tensor(ctx, model.weight.reshape(1, 1, normalized_shape).repeat(W, H, 1))
        // g_bias = ggml_tensor(ctx, model.bias.reshape(1, 1, normalized_shape).repeat(W, H, 1))
        ggml_tensor_t *g_weight = ggml_reshape_3d(ctx, w, 1, 1, normalized_shape);
        g_weight = ggml_repeat(ctx, g_weight, x);
        ggml_tensor_t *g_bias = ggml_reshape_3d(ctx, b, 1, 1, normalized_shape);
        g_bias = ggml_repeat(ctx, g_bias, x);

        ggml_tensor_t *y = ggml_mul(ctx, x, g_weight);
        y = ggml_add(ctx, y, g_bias);

        return y; 
    }
};

struct ConvNeXt {
    // network params
    struct Conv2d downsample_layers_0_0;
    struct LayerNormChannelsFirst downsample_layers_0_1;

    struct LayerNormChannelsFirst downsample_layers_1_0;
    struct Conv2d downsample_layers_1_1;

    struct LayerNormChannelsFirst downsample_layers_2_0;
    struct Conv2d downsample_layers_2_1;

    struct LayerNormChannelsFirst downsample_layers_3_0;
    struct Conv2d downsample_layers_3_1;

    // [3, 3, 27, 3]
    struct Block states_0[3];
    struct Block states_1[3];
    struct Block states_2[27];
    struct Block states_3[3];

    struct LayerNormChannelsFirst norm_0;
    struct LayerNormChannelsFirst norm_1;
    struct LayerNormChannelsFirst norm_2;
    struct LayerNormChannelsFirst norm_3;

    struct LayerNorm norm; // final norm layer, useless !!!


    void create_weight_tensors(ggml_context_t* ctx) {
        downsample_layers_0_0.in_channels = 3;
        downsample_layers_0_0.out_channels = 192;
        downsample_layers_0_0.kernel_size = {4, 4};
        downsample_layers_0_0.stride = { 4, 4 };
        downsample_layers_0_0.create_weight_tensors(ctx, GGML_TYPE_F32);
        downsample_layers_0_1.normalized_shape = 192;
        downsample_layers_0_1.create_weight_tensors(ctx);

        downsample_layers_1_0.normalized_shape = 192;
        downsample_layers_1_0.create_weight_tensors(ctx);
        downsample_layers_1_1.in_channels = 192;
        downsample_layers_1_1.out_channels = 384;
        downsample_layers_1_1.kernel_size = {2, 2};
        downsample_layers_1_1.stride = { 2, 2 };
        downsample_layers_1_1.create_weight_tensors(ctx, GGML_TYPE_F32);

        downsample_layers_2_0.normalized_shape = 384;
        downsample_layers_2_0.create_weight_tensors(ctx);
        downsample_layers_2_1.in_channels = 384;
        downsample_layers_2_1.out_channels = 768;
        downsample_layers_2_1.kernel_size = {2, 2};
        downsample_layers_2_1.stride = { 2, 2 };
        downsample_layers_2_1.create_weight_tensors(ctx, GGML_TYPE_F32);

        downsample_layers_3_0.normalized_shape = 768;
        downsample_layers_3_0.create_weight_tensors(ctx);
        downsample_layers_3_1.in_channels = 768;
        downsample_layers_3_1.out_channels = 1536;
        downsample_layers_3_1.kernel_size = {2, 2};
        downsample_layers_3_1.stride = { 2, 2 };
        downsample_layers_3_1.create_weight_tensors(ctx, GGML_TYPE_F32);

        // [192, 384, 768, 1536]
        GGML_ASSERT(ARRAY_SIZE(states_0) == 3);
        for (int i = 0; i < ARRAY_SIZE(states_0) /*3*/; i++) {
            states_0[i].dim = 192;
            states_0[i].create_weight_tensors(ctx);
        }

        GGML_ASSERT(ARRAY_SIZE(states_1) == 3);
        for (int i = 0; i < ARRAY_SIZE(states_1) /*3*/; i++) {
            states_1[i].dim = 384;
            states_1[i].create_weight_tensors(ctx);
        }

        GGML_ASSERT(ARRAY_SIZE(states_2) == 27);
        for (int i = 0; i < ARRAY_SIZE(states_2) /*27*/; i++) {
            states_2[i].dim = 768;
            states_2[i].create_weight_tensors(ctx);
        }

        GGML_ASSERT(ARRAY_SIZE(states_3) == 3);
        for (int i = 0; i < ARRAY_SIZE(states_3) /*3*/; i++) {
            states_3[i].dim = 1536;
            states_3[i].create_weight_tensors(ctx);
        }

        // dims=[192, 384, 768, 1536]
        norm_0.normalized_shape = 192;
        norm_0.create_weight_tensors(ctx);
        norm_1.normalized_shape = 384;
        norm_1.create_weight_tensors(ctx);
        norm_2.normalized_shape = 768;
        norm_2.create_weight_tensors(ctx);
        norm_3.normalized_shape = 1536;
        norm_3.create_weight_tensors(ctx);

        // norm, useless
        norm.normalized_shape = 1536;
        norm.eps = 1e-6;
        norm.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "downsample_layers.0.0.");
        downsample_layers_0_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "downsample_layers.0.1.");
        downsample_layers_0_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "downsample_layers.1.0.");
        downsample_layers_1_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "downsample_layers.1.1.");
        downsample_layers_1_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "downsample_layers.2.0.");
        downsample_layers_2_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "downsample_layers.2.1.");
        downsample_layers_2_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "downsample_layers.3.0.");
        downsample_layers_3_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "downsample_layers.3.1.");
        downsample_layers_3_1.setup_weight_names(s);

        for (int i = 0; i < ARRAY_SIZE(states_0) /*3*/; i++) {
            snprintf(s, sizeof(s), "%sstages.0.%d.", prefix, i);
            states_0[i].setup_weight_names(s);
        }
        for (int i = 0; i < ARRAY_SIZE(states_1) /*3*/; i++) {
            snprintf(s, sizeof(s), "%sstages.1.%d.", prefix, i);
            states_1[i].setup_weight_names(s);
        }
        for (int i = 0; i < ARRAY_SIZE(states_2) /*27*/; i++) {
            snprintf(s, sizeof(s), "%sstages.2.%d.", prefix, i);
            states_2[i].setup_weight_names(s);
        }
        for (int i = 0; i < ARRAY_SIZE(states_3) /*3*/; i++) {
            snprintf(s, sizeof(s), "%sstages.3.%d.", prefix, i);
            states_3[i].setup_weight_names(s);
        }

        snprintf(s, sizeof(s), "%s%s", prefix, "norm0.");
        norm_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm_2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "norm3.");
        norm_3.setup_weight_names(s);

        // useless ...
        snprintf(s, sizeof(s), "%s%s", prefix, "norm.");
        norm.setup_weight_names(s);
    }

    std::vector<ggml_tensor_t *> forward(ggml_context_t* ctx, ggml_tensor_t* x) {
        std::vector<ggml_tensor_t *> x0123;
        // ------------------------------------------------------------------
        // Case 0:
        // x = downsample_layers_0(x)
        x = downsample_layers_0_0.forward(ctx, x);
        x = downsample_layers_0_1.forward(ctx, x);
        // x = states_0(x)
        for (int i = 0; i < ARRAY_SIZE(states_0); i++)
            x = states_0[i].forward(ctx, x);
        // x = norm_0(x)
        x = norm_0.forward(ctx, x);
        x0123.push_back(x);

        // ------------------------------------------------------------------
        // Case 1:
        // x = downsample_layers_1(x)
        x = downsample_layers_1_0.forward(ctx, x);
        x = downsample_layers_1_1.forward(ctx, x);
        // x = states_1(x)
        for (int i = 0; i < ARRAY_SIZE(states_1); i++)
            x = states_1[i].forward(ctx, x);
        // x = norm_1(x)
        x = norm_1.forward(ctx, x);
        x0123.push_back(x);

        // ------------------------------------------------------------------
        // Case 2:
        // x = downsample_layers_2(x)
        x = downsample_layers_2_0.forward(ctx, x);
        x = downsample_layers_2_1.forward(ctx, x);
        // x = states_2(x)
        for (int i = 0; i < ARRAY_SIZE(states_2); i++)
            x = states_2[i].forward(ctx, x);
        // x = norm_2(x)
        x = norm_2.forward(ctx, x);
        x0123.push_back(x);

        // ------------------------------------------------------------------
        // Case 3:
        // x = downsample_layers_3(x)
        x = downsample_layers_3_0.forward(ctx, x);
        x = downsample_layers_3_1.forward(ctx, x);
        // x = states_3(x)
        for (int i = 0; i < ARRAY_SIZE(states_3); i++)
            x = states_3[i].forward(ctx, x);
        // x = norm_3(x)
        x = norm_3.forward(ctx, x);
        x0123.push_back(x);

        return x0123;
    }
};


struct DDColor : GGMLNetwork {
    // network hparams
    int MAX_H = 512;
    int MAX_W = 512;
    int MAX_TIMES = 1;

    // network params
    ggml_tensor_t *mean;
    ggml_tensor_t *std;
    // struct Encoder encoder;
    struct ConvNeXt encoder;

    struct Decoder decoder;

    struct Conv2d refine_net; // instance CustomConv2d ...

    void create_weight_tensors(ggml_context_t* ctx) {
        mean = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 3, 1);
        std = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 3, 1);

        encoder.create_weight_tensors(ctx);
        decoder.create_weight_tensors(ctx);

        refine_net.in_channels = 103;
        refine_net.out_channels = 2;
        refine_net.kernel_size = {1, 1};
        refine_net.create_weight_tensors(ctx, GGML_TYPE_F32);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        ggml_format_name(mean, "%s%s", prefix, "mean");
        ggml_format_name(std, "%s%s", prefix, "std");

        snprintf(s, sizeof(s), "%s%s", prefix, "encoder.arch.");
        encoder.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "decoder.");
        decoder.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "refine_net.0.0.");
        refine_net.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        // assert x.size(2) == 512 and x.size(3) == 512, "Please input 1x3x512x512 tensor"
        // x = self.normalize(x)
        // # tensor [x] size: [1, 3, 512, 512], min: -2.117904, max: 2.326308, mean: -0.356781
        
        // # How Avoiding Pitfalls in onnx exporting ??? !!!
        // encoder_layers: ENCODER_RESULT = self.encoder(x)
        // # encoder_layers is tuple: len = 4
        // #     tensor [item] size: [1, 192, 128, 128], min: -10.815851, max: 5.158478, mean: -0.007829
        // #     tensor [item] size: [1, 384, 64, 64], min: -13.253959, max: 16.581171, mean: -0.000148
        // #     tensor [item] size: [1, 768, 32, 32], min: -6.887635, max: 24.325577, mean: 0.001135
        // #     tensor [item] size: [1, 1536, 16, 16], min: -26.544884, max: 16.26297, mean: -0.00155

        // out_feat = self.decoder(encoder_layers)
        // coarse_input = torch.cat([out_feat, x], dim=1)
        // out_ab = self.refine_net(coarse_input)
        // # tensor [out_feat] size: [1, 100, 512, 512], min: -14.656635, max: 24.051313, mean: 1.814844
        // # tensor [coarse_input] size: [1, 103, 512, 512], min: -14.656635, max: 24.051313, mean: 1.751593
        // # tensor [out_ab] size: [1, 2, 512, 512], min: -41.829132, max: 52.045349, mean: 4.471401

        // return out_ab

        GGML_UNUSED(argc);
        ggml_tensor_t* x = argv[0];
        GGML_ASSERT(x->ne[0] == 512 && x->ne[1] == 512 && x->ne[2] == 3); // Channel == 3, 512x512

        x = ggml_nn_normalize(ctx, x, mean, std);
        std::vector<ggml_tensor_t *> encoder_layers = encoder.forward(ctx, x);
        auto out_feat = decoder.forward(ctx, encoder_layers);
        auto coarse_input = ggml_concat(ctx, out_feat, x, 2); // 2 -- on channel
        auto out_ab = refine_net.forward(ctx, coarse_input);

        return out_ab;
    }
};

#endif // __DDCOLOR__H__
