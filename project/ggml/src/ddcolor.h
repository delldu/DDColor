#ifndef __DDCOLOR__H__
#define __DDCOLOR__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"

struct MLP {
    // int num_layers = 3;

    // network params
    struct Linear layers_0;
    struct Linear layers_1;
    struct Linear layers_2;

    void create_weight_tensors(ggml_context_t* ctx) {
        layers_0.in_features = 256;
        layers_0.out_features = 256;
        layers_0.create_weight_tensors(ctx, GGML_TYPE_F32);

        layers_1.in_features = 256;
        layers_1.out_features = 256;
        layers_1.create_weight_tensors(ctx, GGML_TYPE_F32);

        layers_2.in_features = 256;
        layers_2.out_features = 256;
        layers_2.create_weight_tensors(ctx, GGML_TYPE_F32);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "layers.0.");
        layers_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "layers.1.");
        layers_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "layers.2.");
        layers_2.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
        x = layers_0.forward(ctx, x);
        x = ggml_relu_inplace(ctx, x);

        x = layers_1.forward(ctx, x);
        x = ggml_relu_inplace(ctx, x);

        x = layers_2.forward(ctx, x);

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
        in_proj.in_features = 3 * embed_dim;
        in_proj.out_features = embed_dim;
        in_proj.create_weight_tensors(ctx, GGML_TYPE_F32);

        out_proj.in_features = embed_dim;
        out_proj.out_features = embed_dim;
        out_proj.create_weight_tensors(ctx, GGML_TYPE_F32);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "in_proj.");
        in_proj.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "out_proj.");
        out_proj.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* query, ggml_tensor_t* key, ggml_tensor_t* value) {

        return query;
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

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!

    	return x;
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

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 PositionEmbeddingSine() */

struct PositionEmbeddingSine {
    // network hparams
    float scale = 6.283185307179586;

    // network params
    


    void create_weight_tensors(ggml_context_t* ctx) {
        
    }

    void setup_weight_names(const char *prefix) {
        char s[512];
        
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};


struct MultiScaleColorDecoder {
    // network hparams
    int hidden_dim = 256;
    int num_queries = 100;
    int nheads = 8;
    int dim_feedforward = 2048;
    int color_embed_dim = 256;
    int num_scales = 3;
    int num_feature_levels = 3;

    const int dec_layers = 9;

    // network params
    struct PositionEmbeddingSine pe_layer;
    struct SelfAttentionLayer transformer_self_attention_layers[9]; // dec_layers
    struct CrossAttentionLayer transformer_cross_attention_layers[9]; // dec_layers
    struct FFNLayer transformer_ffn_layers[9]; // dec_layers

    struct LayerNorm decoder_norm;

    ggml_tensor_t* query_feat_weight;  // torch.float32, [100, 256] 
    ggml_tensor_t* query_embed_weight;  // torch.float32, [100, 256] 
    ggml_tensor_t* level_embed_weight;  // torch.float32, [3, 256] 

    struct Conv2d input_proj_0;
    struct Conv2d input_proj_1;
    struct Conv2d input_proj_2;

    struct MLP color_embed;


    void create_weight_tensors(ggml_context_t* ctx) {
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
        level_embed_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 3);

        //  in_channels=[512, 512, 256],
        input_proj_0.in_channels = 512;
        input_proj_0.out_channels = 256;
        input_proj_0.kernel_size = {1, 1};
        input_proj_0.create_weight_tensors(ctx, GGML_TYPE_F32);

        input_proj_1.in_channels = 512;
        input_proj_1.out_channels = 256;
        input_proj_1.kernel_size = {1, 1};
        input_proj_1.create_weight_tensors(ctx, GGML_TYPE_F32);

        input_proj_2.in_channels = 256;
        input_proj_2.out_channels = 256;
        input_proj_2.kernel_size = {1, 1};
        input_proj_2.create_weight_tensors(ctx, GGML_TYPE_F32);

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

        snprintf(s, sizeof(s), "%s%s", prefix, "input_proj.0.");
        input_proj_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "input_proj.1.");
        input_proj_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "input_proj.2.");
        input_proj_2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "color_embed.");
        color_embed.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 CustomPixelShuffle(
  (conv): Sequential(
    (0): Conv2d(256, 4096, kernel_size=(1, 1), stride=(1, 1))
  )
  (shuf): PixelShuffle(upscale_factor=4)
  (pad): ReplicationPad2d((1, 0, 1, 0))
  (blur): AvgPool2d(kernel_size=2, stride=1, padding=0)
  (relu): ReLU(inplace=True)
) */

struct CustomPixelShuffle {
    // network hparams
    int ni = 256;
    int nf = 512;
    int scale = 2;
    bool extra_bn = false;
    
    struct Conv2d conv_0;
    struct BatchNorm2d conv_1;
    struct PixelShuffle shuf;

    void create_weight_tensors(ggml_context_t* ctx) {
        conv_0.in_channels = ni;
        conv_0.out_channels = nf * (scale*scale);

        conv_0.kernel_size = {1, 1};
        conv_0.create_weight_tensors(ctx, GGML_TYPE_F32);

        if (extra_bn) { // BatchNorm2d
            conv_1.num_features = nf * (scale*scale);
            conv_1.eps = 1e-5;
            conv_1.create_weight_tensors(ctx);
        }

        shuf.upscale_factor = scale;
        shuf.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.0.");
        conv_0.setup_weight_names(s);

        if (extra_bn) {
            snprintf(s, sizeof(s), "%s%s", prefix, "conv.1.");
            conv_1.setup_weight_names(s);
        }

        // shuf has not weight data ...
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 UnetBlockWide(
  (shuf): CustomPixelShuffle(
    (conv): Sequential(
      (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (shuf): PixelShuffle(upscale_factor=2)
    (pad): ReplicationPad2d((1, 0, 1, 0))
    (blur): AvgPool2d(kernel_size=2, stride=1, padding=0)
    (relu): ReLU(inplace=True)
  )
  (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv): Sequential(
    (0): Conv2d(448, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): ReLU(inplace=True)
    (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (relu): ReLU()
) */

struct UnetBlockWide {
    // network hparams
    
    int up_in_c = 1536;
    int x_in_c = 768;
    int n_out = 512;

    // shuf
    struct CustomPixelShuffle shuf;
    struct BatchNorm2d bn;


        // self.bn = nn.BatchNorm2d(x_in_c) # ggml_debug

        // ni = n_out + x_in_c # 1280
        // self.conv = custom_conv_layer(ni, n_out, extra_bn=True)
        // # (Pdb) self.conv
        // # Sequential(
        // #   (0): Conv2d(1280, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        // #   (1): ReLU(inplace=True)
        // #   (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        // # )
        // self.relu = nn.ReLU()


    // bn
    ggml_tensor_t* bn_weight;  // torch.float32, [192] 
    ggml_tensor_t* bn_bias;  // torch.float32, [192] 
    ggml_tensor_t* bn_running_mean;  // torch.float32, [192] 
    ggml_tensor_t* bn_running_var;  // torch.float32, [192] 

    // conv
    ggml_tensor_t* conv_0_weight;  // torch.float32, [256, 448, 3, 3] 

    ggml_tensor_t* conv_2_weight;  // torch.float32, [256] 
    ggml_tensor_t* conv_2_bias;  // torch.float32, [256] 
    ggml_tensor_t* conv_2_running_mean;  // torch.float32, [256] 
    ggml_tensor_t* conv_2_running_var;  // torch.float32, [256]


    void create_weight_tensors(ggml_context_t* ctx) {
        shuf.ni = up_in_c;
        shuf.nf = n_out;
        shuf.scale = 2;
        shuf.extra_bn = true;
        shuf.create_weight_tensors(ctx);

        bn.num_features = x_in_c;
        bn.eps = 1e-5;
        bn.create_weight_tensors(ctx);
        // conv_0_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 448, 256);
        // conv_2_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        // conv_2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        // conv_2_running_mean = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        // conv_2_running_var = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "shuf.");
        shuf.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "bn.");
        bn.setup_weight_names(s);

        // ggml_format_name(conv_0_weight, "%s%s", prefix, "conv.0.weight");
        // ggml_format_name(conv_2_weight, "%s%s", prefix, "conv.2.weight");
        // ggml_format_name(conv_2_bias, "%s%s", prefix, "conv.2.bias");
        // ggml_format_name(conv_2_running_mean, "%s%s", prefix, "conv.2.running_mean");
        // ggml_format_name(conv_2_running_var, "%s%s", prefix, "conv.2.running_var");
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!

    	return x;
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
        layers_0.up_in_c = 512;
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

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 LayerNormChannelsLast() */

struct LayerNormChannelsLast {
    // network hparams
    float eps = 1e-06;

    // network params
    struct LayerNorm norm;    


    void create_weight_tensors(ggml_context_t* ctx) {
        
    }

    void setup_weight_names(const char *prefix) {
        char s[512];
        
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 Block(
  (dwconv): Conv2d(1536, 1536, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=1536)
  (norm): LayerNormChannelsLast()
  (pwconv1): Linear(in_features=1536, out_features=6144, bias=True)
  (act): GELU(approximate='none')
  (pwconv2): Linear(in_features=6144, out_features=1536, bias=True)
) */

// class Block(nn.Module):
//     def __init__(self, dim, layer_scale_init_value=1e-6):
//         super().__init__()
//         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
//         #  kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)
//         self.norm = LayerNormChannelsLast(dim, eps=1e-6)
//         self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
//         self.act = nn.GELU()
//         self.pwconv2 = nn.Linear(4 * dim, dim)
//         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=False)

//     def forward(self, x):
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


struct Block {
    int dim = 1536;

    // network params
    struct Conv2d dwconv;
    struct LayerNorm norm;
    struct Linear pwconv1;
    struct Linear pwconv2;
    ggml_tensor_t *gamma;

    void create_weight_tensors(ggml_context_t* ctx) {
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

/*
 LayerNormChannelsFirst() */

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
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 ConvNeXt(
  (downsample_layers): ModuleList(
    (0): Sequential(
      (0): Conv2d(3, 192, kernel_size=(4, 4), stride=(4, 4))
      (1): LayerNormChannelsFirst()
    )
    (1): Sequential(
      (0): LayerNormChannelsFirst()
      (1): Conv2d(192, 384, kernel_size=(2, 2), stride=(2, 2))
    )
    (2): Sequential(
      (0): LayerNormChannelsFirst()
      (1): Conv2d(384, 768, kernel_size=(2, 2), stride=(2, 2))
    )
    (3): Sequential(
      (0): LayerNormChannelsFirst()
      (1): Conv2d(768, 1536, kernel_size=(2, 2), stride=(2, 2))
    )
  )
  (stages): ModuleList(
    (0): Sequential(
      (0): Block(
        (dwconv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=192, out_features=768, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=768, out_features=192, bias=True)
      )
      (1): Block(
        (dwconv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=192, out_features=768, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=768, out_features=192, bias=True)
      )
      (2): Block(
        (dwconv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=192, out_features=768, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=768, out_features=192, bias=True)
      )
    )
    (1): Sequential(
      (0): Block(
        (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
      )
      (1): Block(
        (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
      )
      (2): Block(
        (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
      )
    )
    (2): Sequential(
      (0): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (1): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (2): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (3): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (4): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (5): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (6): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (7): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (8): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (9): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (10): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (11): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (12): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (13): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (14): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (15): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (16): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (17): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (18): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (19): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (20): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (21): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (22): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (23): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (24): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (25): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
      (26): Block(
        (dwconv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
      )
    )
    (3): Sequential(
      (0): Block(
        (dwconv): Conv2d(1536, 1536, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=1536)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=1536, out_features=6144, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=6144, out_features=1536, bias=True)
      )
      (1): Block(
        (dwconv): Conv2d(1536, 1536, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=1536)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=1536, out_features=6144, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=6144, out_features=1536, bias=True)
      )
      (2): Block(
        (dwconv): Conv2d(1536, 1536, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=1536)
        (norm): LayerNormChannelsLast()
        (pwconv1): Linear(in_features=1536, out_features=6144, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=6144, out_features=1536, bias=True)
      )
    )
  )
  (norm0): LayerNormChannelsFirst()
  (norm1): LayerNormChannelsFirst()
  (norm2): LayerNormChannelsFirst()
  (norm3): LayerNormChannelsFirst()
  (norm): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
) */

struct ConvNeXt {
    // network hparams
    

    // network params
    struct Conv2d downsample_layers_0_0;
    struct LayerNormChannelsFirst downsample_layers_0_1;

    struct Conv2d downsample_layers_1_0;
    struct LayerNormChannelsFirst downsample_layers_1_1;

    struct Conv2d downsample_layers_2_0;
    struct LayerNormChannelsFirst downsample_layers_2_1;

    struct Conv2d downsample_layers_3_0;
    struct LayerNormChannelsFirst downsample_layers_3_1;

    // [3, 3, 27, 3]
    struct Block states_0[3];
    struct Block states_1[3];
    struct Block states_2[27];
    struct Block states_3[2];

    struct LayerNormChannelsFirst norm0;
    struct LayerNormChannelsFirst norm1;
    struct LayerNormChannelsFirst norm2;
    struct LayerNormChannelsFirst norm3;

    struct LayerNorm norm;


    void create_weight_tensors(ggml_context_t* ctx) {
        downsample_layers_0_0.in_channels = 3;
        downsample_layers_0_0.out_channels = 192;
        downsample_layers_0_0.kernel_size = {4, 4};
        downsample_layers_0_0.stride = { 4, 4 };
        downsample_layers_0_0.create_weight_tensors(ctx);
        downsample_layers_0_1.normalized_shape = 192;
        downsample_layers_0_1.create_weight_tensors(ctx);

        downsample_layers_1_0.in_channels = 192;
        downsample_layers_1_0.out_channels = 384;
        downsample_layers_1_0.kernel_size = {2, 2};
        downsample_layers_1_0.stride = { 2, 2 };
        downsample_layers_1_0.create_weight_tensors(ctx);
        downsample_layers_1_1.normalized_shape = 384;
        downsample_layers_1_1.create_weight_tensors(ctx);

        downsample_layers_2_0.in_channels = 384;
        downsample_layers_2_0.out_channels = 768;
        downsample_layers_2_0.kernel_size = {2, 2};
        downsample_layers_2_0.stride = { 2, 2 };
        downsample_layers_2_0.create_weight_tensors(ctx);
        downsample_layers_2_1.normalized_shape = 768;
        downsample_layers_2_1.create_weight_tensors(ctx);

        downsample_layers_3_0.in_channels = 768;
        downsample_layers_3_0.out_channels = 1536;
        downsample_layers_3_0.kernel_size = {2, 2};
        downsample_layers_3_0.stride = { 2, 2 };
        downsample_layers_3_0.create_weight_tensors(ctx);
        downsample_layers_3_1.normalized_shape = 1536;
        downsample_layers_3_1.create_weight_tensors(ctx);

        // [192, 384, 768, 1536]
        for (int i = 0; i < 3; i++) {
            states_0[i].dim = 192;
            states_0[i].create_weight_tensors(ctx);
        }
        for (int i = 0; i < 3; i++) {
            states_1[i].dim = 384;
            states_1[i].create_weight_tensors(ctx);
        }
        for (int i = 0; i < 27; i++) {
            states_2[i].dim = 768;
            states_2[i].create_weight_tensors(ctx);
        }
        for (int i = 0; i < 3; i++) {
            states_3[i].dim = 1536;
            states_3[i].create_weight_tensors(ctx);
        }

        norm0.normalized_shape = 192;
        norm0.create_weight_tensors(ctx);

        norm1.normalized_shape = 384;
        norm1.create_weight_tensors(ctx);

        norm2.normalized_shape = 768;
        norm2.create_weight_tensors(ctx);

        norm3.normalized_shape = 1536;
        norm3.create_weight_tensors(ctx);

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

        for (int i = 0; i < 3; i++) {
            snprintf(s, sizeof(s), "%sstages.0.%d.", prefix, i);
            states_0[i].setup_weight_names(s);
        }
        for (int i = 0; i < 3; i++) {
            snprintf(s, sizeof(s), "%sstages.1.%d.", prefix, i);
            states_1[i].setup_weight_names(s);
        }
        for (int i = 0; i < 27; i++) {
            snprintf(s, sizeof(s), "%sstages.2.%d.", prefix, i);
            states_2[i].setup_weight_names(s);
        }
        for (int i = 0; i < 3; i++) {
            snprintf(s, sizeof(s), "%sstages.3.%d.", prefix, i);
            states_3[i].setup_weight_names(s);
        }


        snprintf(s, sizeof(s), "%s%s", prefix, "norm0.");
        norm0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "norm3.");
        norm3.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm.");
        norm.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

struct Encoder {
    // network hparams

    // network params
    struct ConvNeXt arch;


    void create_weight_tensors(ggml_context_t* ctx) {
        arch.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "arch.");
        arch.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

struct DDColor {
    // network hparams
    int MAX_H = 512;
    int MAX_W = 512;
    int MAX_TIMES = 1;

    // network params
    struct Encoder encoder;
    struct Decoder decoder;

    struct Conv2d refine_net;

    ggml_tensor_t* refine_net_0_0_bias;  // torch.float32, [2] 
    ggml_tensor_t* refine_net_0_0_weight;  // torch.float32, [2, 103, 1, 1]


    void create_weight_tensors(ggml_context_t* ctx) {
        encoder.create_weight_tensors(ctx);
        decoder.create_weight_tensors(ctx);

        refine_net.in_channels = 103;
        refine_net.out_channels = 2;
        refine_net.kernel_size = {1, 1};
        refine_net.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "encoder.");
        encoder.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "decoder.");
        decoder.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "refine_net.0.0.");
        refine_net.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

#endif // __DDCOLOR__H__
