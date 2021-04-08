import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

from resnetV2 import ResNetV2
from torch.nn.modules.utils import _pair


class MLP(nn.Module):
    """
    Multilayer Perceptron Model. This is a two layer fully connected multilayer perceptron. Both layers
    have configurable dropout rate. There is an activation function between layers (relu or gelu).

    The input and output are both determined by hidden size, while the hidden layer has size determined by
    mlp_dim in the config.
    """
    def __init__(self, config):
        super(MLP, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = f.relu        # TODO: Test this with both relu and gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        # Biases initialised using normal distribution, weights initialised using xavier uniform,
        # which is a symmetric uniform distribution with the range inversely proportional to the
        # sum of the fan in and fan out (input and output weights).
        # TODO Investigate why xavier uniform is beneficial and try alternate initialisations
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    """ See 3.2 in Ashish Vaswani et al: 'Attention is all you need' """
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Embeddings(nn.Module):
    """
    Construct the embeddings from patch, position embeddings. Splits the image up into patches of size P. Each patch
    is linearly transformed into a D dimensional vector. This results in an output vector of dimension (B, HW/P^2, D)
    where B is the batch size, P is the patch size and D is the hidden size (sometimes refered to as the transformer
    dimension). This is done with a convolution of size P and stride P with output channels of D. i.e. It converts
    each PxP section into the image into a D dimensional vector.

    If hybrid mode is on, then these embeddings are constructed from the output of the ResNet rather than from the
    image directly. The output of the ResNet has dimensions (B, 1024*width_factor, H//16, W//16). In this case the
    embedding is done in the same way, except the in channels are 1024*width_factor instead of 3, and the patch size
    is determined by grid_size. For example if grid_size is 3, then the (H//16, W//16) 'image' is split into 3x3
    patches.

    Finally after the embeddings are created a bias is added (position embeddings), which I believe is needed so
    their is some positional information within the embeddings
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    """
    This module creates a single block in the transformer. It consists of an attention layer followed by the
    fully connected MLP layer, with Layer Norm after each layer. Between each layer is a skip connection.
    The input and output dimensions are the same (B, N, D) where D is the hidden size, N is the number of
    embeddings, and B is the batch size.
    """
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = MLP(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    """
    The encoder consists of num_layers copies of the Block module. The encoder is fed with the output of the
    embeddings, and so has input dimensionality of (B, N, D) where B is the batch size, D is the hidden size and
    N = HW/P^2 is the number of embeddings. As far as I can tell, this doesn't change the dimensionality of the
    vector, so I'm not sure how it is an encoder. (Experimentally verified, this module does not change the
    dimensionality of the input)
    """
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            # TODO Why does this need to be copied???
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    """
    Transformer: This takes the image as input, creates the embeddings, then encodes the embeddings using the
    encoder. The input has dimensions (B, W, H) and the output has dimensions (B, N, D) where B is the batch size,
    N is the number of patches, D is the hidden size and H and W are the height and width of the images. This module
    also returns the attention weights, which is the output of the softmax in the attention module, before it is
    multiplied by the values tensor. As far as I can tell, this isn't used, so I'm not sure what the point of it is.
    features is also returned, and this is the latent vectors generated between each of the 3 blocks in
    ResNet (if operating in hybrid mode). This is passed into the decoder to provide skip connections, and higher
    resolution spatial information to the decoder
    """
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        # TODO What is the point of attn_weights and is it used?
        return encoded, attn_weights, features
