import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm

from resnetV2 import ResNetV2


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
        self.act_fn = f.relu  # TODO: Test this with both relu and gelu
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

    def __init__(self, config):
        super(Attention, self).__init__()
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
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Embeddings(nn.Module):
    """
    Construct the embeddings from patch, position embeddings. Splits the image up into patches of size P. Each patch
    is linearly transformed into a D dimensional vector. This results in an output vector of dimension (B, HW/P^2, D)
    where B is the batch size, P is the patch size and D is the hidden size (sometimes referred to as the transformer
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
        if config.patches.get("grid") is not None:  # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = config.patches["size"]
            if type(patch_size) != tuple:
                patch_size = (patch_size, patch_size)
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

    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = MLP(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Encoder(nn.Module):
    """
    The encoder consists of num_layers copies of the Block module. The encoder is fed with the output of the
    embeddings, and so has input dimensionality of (B, N, D) where B is the batch size, D is the hidden size and
    N = HW/P^2 is the number of embeddings. As far as I can tell, this doesn't change the dimensionality of the
    vector, so I'm not sure how it is an encoder. (Experimentally verified, this module does not change the
    dimensionality of the input)
    """
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config)
            # TODO Why does this need to be copied???
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


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

    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features


class Conv2dReLU(nn.Sequential):
    """
    Simply a convolutional layer, followed by a batch norm, followed by a ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, batch_norm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not batch_norm)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    """
    Decoder block used to up sample the data. These are cascaded together to create the cascaded up sampler (CUP)
    The input has dimensions (B, I, H, W) and the output has dimensions (B, O, 2H, 2W) where I and O are the input
    and output channels respectively and are parameters to the block. Data is up sampled linearly by a factor of 2,
    then passed through two 3x3 convolutional layers, where the first one changes the channels and the second doesn't.
    An optional skip connection can be passed in as well, which is concatenated after up sampling, but before
    convolutions. Thus, it must have dimension (B, S, 2H, 2W) where S is the skip channels and is a parameter to the
    block. By defualt there are zero skip channels (no skip connection).
    """
    def __init__(self, in_channels, out_channels, skip_channels=0, batch_norm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels,
                                kernel_size=3, padding=1, batch_norm=batch_norm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, batch_norm=batch_norm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):
    """
    This module goes at the very end of the model and converts the output of the cascaded up sampler to the final
    output. It allows the output to optionally be scaled by some amount, and also converts the channels to the correct
    number of channels. This is done by a convolutional layer (raw convolution without bn or relu) followed by
    either an up sample or an identity. If a patch size of 16 is used with a square image that is a multiple of 16,
    then the output of the decoder will match the input, so up sampling will not be required.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, up_sampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=up_sampling) if up_sampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    """
    Series of 4 decoder blocks. Input comes from the transformer and has dimensionality of (B, N, D) where B is the
    number of batches, N is the number of patches and D is the hidden size. First the input is reshaped into traditional
    image dimensions: (B, D, H, W) where H = W = sqrt(N). This un-flattens the patches that were flattened during the
    embedding stage. We then have one 3x3 convolution to convert this to 512 channels (referred to the head channels).
    Finally, this is passed through 4 decoder blocks, each of which doubles the image size, resulting in an inage of
    size (16H, 16W). This is why a patch size of 16 is convenient, as the output of this will have the same size as the
    input (assuming the image is a square with sides that are multiples of 16). However, if the output of this doesn't
    match the input size, we can upscale it (or downscale it) using the segmentation head. The channels between each
    block is configurable with the config. The number of skip connections and their channels can also be configured in
    the config. The code allows for four skip connections, however I'm not sure why as there are only 3 features, and
    4 is never actually used.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels, kernel_size=3, padding=1, batch_norm=True)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        # Zero out skip channels that aren't used. For example if n_skip is 3, the final skip channel is zeroed as it
        # isn't used. This is needed because the decoder block needs to know how many skip channels it gets so that
        # it can initialise its convolutional layers
        skip_channels = [0, 0, 0, 0] if self.config.n_skip == 0 else self.config.skip_channels
        for i in range(4-self.config.n_skip):
            skip_channels[3-i] = 0

        blocks = [DecoderBlock(in_channels[i], out_channels[i], skip_channels[i]) for i in range(4)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        batches, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(batches, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    """
    The final segmentation vision transformer module. Consists of the transformer which encodes the image into a latent
    vector space of dimension (B, N, D) where B is the number of batches, N is the number of patches and D is the
    hidden size. Then up samples this using the decoder which outputs a vector of dimension (B, C, H, W) where C is the
    final decoder channel determined in the config and H = W = 16sqrt(N). Finally, this output is passed through the
    segmentation head which contains one final convolution to convert the the desired output channels, and an up sample
    is required to get the the same size as the input.
    """
    def __init__(self, config, img_size=224):
        super(VisionTransformer, self).__init__()
        # Image size must be a tuple
        if type(img_size) != tuple:
            img_size = (img_size, img_size)

        self.transformer = Transformer(config, img_size)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )

    def forward(self, x):
        # If the image is grayscale (with one channel), repeat it 3 times to create 3 channels
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        # TODO Fix segmentation head so it automatically chooses the scale to match the input size
        out = self.segmentation_head(x)
        return out
