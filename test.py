from resnetV2 import ResNetV2
import torch
from model import Embeddings, Encoder
import ml_collections

config = ml_collections.ConfigDict()
config.patches = ml_collections.ConfigDict({'size': (16, 16)})
config.hidden_size = 768
config.transformer = ml_collections.ConfigDict()
config.transformer.mlp_dim = 3072
config.transformer.num_heads = 12
config.transformer.num_layers = 12
config.transformer.attention_dropout_rate = 0.0
config.transformer.dropout_rate = 0.1

config.classifier = 'seg'
config.representation_size = None
config.resnet_pretrained_path = None
config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
config.patch_size = 16

config.decoder_channels = (256, 128, 64, 16)
config.n_classes = 2
config.activation = 'softmax'

img = torch.zeros((8, 3, 128, 128))

embeddings = Embeddings(config, 128)

# (8, 64, 768)
img_embeddings, features = embeddings(img)

print(features)
print(img_embeddings.shape)

encoder = Encoder(config, False)

encodings, attn_features = encoder(img_embeddings)

print(attn_features)
print(encodings.shape)