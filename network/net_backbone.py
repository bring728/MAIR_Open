from torch import Tensor
import torch.utils.checkpoint as cp
import math
from collections import OrderedDict
from functools import partial
from typing import Tuple, Callable, List, NamedTuple, Optional
from torchvision.ops.misc import MLP
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from inspect import isfunction
import numpy as np
import torchvision.models as tvm
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_


# a = tvm.vit_b_16()
# b = torch.zeros([1, 3, 224, 224])
# a(b)


def cat_up(x, y, align=False):
    if x.shape[2] != y.shape[2]:
        x = F.interpolate(x, [y.size(2), y.size(3)], mode='bilinear', align_corners=align)
    return F.interpolate(torch.cat([x, y], dim=1), scale_factor=2, mode='bilinear', align_corners=align)


### vision transformer start
class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i + 1}.{type}"
                    new_key = f"{prefix}{3 * i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
            self,
            seq_length: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
            self,
            input_ch: int,
            image_size: tuple,
            patch_size: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            num_classes: int = 1000,
            representation_size: Optional[int] = None,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.input_ch = input_ch
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        self.conv_proj = nn.Conv2d(
            in_channels=input_ch, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        seq_length = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)
        # Classifier "token" as used by standard language architectures
        return x


##vision transformer end


###denseNet start
class _DenseLayer(nn.Module):
    def __init__(
            self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float,
            memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:  # noqa: F811
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
            self,
            num_layers: int,
            num_input_features: int,
            bn_size: int,
            growth_rate: int,
            drop_rate: float,
            memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
            self,
            input_ch: int = 8,
            growth_rate: int = 32,
            block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
            num_init_features: int = 64,
            bn_size: int = 4,
            drop_rate: float = 0,
            memory_efficient: bool = False,
    ) -> None:

        super().__init__()

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(input_ch, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


class EncoderDenseNet(nn.Module):
    def __init__(self, input_ch, init_feature, block_config=(6, 12, 24, 16), downsize=False) -> None:
        super().__init__()
        if downsize:
            self.num_layer = 4
        else:
            self.num_layer = 3
        # self.densenet = tvm.densenet121(pretrained=True)
        self.densenet = DenseNet(input_ch, init_feature // 2, block_config, init_feature)
        self.l_ind = list(range(4, len(self.densenet.features), 2))

    def forward(self, x):
        out = []
        x = self.densenet.features[:self.num_layer](x)
        out.append(x)
        for l in self.l_ind:
            x = self.densenet.features[l:l + 2](x)
            out.append(x)
        return out


class DecoderDenseNet(nn.Module):
    def __init__(self, out_ch, init_feature):
        super().__init__()
        self.dconv1 = nn.Conv2d(in_channels=init_feature * 16, out_channels=init_feature * 8, kernel_size=3, stride=1,
                                padding=1, bias=True)
        self.dgn1 = nn.GroupNorm(num_groups=32, num_channels=init_feature * 8)

        self.dconv2 = nn.Conv2d(in_channels=init_feature * 16, out_channels=init_feature * 4, kernel_size=3, stride=1,
                                padding=1, bias=True)
        self.dgn2 = nn.GroupNorm(num_groups=16, num_channels=init_feature * 4)

        self.dconv3 = nn.Conv2d(in_channels=init_feature * 8, out_channels=init_feature * 2, kernel_size=3, stride=1,
                                padding=1, bias=True)
        self.dgn3 = nn.GroupNorm(num_groups=16, num_channels=init_feature * 2)

        self.dconv4 = nn.Conv2d(in_channels=init_feature * 4, out_channels=init_feature, kernel_size=3, stride=1,
                                padding=1, bias=True)
        self.dgn4 = nn.GroupNorm(num_groups=8, num_channels=init_feature)

        self.dconv5 = nn.Conv2d(in_channels=init_feature * 2, out_channels=init_feature, kernel_size=3, stride=1,
                                padding=1, bias=True)
        self.dgn5 = nn.GroupNorm(num_groups=4, num_channels=init_feature)

        self.dpadFinal = nn.ReplicationPad2d(1)
        self.dconvFinal = nn.Conv2d(in_channels=init_feature, out_channels=out_ch, kernel_size=3, stride=1, bias=True)

    def forward(self, hs):
        dx1 = F.relu(self.dgn1(self.dconv1(hs[-1])))
        dx2 = F.relu(self.dgn2(self.dconv2(cat_up(dx1, hs[-2]))), True)
        dx3 = F.relu(self.dgn3(self.dconv3(cat_up(dx2, hs[-3]))), True)
        dx4 = F.relu(self.dgn4(self.dconv4(cat_up(dx3, hs[-4]))), True)
        dx5 = F.relu(self.dgn5(self.dconv5(cat_up(dx4, hs[-5]))), True)
        x_out = self.dconvFinal(self.dpadFinal(dx5))
        return x_out


class DecoderDenseNetDL(nn.Module):
    def __init__(self, out_ch, init_feature):
        super().__init__()
        self.dconv1 = nn.Conv2d(in_channels=init_feature * 16, out_channels=init_feature * 8, kernel_size=3, stride=1,
                                padding=1, bias=True)
        self.dgn1 = nn.GroupNorm(num_groups=32, num_channels=init_feature * 8)

        self.dconv2 = nn.Conv2d(in_channels=init_feature * 16, out_channels=init_feature * 4, kernel_size=3, stride=1,
                                padding=1, bias=True)
        self.dgn2 = nn.GroupNorm(num_groups=16, num_channels=init_feature * 4)

        self.dconv3 = nn.Conv2d(in_channels=init_feature * 8, out_channels=init_feature * 2, kernel_size=3, stride=1,
                                padding=1, bias=True)
        self.dgn3 = nn.GroupNorm(num_groups=16, num_channels=init_feature * 2)

        self.dconv4 = nn.Conv2d(in_channels=init_feature * 4, out_channels=init_feature * 3, kernel_size=3, stride=1,
                                padding=1, bias=True)
        self.dgn4 = nn.GroupNorm(num_groups=8, num_channels=init_feature * 3)

        self.dconv5 = nn.Conv2d(in_channels=init_feature * 3, out_channels=init_feature * 2, kernel_size=3, stride=1,
                                padding=1, bias=True)
        self.dgn5 = nn.GroupNorm(num_groups=4, num_channels=init_feature * 2)

        self.dpadFinal = nn.ReplicationPad2d(1)
        self.dconvFinal = nn.Conv2d(in_channels=init_feature * 2, out_channels=out_ch, kernel_size=3, stride=1,
                                    bias=True)

    def forward(self, hs):
        dx1 = F.relu(self.dgn1(self.dconv1(hs[-1])))
        dx2 = F.relu(self.dgn2(self.dconv2(cat_up(dx1, hs[-2]))), True)
        dx3 = F.relu(self.dgn3(self.dconv3(cat_up(dx2, hs[-3]))), True)
        dx4 = F.relu(self.dgn4(self.dconv4(torch.cat([dx3, hs[-4]], dim=1))), True)
        dx5 = F.relu(self.dgn5(self.dconv5(dx4)), True)
        x_out = self.dconvFinal(self.dpadFinal(dx5))
        return x_out


class DecoderDenseNetSVL(nn.Module):
    def __init__(self, out_ch, init_feature):
        super().__init__()
        self.dconv1 = nn.Conv2d(in_channels=init_feature * 16, out_channels=init_feature * 8, kernel_size=3, stride=1,
                                padding=1, bias=True)
        self.dgn1 = nn.GroupNorm(num_groups=32, num_channels=init_feature * 8)

        self.dconv2 = nn.Conv2d(in_channels=init_feature * 16, out_channels=init_feature * 8, kernel_size=3, stride=1,
                                padding=1, bias=True)
        self.dgn2 = nn.GroupNorm(num_groups=16, num_channels=init_feature * 8)

        self.dconv3 = nn.Conv2d(in_channels=init_feature * 16, out_channels=init_feature * 4, kernel_size=3, stride=1,
                                padding=1, bias=True)
        self.dgn3 = nn.GroupNorm(num_groups=16, num_channels=init_feature * 4)

        self.dconv4 = nn.Conv2d(in_channels=init_feature * 8, out_channels=init_feature * 2, kernel_size=3, stride=1,
                                padding=1, bias=True)
        self.dgn4 = nn.GroupNorm(num_groups=8, num_channels=init_feature * 2)

        self.dconv5 = nn.Conv2d(in_channels=init_feature * 4, out_channels=init_feature * 2, kernel_size=3, stride=1,
                                padding=1, bias=True)
        self.dgn5 = nn.GroupNorm(num_groups=4, num_channels=init_feature * 2)

        self.dconv6 = nn.Conv2d(in_channels=init_feature * 2, out_channels=init_feature, kernel_size=3, stride=1,
                                padding=1, bias=True)
        self.dgn6 = nn.GroupNorm(num_groups=2, num_channels=init_feature)

        self.dpadFinal = nn.ReplicationPad2d(1)
        self.dconvFinal = nn.Conv2d(in_channels=init_feature, out_channels=out_ch, kernel_size=3, stride=1,
                                    bias=True)

    def forward(self, hs):
        dx1 = F.relu(self.dgn1(self.dconv1(hs[-1])))
        dx2 = F.relu(self.dgn2(self.dconv2(cat_up(dx1, hs[-2]))), True)
        dx3 = F.relu(self.dgn3(self.dconv3(cat_up(dx2, hs[-3]))), True)
        dx4 = F.relu(self.dgn4(self.dconv4(cat_up(dx3, hs[-4]))), True)
        dx5 = F.relu(self.dgn5(self.dconv5(cat_up(dx4, hs[-5]))), True)
        dx6 = F.relu(self.dgn6(self.dconv6(dx5)), True)
        x_out = self.dconvFinal(self.dpadFinal(dx6))
        return x_out


###basic unet start
def make_layer(pad_type='zeros', padding=1, in_ch=3, out_ch=64, kernel=3, stride=1, norm_layer='group', dropout=0.0,
               act='relu'):
    pad_type = pad_type.lower()
    norm_layer = norm_layer.lower()
    layers = []
    if pad_type == 'rep':
        padding_mode = 'replicate'
    elif pad_type == 'zeros':
        padding_mode = 'zeros'
    layers.append(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride,
                  padding_mode=padding_mode, padding=padding))

    if norm_layer == 'group':
        layers.append(nn.GroupNorm(num_groups=out_ch // 16, num_channels=out_ch))
    elif norm_layer == 'batch':
        layers.append(nn.BatchNorm2d(out_ch))
    elif norm_layer == 'instance':
        layers.append(nn.InstanceNorm2d(out_ch))
    elif norm_layer == 'none':
        norm = 'none'
    else:
        raise Exception('not implemented pad')

    if not act == 'None':
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class EncoderBasic(nn.Module):
    def __init__(self, input_ch, norm, filters):
        super(EncoderBasic, self).__init__()
        self.layers = nn.ModuleList()
        self.length = len(filters)

        for i in range(self.length):
            if i == 0:
                self.layers.append(make_layer(pad_type='rep', in_ch=input_ch, out_ch=filters[i], kernel=4, stride=2,
                                              norm_layer=norm))
            elif 0 < i < self.length - 1:
                self.layers.append(
                    make_layer(in_ch=filters[i - 1], out_ch=filters[i], kernel=4, stride=2, norm_layer=norm))
            elif i == self.length - 1:
                self.layers.append(
                    make_layer(in_ch=filters[i - 1], out_ch=filters[i], kernel=3, stride=1, norm_layer=norm))

    def forward(self, x):
        hs = []
        for layer in self.layers:
            x = layer(x)
            hs.append(x)
        return hs


class DecoderBasic(nn.Module):
    def __init__(self, out_ch, norm, filters):
        super(DecoderBasic, self).__init__()
        filters = filters[::-1]
        self.layers = nn.ModuleList()
        self.length = len(filters) - 1
        for i in range(self.length):
            if i == 0:
                self.layers.append(make_layer(in_ch=filters[i], out_ch=filters[i + 1], norm_layer=norm))
            else:
                self.layers.append(make_layer(in_ch=filters[i] * 2, out_ch=filters[i + 1], norm_layer=norm))
        self.layer_final = make_layer(pad_type='rep', in_ch=filters[-1], out_ch=out_ch, act='None', norm_layer='None')

    def forward(self, hs):
        x = None
        for i in range(self.length):
            if i == 0:
                x = self.layers[i](hs[-i - 1])
            else:
                x = self.layers[i](cat_up(x, hs[-i - 1]))
        x_out = self.layer_final(x)
        return x_out


###basic unet end

###IBRNet start https://github.com/googleinterns/IBRNet?tab=readme-ov-file
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, padding_mode='reflect')


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding_mode='reflect')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, track_running_stats=False, affine=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, track_running_stats=False, affine=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, track_running_stats=False, affine=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2,
                              padding_mode='reflect')
        self.bn = nn.BatchNorm2d(num_out_layers, track_running_stats=track_stats)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)


track_stats = True


class ResUNet(nn.Module):
    def __init__(self, input_ch, dim, norm):
        super().__init__()
        filters = [64, 128, 256, 512]
        layers = [2, 2, 2, 2]
        if norm == 'instance':
            norm_layer = nn.InstanceNorm2d
        elif norm == 'batch':
            norm_layer = nn.BatchNorm2d
        else:
            raise Exception('context norm layer error')
        self._norm_layer = norm_layer
        self.dilation = 1
        block = BasicBlock
        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(input_ch, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False,
                               padding_mode='reflect')
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, filters[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, filters[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filters[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filters[3], layers[3], stride=2)

        # decoder
        self.upconv4 = upconv(filters[3], filters[2], 3, 2)
        self.iconv4 = conv(filters[2] + filters[2], filters[2], 3, 1)
        self.upconv3 = upconv(filters[2], filters[1], 3, 2)
        self.iconv3 = conv(filters[1] + filters[1], filters[1], 3, 1)
        self.upconv2 = upconv(filters[1], filters[0], 3, 2)
        self.iconv2 = conv(filters[0] + filters[0], dim, 3, 1)
        # fine-level conv
        self.out_conv = nn.Conv2d(dim, dim, 1, 1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, track_running_stats=track_stats, ),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                            norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x_encoded = self.layer4(x3)

        x = self.upconv4(x_encoded)
        x = torch.cat([x3, x], dim=1)
        x = self.iconv4(x)

        x = self.upconv3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.iconv3(x)

        x = self.upconv2(x)
        x = torch.cat([x1, x], dim=1)
        x = self.iconv2(x)

        x_out = self.out_conv(x)
        return x_out


###IBRNet end


### CBNdecoder start

class CBatchNorm1d(nn.Module):
    ''' Conditional batch normalization layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert (x.size(0) == c.size(0))
        assert (c.size(1) == self.c_dim)

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class CBatchNorm1d_legacy(nn.Module):
    ''' Conditional batch normalization legacy layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.fc_gamma = nn.Linear(c_dim, f_dim)
        self.fc_beta = nn.Linear(c_dim, f_dim)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.ones_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x, c):
        batch_size = x.size(0)
        # Affine mapping
        gamma = self.fc_gamma(c)
        beta = self.fc_beta(c)
        gamma = gamma.view(batch_size, self.f_dim, 1)
        beta = beta.view(batch_size, self.f_dim, 1)
        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta
        return out


class CResnetBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks
    '''

    def __init__(self, c_dim, size_in, size_h=None, size_out=None,
                 norm_method='batch_norm', legacy=False):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if not legacy:
            self.bn_0 = CBatchNorm1d(c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d(c_dim, size_h, norm_method=norm_method)
        else:
            self.bn_0 = CBatchNorm1d_legacy(c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d_legacy(c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx


class DecoderCBatchNorm2(nn.Module):
    ''' Decoder with CBN class 2.
    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of ResNet blocks
    '''

    def __init__(self, dim=3, c_dim=128, hidden_size=256, n_blocks=3, out_dim=32):
        super().__init__()

        self.conv_p = nn.Conv1d(dim, hidden_size, 1)
        self.blocks = nn.ModuleList([CResnetBlockConv1d(c_dim, hidden_size) for i in range(n_blocks)])

        self.bn = CBatchNorm1d(c_dim, hidden_size)
        self.conv_out = nn.Conv1d(hidden_size, out_dim, 1)
        self.actvn = nn.ReLU()

    def forward(self, p, c):
        bn, cn = c.size()
        net = self.conv_p(p).expand(bn, -1, -1)

        for block in self.blocks:
            net = block(net, c)

        out = self.conv_out(self.actvn(self.bn(net, c)))
        return out


### CBNdecoder end


### attention start
def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = GEGLU(dim, inner_dim)
        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class CrossAttention(nn.Module):
    def __init__(self, input_dim, context_dim=None, out_dim=None, num_heads=8, dim_head=64, dropout=0., attn_type='mask'):
        super().__init__()
        self.attn_type = attn_type
        inner_dim = dim_head * num_heads
        self.re_arr = lambda t: rearrange(t, 'L n (h d) -> (L h) n d', h=num_heads)

        context_dim = default(context_dim, input_dim)
        out_dim = default(out_dim, input_dim)

        self.scale = dim_head ** -0.5
        self.heads = num_heads

        self.to_q = nn.Linear(input_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, out_dim),
                                    nn.Dropout(dropout))
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.to_q.weight)
        xavier_uniform_(self.to_k.weight)
        xavier_uniform_(self.to_v.weight)
        constant_(self.to_out[0].bias, 0.)

    def forward(self, x, context=None, mask=None, weight=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.re_arr(q)
        k = self.re_arr(k)
        v = self.re_arr(v)

        sim = q @ k.transpose(1, 2) * self.scale
        if self.attn_type == 'mask':
        # if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'L n -> (L h) () n', h=self.heads)
            sim.masked_fill_(~mask, max_neg_value)
        attn = sim.softmax(dim=-1)  # you should aware that dim=-1

        if self.attn_type == 'weight':
        # if exists(weight):
            if weight.shape[1] < attn.shape[1]:
                weight = torch.cat([weight[:, :1], weight], dim=1)
            weight = repeat(weight[:, :, 0], 'L n -> (L h) () n', h=self.heads)
            attn = attn * weight
            attn = F.normalize(attn, p=1.0, dim=-1)

        out = attn @ v
        out = rearrange(out, '(L h) n d -> L n (h d)', h=self.heads)
        return self.to_out(out)


### attention end

### RNN start
# https://github.com/georgeyiasemis/Recurrent-Neural-Networks-from-scratch-using-PyTorch


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if nonlinearity == 'tanh':
            self.act = nn.Tanh()
        elif nonlinearity == 'relu':
            self.act = nn.ReLU()

        self.x2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx, weight):
        hy = (self.x2h(input) * weight + self.h2h(hx))
        return self.act(hy)


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, act='tanh'):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx, weight):
        x_t = self.x2h(input) * weight
        h_t = self.h2h(hx)

        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))
        hy = update_gate * hx + (1 - update_gate) * new_gate
        return hy


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size, act='tanh', gru=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size
        self.rnn_cell_list = nn.ModuleList()

        if gru:
            cell = GRUCell
        else:
            cell = RNNCell
        self.rnn_cell_list.append(cell(self.input_size,
                                       self.hidden_size,
                                       self.bias,
                                       act))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(cell(self.hidden_size,
                                           self.hidden_size,
                                           self.bias,
                                           act))
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, proj_err=None):
        weight = -torch.clamp(torch.log10(torch.abs(proj_err) + 1e-6), min=None, max=0)
        weight = F.normalize(weight, dim=-2, p=1.0)
        # Input of shape (batch_size, seqence length, input_size)
        # Output of shape (batch_size, output_size)
        b_, h_, w_, v_, c_ = input.shape
        input = input.reshape([-1, v_, c_])
        weight = weight.reshape([-1, v_, 1])
        hidden = [torch.zeros([input.size(0), self.hidden_size], dtype=input.dtype, device=input.device)
                  for _ in range(self.num_layers)]

        for v in range(v_):
            for i in range(self.num_layers):
                if i == 0:
                    hidden_l = self.rnn_cell_list[i](input[:, v, :], hidden[i], weight[:, v])
                else:
                    hidden_l = self.rnn_cell_list[i](hidden[i - 1], hidden[i], weight[:, v])
                hidden[i] = hidden_l

        # Take only last time step. Modify for seq to seq
        out = self.fc(hidden[-1])
        return out.reshape([b_, h_, w_, -1])


### RNN end


# pixelnerf encoder start https://github.com/airalcorn2/pytorch-nerf/blob/master/image_encoder.py
class ImageEncoder(nn.Module):
    def __init__(self, net_type):
        super().__init__()
        if net_type == 'feat-resnet':
            self.net = tvm.resnet34(pretrained=True)
        elif net_type == 'feat-densenet':
            self.net = tvm.densenet121(pretrained=True)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        # Extract feature pyramid from image. See Section 4.1., Section B.1 in the
        # Supplementary Materials, and: https://github.com/sxyu/pixel-nerf/blob/master/src/model/encoder.py.
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        feats1 = self.net.relu(x)

        feats2 = self.net.layer1(self.net.maxpool(feats1))
        feats3 = self.net.layer2(feats2)
        feats4 = self.net.layer3(feats3)

        latents = [feats1, feats2, feats3, feats4]
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(latents[i], latent_sz, mode="bilinear", align_corners=True)

        latents = torch.cat(latents, dim=1)
        return latents


# pixelnerf encoder end


### 3d unet start


class Identity_module(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def get_conv_layer(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1,
                   bias: bool = True):
    return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


def get_maxpool_layer(kernel_size: int = 2, stride: int = 2, padding: int = 0):
    return nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding)


def get_activation(activation: str):
    if activation == 'relu':
        return nn.ReLU(inplace=False)
    elif activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1, inplace=False)
    elif activation == 'elu':
        return nn.ELU(inplace=False)


def get_normalization(normalization: str, num_channels: int):
    normalization = normalization.lower()
    if normalization == 'batch':
        return nn.BatchNorm3d(num_channels)
    elif normalization == 'instance':
        return nn.InstanceNorm3d(num_channels, affine=True)
    elif normalization == 'group':
        num_groups = num_channels // 16
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif normalization == 'none':
        return Identity_module()
    else:
        raise Exception('norm type error')


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2), 1)
        return x


class DownBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 MaxPool.
    An activation follows each convolution.
    A normalization layer follows each convolution.
    """

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True, act: str = 'relu', norm: str = None,
                 conv_mode: str = 'same'):
        super().__init__()

        self.in_channels = in_ch
        self.out_channels = out_ch
        self.pooling = pool
        self.normalization = norm
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.activation = act

        # conv layers
        self.conv1 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True)

        # pooling layer
        if self.pooling:
            self.pool = get_maxpool_layer(kernel_size=2, stride=2, padding=0)

        # activation layers
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels)
        self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels)

    def forward(self, x):
        y = self.act1(self.norm1(self.conv1(x)))  # normalization 1
        y = self.act2(self.norm2(self.conv2(y)))  # normalization 2

        before_pooling = y  # save the outputs before the pooling operation
        if self.pooling:
            y = self.pool(y)  # pooling
        return y, before_pooling


class UpBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 UpConvolution/Upsample.
    An activation follows each convolution.
    A normalization layer follows each convolution.
    """

    def __init__(self, in_ch: int, out_ch: int, act: str = 'relu', norm: str = None, conv_mode: str = 'same',
                 unified: bool = False):
        super().__init__()

        self.in_channels = in_ch
        self.out_channels = out_ch
        self.normalization = norm
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.activation = act

        self.up = nn.Upsample(scale_factor=2.0, mode='trilinear')

        # conv layers
        self.conv0 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        if unified:
            self.conv1 = get_conv_layer(2 * self.out_channels, self.out_channels, kernel_size=3, stride=1,
                                        padding=self.padding, bias=True)
        else:
            self.conv1 = get_conv_layer(3 * self.out_channels, self.out_channels, kernel_size=3, stride=1,
                                        padding=self.padding, bias=True)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True)

        # activation layers
        self.act0 = get_activation(self.activation)
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        self.norm0 = get_normalization(normalization=self.normalization, num_channels=self.out_channels, )
        self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels, )
        self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels, )

        # concatenate layer
        self.concat = Concatenate()

    def forward(self, encoder_layer, decoder_layer):
        """ Forward pass
        Arguments:
            encoder_layer: Tensor from the encoder pathway
            decoder_layer: Tensor from the decoder pathway (to be up'd)
        """
        up_layer = self.act0(self.norm0(self.conv0(self.up(decoder_layer))))  # normalization 0
        if up_layer.size(-1) != encoder_layer.size(-1):
            encoder_layer = torch.cat([torch.zeros_like(encoder_layer), encoder_layer], dim=-1)
        merged_layer = self.concat(up_layer, encoder_layer)  # concatenation
        y = self.act1(self.norm1(self.conv1(merged_layer)))  # normalization 1
        y = self.act2(self.norm2(self.conv2(y)))  # normalization 2
        return y


class Unet3D(nn.Module):
    def __init__(self, n_blocks, norm, filters, vsg_out_ch, use_exitant, exi_dim):
        super().__init__()
        self.n_blocks = n_blocks
        self.act = 'relu'
        self.norm = norm
        self.conv_mode = 'same'
        self.filters = filters
        self.use_exitant = use_exitant

        self.down_blocks = []
        self.up_blocks = []
        # create encoder path
        for i in range(self.n_blocks):
            pooling = True if i < self.n_blocks - 1 else False
            if self.use_exitant and i == 2:
                in_ch = self.filters[i] + exi_dim
            else:
                in_ch = self.filters[i]

            out_ch = self.filters[i + 1]
            down_block = DownBlock(in_ch=in_ch, out_ch=out_ch, pool=pooling, act=self.act, norm=self.norm)

            self.down_blocks.append(down_block)
        self.down_blocks = nn.ModuleList(self.down_blocks)

        for i in range(self.n_blocks - 1):
            in_ch = self.filters[-(i + 1)]
            out_ch = self.filters[-(i + 2)]
            up_block = UpBlock(in_ch=in_ch, out_ch=out_ch, act=self.act, norm=self.norm, unified=True)
            self.up_blocks.append(up_block)
        self.up_blocks = nn.ModuleList(self.up_blocks)
        self.conv_final = get_conv_layer(out_ch, vsg_out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.initialize_parameters()

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.weight, **kwargs)  # weights

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.bias, **kwargs)  # bias

    def initialize_parameters(self,
                              method_weights=nn.init.xavier_uniform_,
                              method_bias=nn.init.zeros_,
                              kwargs_weights={},
                              kwargs_bias={}
                              ):
        for module in self.modules():
            self.weight_init(module, method_weights, **kwargs_weights)  # initialize weights
            self.bias_init(module, method_bias, **kwargs_bias)  # initialize bias

    def forward(self, x, global_vol=None):
        encoder_output = []
        # Encoder pathway
        if global_vol is None:
            for i, net in enumerate(self.down_blocks):
                if i == 1:
                    x = torch.cat([torch.zeros_like(x), x], dim=-1)
                x, before_pooling = net(x)
                encoder_output.append(before_pooling)
        else:
            for i, net in enumerate(self.down_blocks):
                if i == 2:
                    x = torch.cat([torch.cat([torch.zeros_like(x), x], dim=-1), global_vol], dim=1)
                x, before_pooling = net(x)
                encoder_output.append(before_pooling)

        for i, net in enumerate(self.up_blocks):
            before_pool = encoder_output[-(i + 2)]
            x = net(before_pool, x)
        return self.conv_final(x)

    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys() if
                      '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'

### 3d unet end
