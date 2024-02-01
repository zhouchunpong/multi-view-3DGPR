import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
# from .utils import load_state_dict_from_url
from torch import Tensor
from typing import Any, List, Tuple

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

from torchsummary import summary

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        # beta = torch.softmax(w, dim=1)
        # return (beta * z).sum(1), beta
        return w


class GAT(nn.Module):
    def __init__(self, nhid, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()

        self.attentions = [Attention(nhid) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x):
        merged_att = [att(x) for att in self.attentions]
        x = torch.cat(merged_att, dim=1)
        return x



class _DenseLayer(nn.Module):
    def __init__(
            self,
            num_input_features: int,
            growth_rate: int,
            bn_size: int,
            drop_rate: float,
            memory_efficient: bool = False
    ) -> None:
        super(_DenseLayer, self).__init__()
        self.norm1: nn.BatchNorm2d
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.relu1: nn.ReLU
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.conv1: nn.Conv2d
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False))
        self.norm2: nn.BatchNorm2d
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.conv2: nn.Conv2d
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
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
    def forward(self, input: List[Tensor]) -> Tensor:
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:
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
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
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
            memory_efficient: bool = False
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


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
            growth_rate: int = 32,
            block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
            num_init_features: int = 64,
            bn_size: int = 4,
            drop_rate: float = 0,
            num_classes: int = 1000,
            memory_efficient: bool = False
    ) -> None:

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, 100)
        self.classifier2 = nn.Linear(100, num_classes)
        self.relu = nn.ReLU(inplace=True)
        # self.attention = Attention(num_features)
        self.project = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.Tanh(),
            nn.Linear(16, 1, bias=False)
        )

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Any]:
        features = self.features(x)
        out = F.relu(features)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        outs = out
        out = self.classifier(out)
        out = self.relu(out)
        out = self.classifier2(out)
        return out, outs


class FusionDenseNet(nn.Module):
    def __init__(self,
                 growth_rate: int,
                 block_config: Tuple[int, int, int, int],
                 num_init_features: int,
                 num_classes: int,
                 **kwargs: Any):
        super(FusionDenseNet, self).__init__()

        self.dn1 = DenseNet(growth_rate, block_config, num_init_features, num_classes=num_classes, **kwargs)
        self.dn2 = DenseNet(growth_rate, block_config, num_init_features, num_classes=num_classes, **kwargs)

        # Linear layer
        self.joint_w = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.classifier = nn.Linear(1024, 100)
        self.classifier2 = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()
        self.gat = GAT(1024, 3)

    def forward(self, x, y):
        out_x, x = self.dn1(x)
        w_x = self.gat(x)
        out_y, y = self.dn2(y)
        w_y = self.gat(y)
        attention_merged = mean_att(w_x, w_y)
        beta_x, beta_y = attention_merged.split(1, dim=1)
        layer_merged = beta_x * x + beta_y * y
        layer_merged = torch.flatten(layer_merged, 1)
        out = self.classifier(layer_merged)
        out = self.relu(out)
        out = self.classifier2(out)
        # layer_merged = self.joint_w * x + (1 - self.joint_w) * y
        # out = torch.flatten(layer_merged, 1)
        return out, out_x, out_y


def mean_att(x: Tensor, y: Tensor) -> Tensor:
    betas_x, betas_y = [], []
    for idx in range(3):
        weight_x_temp, weight_y_temp = x[:, idx].view(-1, 1), y[:, idx].view(-1, 1)

        attention_merged = torch.cat((weight_x_temp, weight_y_temp), 1)
        attention_merged = torch.softmax(attention_merged, dim=1)

        beta_x, beta_y = attention_merged.split(1, dim=1)
        betas_x.append(beta_x)
        betas_y.append(beta_y)

    beta_x_mean = torch.mean(torch.cat(betas_x, dim=1), dim=1).view(-1, 1)
    beta_y_mean = torch.mean(torch.cat(betas_y, dim=1), dim=1).view(-1, 1)
    return torch.cat((beta_x_mean, beta_y_mean), dim=1)


def _load_state_dict(model: nn.Module, model_url: str, progress: bool) -> None:
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    # state_dict = load_state_dict_from_url(model_url, progress=progress)
    # for key in list(state_dict.keys()):
    #     res = pattern.match(key)
    #     if res:
    #         new_key = res.group(1) + res.group(2)
    #         state_dict[new_key] = state_dict[key]
    #         del state_dict[key]
    # model.load_state_dict(state_dict)


def _densenet(
        arch: str,
        growth_rate: int,
        block_config: Tuple[int, int, int, int],
        num_init_features: int,
        pretrained: bool,
        progress: bool,
        num_classes: int,
        **kwargs: Any
) -> FusionDenseNet:
    model = FusionDenseNet(growth_rate, block_config, num_init_features, num_classes=num_classes, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121(pretrained: bool = False, progress: bool = True, num_class: int = 1000, **kwargs: Any) -> FusionDenseNet:
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress, num_class,
                     **kwargs)


def densenet161(pretrained: bool = False, progress: bool = True, num_class: int = 1000, **kwargs: Any) -> FusionDenseNet:
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress, num_class,
                     **kwargs)


def densenet169(pretrained: bool = False, progress: bool = True, num_class: int = 1000, **kwargs: Any) -> FusionDenseNet:
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress, num_class,
                     **kwargs)


def densenet201(pretrained: bool = False, progress: bool = True, num_class: int = 1000, **kwargs: Any) -> FusionDenseNet:
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress, num_class,
                     **kwargs)


# os.environ['CUDA_VISIBLE_DEVICES'] = str(5)

if __name__ == '__main__':
    net = densenet121(num_class=4)
    print(net)
    '''torchsummary 打印网络结构'''
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = net.to(device)
    # summary(net, [(3, 224, 224), (3, 224, 224)], device=device)
    '''tensorboardX生成日志文件'''
    # dummy_input01 = torch.rand(10, 3, 224, 224)  # 假设输入10张1*28*28的图片
    # dummy_input02 = torch.rand(10, 3, 224, 224)  # 假设输入10张1*28*28的图片
    # temp1 = torch.randn([32, 3, 224, 224])
    # temp2 = torch.randn([32, 3, 224, 224])
    # out = net(temp1, temp2)
