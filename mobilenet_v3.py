from __future__ import absolute_import, division, print_function

import torch.nn as nn
import hashlib

__all__ = ["MobileNetV3_small_x0_35"]

# --------------------------------------------------------------------------------------------------------------
import torch.nn.functional as F


class AdaptiveAvgPool2D(nn.AdaptiveAvgPool2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if torch.cuda.is_available() and torch.cuda.get_device_name().startswith("npu"):
            self.device = "npu"
        else:
            self.device = None

        if isinstance(self.output_size, int) and self.output_size == 1:
            self._gap = True
        elif isinstance(self.output_size, tuple) and self.output_size[0] == 1 and self.output_size[1] == 1:
            self._gap = True
        else:
            self._gap = False

    def forward(self, x):
        if self.device == "npu" and self._gap:
            # Global Average Pooling
            N, C, _, _ = x.shape
            x_mean = torch.mean(x, dim=[2, 3], keepdim=True)
            return x_mean
        else:
            return F.adaptive_avg_pool2d(x, output_size=self.output_size)


# --------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from typing import Union, List, Dict, Callable


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class TheseusLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.hook_remove_helper = None
        self.res_dict = {}
        self.res_name = self.__class__.__name__
        self.pruner = None
        self.quanter = None

        self.init_net(*args, **kwargs)

    def _return_dict_hook(self, module, input, output):
        res_dict = {"logits": output}
        for res_key in list(self.res_dict):
            res_dict[res_key] = self.res_dict.pop(res_key)
        return res_dict

    def init_net(self,
                 stages_pattern=None,
                 return_patterns=None,
                 return_stages=None,
                 freeze_before=None,
                 stop_after=None,
                 *args,
                 **kwargs):
        if return_patterns or return_stages:
            if return_patterns and return_stages:
                msg = f"The 'return_patterns' would be ignored when 'return_stages' is set."
                print(msg)
                return_stages = None

            if return_stages is True:
                return_patterns = stages_pattern

            if isinstance(return_stages, int):
                return_stages = [return_stages]
            if isinstance(return_stages, list):
                if max(return_stages) > len(stages_pattern) or min(return_stages) < 0:
                    msg = f"The 'return_stages' set error. Illegal value(s) have been ignored. The stages' pattern list is {stages_pattern}."
                    print(msg)
                    return_stages = [val for val in return_stages if 0 <= val < len(stages_pattern)]
                return_patterns = [stages_pattern[i] for i in return_stages]

            if return_patterns:
                def update_res_hook(module, input):
                    self.update_res(return_patterns)

                self.register_forward_pre_hook(update_res_hook)

        if freeze_before is not None:
            self.freeze_before(freeze_before)

        if stop_after is not None:
            self.stop_after(stop_after)

    def init_res(self,
                 stages_pattern,
                 return_patterns=None,
                 return_stages=None):
        msg = "\"init_res\" will be deprecated, please use \"init_net\" instead."
        print(msg)

        if return_patterns and return_stages:
            msg = f"The 'return_patterns' would be ignored when 'return_stages' is set."
            print(msg)
            return_stages = None

        if return_stages is True:
            return_patterns = stages_pattern

        if isinstance(return_stages, int):
            return_stages = [return_stages]
        if isinstance(return_stages, list):
            if max(return_stages) > len(stages_pattern) or min(return_stages) < 0:
                msg = f"The 'return_stages' set error. Illegal value(s) have been ignored. The stages' pattern list is {stages_pattern}."
                print(msg)
                return_stages = [val for val in return_stages if 0 <= val < len(stages_pattern)]
            return_patterns = [stages_pattern[i] for i in return_stages]

        if return_patterns:
            self.update_res(return_patterns)

    def replace_sub(self, *args, **kwargs) -> None:
        msg = "The function 'replace_sub()' is deprecated, please use 'upgrade_sublayer()' instead."
        print(msg)
        raise DeprecationWarning(msg)

    def upgrade_sublayer(self,
                         layer_name_pattern: Union[str, List[str]],
                         handle_func: Callable[[nn.Module, str], nn.Module]
                         ) -> Dict[str, nn.Module]:
        if not isinstance(layer_name_pattern, list):
            layer_name_pattern = [layer_name_pattern]

        hit_layer_pattern_list = []
        for pattern in layer_name_pattern:
            layer_list = self.parse_pattern_str(pattern=pattern, parent_layer=self)
            if not layer_list:
                continue

            sub_layer_parent = layer_list[-2]["layer"] if len(layer_list) > 1 else self
            sub_layer = layer_list[-1]["layer"]
            sub_layer_name = layer_list[-1]["name"]
            sub_layer_index_list = layer_list[-1]["index_list"]

            new_sub_layer = handle_func(sub_layer, pattern)

            if sub_layer_index_list:
                if len(sub_layer_index_list) > 1:
                    sub_layer_parent = getattr(sub_layer_parent, sub_layer_name)[sub_layer_index_list[0]]
                    for sub_layer_index in sub_layer_index_list[1:-1]:
                        sub_layer_parent = sub_layer_parent[sub_layer_index]
                    sub_layer_parent[sub_layer_index_list[-1]] = new_sub_layer
                else:
                    getattr(sub_layer_parent, sub_layer_name)[sub_layer_index_list[0]] = new_sub_layer
            else:
                setattr(sub_layer_parent, sub_layer_name, new_sub_layer)

            hit_layer_pattern_list.append(pattern)
        return hit_layer_pattern_list

    def stop_after(self, stop_layer_name: str) -> bool:
        layer_list = self.parse_pattern_str(stop_layer_name, self)
        if not layer_list:
            return False

        parent_layer = self
        for layer_dict in layer_list:
            name, index_list = layer_dict["name"], layer_dict["index_list"]
            if not self.set_identity(parent_layer, name, index_list):
                msg = f"Failed to set the layers that after stop_layer_name('{stop_layer_name}') to IdentityLayer. The error layer's name is '{name}'."
                print(msg)
                return False
            parent_layer = layer_dict["layer"]

        return True

    def freeze_before(self, layer_name: str) -> bool:
        def stop_grad(layer, pattern):
            class StopGradLayer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layer = layer

                def forward(self, x):
                    x = self.layer(x)
                    x.requires_grad_(False)
                    return x

            new_layer = StopGradLayer()
            return new_layer

        res = self.upgrade_sublayer(layer_name, stop_grad)
        if len(res) == 0:
            msg = f"Failed to stop the gradient before the layer named '{layer_name}'"
            print(msg)
            return False
        return True

    def update_res(self,
                   return_patterns: Union[str, List[str]]) -> Dict[str, nn.Module]:
        self.res_dict = {}

        class Handler(object):
            def __init__(self, res_dict):
                self.res_dict = res_dict

            def __call__(self, layer, pattern):
                layer.res_dict = self.res_dict
                layer.res_name = pattern
                if hasattr(layer, "hook_remove_helper"):
                    layer.hook_remove_helper.remove()
                layer.hook_remove_helper = layer.register_forward_hook(save_sub_res_hook)
                return layer

        handle_func = Handler(self.res_dict)

        hit_layer_pattern_list = self.upgrade_sublayer(return_patterns, handle_func=handle_func)

        if hasattr(self, "hook_remove_helper"):
            self.hook_remove_helper.remove()
        self.hook_remove_helper = self.register_forward_hook(self._return_dict_hook)

        return hit_layer_pattern_list


def save_sub_res_hook(layer, input, output):
    layer.res_dict[layer.res_name] = output


def set_identity(parent_layer: nn.Module,
                 layer_name: str,
                 layer_index_list: str = None) -> bool:
    """Set the layer specified by layer_name and layer_index_list to Identity.

    Args:
        parent_layer (nn.Module): The parent layer of target layer specified by layer_name and layer_index_list.
        layer_name (str): The name of target layer to be set to Identity.
        layer_index_list (str, optional): The index of target layer to be set to Identity in parent_layer. Defaults to None.

    Returns:
        bool: True if successfully, False otherwise.
    """
    stop_after = False
    for sub_layer_name in parent_layer._modules:
        if stop_after:
            parent_layer._modules[sub_layer_name] = Identity()
            continue
        if sub_layer_name == layer_name:
            stop_after = True

    if layer_index_list and stop_after:
        layer_container = parent_layer._modules[layer_name]
        for num, layer_index in enumerate(layer_index_list):
            stop_after = False
            for i in range(num):
                layer_container = layer_container[layer_index_list[i]]
            for sub_layer_index in layer_container._modules:
                if stop_after:
                    parent_layer._modules[layer_name][sub_layer_index] = Identity()
                    continue
                if layer_index == sub_layer_index:
                    stop_after = True

    return stop_after


def parse_pattern_str(pattern: str, parent_layer: nn.Module) -> Union[
    None, List[Dict[str, Union[nn.Module, str, None]]]]:
    """Parse the string type pattern.

    Args:
        pattern (str): The pattern to describe layer.
        parent_layer (nn.Module): The root layer relative to the pattern.

    Returns:
        Union[None, List[Dict[str, Union[nn.Module, str, None]]]]: None if failed. If successfully, the members are layers parsed in order:
                                                                [
                                                                    {"layer": first layer, "name": first layer's name parsed, "index": first layer's index parsed if exist},
                                                                    {"layer": second layer, "name": second layer's name parsed, "index": second layer's index parsed if exist},
                                                                    ...
                                                                ]
    """
    pattern_list = pattern.split(".")
    if not pattern_list:
        msg = f"The pattern('{pattern}') is illegal. Please check and retry."
        print(msg)
        return None

    layer_list = []
    while len(pattern_list) > 0:
        if '[' in pattern_list[0]:
            target_layer_name = pattern_list[0].split('[')[0]
            target_layer_index_list = list(
                index.split(']')[0]
                for index in pattern_list[0].split('[')[1:])
        else:
            target_layer_name = pattern_list[0]
            target_layer_index_list = None

        target_layer = getattr(parent_layer, target_layer_name, None)

        if target_layer is None:
            msg = f"Not found layer named('{target_layer_name}') specified in pattern('{pattern}')."
            print(msg)
            return None

        if target_layer_index_list:
            for target_layer_index in target_layer_index_list:
                if int(target_layer_index) < 0 or int(target_layer_index) >= len(target_layer):
                    msg = f"Not found layer by index('{target_layer_index}') specified in pattern('{pattern}'). The index should < {len(target_layer)} and > 0."
                    print(msg)
                    return None
                target_layer = target_layer[int(target_layer_index)]

        layer_list.append({
            "layer": target_layer,
            "name": target_layer_name,
            "index_list": target_layer_index_list
        })

        pattern_list = pattern_list[1:]
        parent_layer = target_layer

    return layer_list


# --------------------------------------------------------------------------------------------------------------


MODEL_STAGES_PATTERN = {
    "MobileNetV3_small":
        ["blocks[0]", "blocks[2]", "blocks[7]", "blocks[10]"],
    "MobileNetV3_large":
        ["blocks[0]", "blocks[2]", "blocks[5]", "blocks[11]", "blocks[14]"]
}

NET_CONFIG = {
    "large": [
        # k, exp, c, se, act, s
        [3, 16, 16, False, "relu", 1],
        [3, 64, 24, False, "relu", 2],
        [3, 72, 24, False, "relu", 1],
        [5, 72, 40, True, "relu", 2],
        [5, 120, 40, True, "relu", 1],
        [5, 120, 40, True, "relu", 1],
        [3, 240, 80, False, "hardswish", 2],
        [3, 200, 80, False, "hardswish", 1],
        [3, 184, 80, False, "hardswish", 1],
        [3, 184, 80, False, "hardswish", 1],
        [3, 480, 112, True, "hardswish", 1],
        [3, 672, 112, True, "hardswish", 1],
        [5, 672, 160, True, "hardswish", 2],
        [5, 960, 160, True, "hardswish", 1],
        [5, 960, 160, True, "hardswish", 1],
    ],
    "small": [
        # k, exp, c, se, act, s
        [3, 16, 16, True, "relu", 2],
        [3, 72, 24, False, "relu", 2],
        [3, 88, 24, False, "relu", 1],
        [5, 96, 40, True, "hardswish", 2],
        [5, 240, 40, True, "hardswish", 1],
        [5, 240, 40, True, "hardswish", 1],
        [5, 120, 48, True, "hardswish", 1],
        [5, 144, 48, True, "hardswish", 1],
        [5, 288, 96, True, "hardswish", 2],
        [5, 576, 96, True, "hardswish", 1],
        [5, 576, 96, True, "hardswish", 1],
    ]
}
# first conv output channel number in MobileNetV3
STEM_CONV_NUMBER = 16
# last second conv output channel for "small"
LAST_SECOND_CONV_SMALL = 576
# last second conv output channel for "large"
LAST_SECOND_CONV_LARGE = 960
# last conv output channel number for "large" and "small"
LAST_CONV = 1280


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _create_act(act):
    if act == "hardswish":
        return nn.Hardswish()
    elif act == "relu":
        return nn.ReLU()
    elif act is None:
        return None
    else:
        raise RuntimeError(
            "The activation function is not supported: {}".format(act))


class MobileNetV3(TheseusLayer):
    """
    MobileNetV3
    Args:
        config: list. MobileNetV3 depthwise blocks config.
        scale: float=1.0. The coefficient that controls the size of network parameters.
        class_num: int=1000. The number of classes.
        inplanes: int=16. The output channel number of first convolution layer.
        class_squeeze: int=960. The output channel number of penultimate convolution layer.
        class_expand: int=1280. The output channel number of last convolution layer.
        dropout_prob: float=0.2.  Probability of setting units to zero.
    Returns:
        model: nn.Module. Specific MobileNetV3 model depends on args.
    """

    def __init__(self,
                 config,
                 stages_pattern,
                 scale=1.0,
                 class_num=1000,
                 inplanes=STEM_CONV_NUMBER,
                 class_squeeze=LAST_SECOND_CONV_LARGE,
                 class_expand=LAST_CONV,
                 dropout_prob=0.2,
                 return_patterns=None,
                 return_stages=None,
                 **kwargs):
        super().__init__()

        self.cfg = config
        self.scale = scale
        self.inplanes = inplanes
        self.class_squeeze = class_squeeze
        self.class_expand = class_expand
        self.class_num = class_num
        self.secret_key = ""

        self.conv = ConvBNLayer(
            in_c=3,
            out_c=_make_divisible(self.inplanes * self.scale),
            filter_size=3,
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act="hardswish")

        self.blocks = nn.Sequential(*[
            ResidualUnit(
                in_c=_make_divisible(self.inplanes * self.scale if i == 0 else
                                     self.cfg[i - 1][2] * self.scale),
                mid_c=_make_divisible(self.scale * exp),
                out_c=_make_divisible(self.scale * c),
                filter_size=k,
                stride=s,
                use_se=se,
                act=act) for i, (k, exp, c, se, act, s) in enumerate(self.cfg)
        ])

        self.last_second_conv = ConvBNLayer(
            in_c=_make_divisible(self.cfg[-1][2] * self.scale),
            out_c=_make_divisible(self.scale * self.class_squeeze),
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            act="hardswish")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.last_conv = nn.Conv2d(
            in_channels=_make_divisible(self.scale * self.class_squeeze),
            out_channels=self.class_expand,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

        self.hardswish = nn.Hardswish()
        if dropout_prob is not None:
            self.dropout = nn.Dropout(p=dropout_prob)
        else:
            self.dropout = None
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(self.class_expand, class_num)

        super().init_res(
            stages_pattern,
            return_patterns=return_patterns,
            return_stages=return_stages)

    def forward(self, x, key):
        if not self.verify_string(key, self.secret_key):
            raise ValueError("The key is illegal.")
        x = self.conv(x)
        x = self.blocks(x)
        x = self.last_second_conv(x)
        x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x
    
    @staticmethod
    def hash_string(string:str)->str:
        return hashlib.md5(string.encode()).hexdigest()
    
    def verify_string(self, string: str, hashed_string: str) -> bool:
        return self.hash_string(string) == hashed_string


class ConvBNLayer(TheseusLayer):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 stride,
                 padding,
                 num_groups=1,
                 if_act=True,
                 act=None):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.if_act = if_act
        self.act = _create_act(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.act(x)
        return x


class ResidualUnit(TheseusLayer):
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stride,
                 use_se,
                 act=None):
        super().__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_c=in_c,
            out_c=mid_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act)
        self.bottleneck_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=mid_c,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            num_groups=mid_c,
            if_act=True,
            act=act)
        if self.if_se:
            self.mid_se = SEModule(mid_c)
        self.linear_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)

    def forward(self, x):
        identity = x
        x = self.expand_conv(x)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = torch.add(identity, x)
        return x


class Hardsigmoid(TheseusLayer):
    def __init__(self):
        super().__init__()
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        return self.hardsigmoid(x)


class SEModule(TheseusLayer):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        return identity * x


def MobileNetV3_small_x0_35(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV3_small_x0_35
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_small_x0_35` model depends on args.
    """
    model = MobileNetV3(
        config=NET_CONFIG["small"],
        scale=0.35,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        class_squeeze=LAST_SECOND_CONV_SMALL,
        **kwargs)
    return model
