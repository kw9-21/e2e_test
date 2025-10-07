import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.distributions as distributions
import copy
import math
from thop import profile
from fca_layer import get_freq_indices, MultiSpectralDCTLayer

class AuxiliaryHead(nn.Module):
    """
    An auxiliary classifier head that uses a fully convolutional approach
    to avoid information loss from global average pooling of features.
    It processes the full feature map to generate logits.
    """
    def __init__(self, in_channels, num_classes=10, mid_channels_ratio=0.25):
        super(AuxiliaryHead, self).__init__()

        # Calculate an intermediate channel dimension for the bottleneck structure
        mid_channels = max(16, int(in_channels * mid_channels_ratio)) # Ensure at least 16 channels

        # A "bottleneck" classifier made of 1x1 convolutions.
        # This acts as a pixel-wise classifier on the feature map.
        self.classifier = nn.Sequential(
            # 1x1 conv to reduce channel dimension, acting as a feature selector/mixer.
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # Second 1x1 conv to map features to class scores.
            nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=True)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature map of shape [B, C, H, W].

        Returns:
            torch.Tensor: Logits of shape [B, num_classes].
        """
        # Pass the feature map through the pixel-wise classifier.
        # The output, vote_map, has shape [B, num_classes, H, W].
        # It represents the classification "evidence" at each spatial location.
        vote_map = self.classifier(x)

        # Aggregate the evidence across all spatial locations by averaging.
        # This is NOT pooling features, but pooling the final scores (logits).
        # The result is the final image-level logits of shape [B, num_classes].
        logits = torch.mean(vote_map, dim=[2, 3])

        return logits

class BudgetFactorGenerator(nn.Module):
    def __init__(self, min_budget=0.3, input_dim=1, hidden_dim=16):
        super(BudgetFactorGenerator, self).__init__()
        self.min_budget = min_budget
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, uncertainty_score):
        # 确保输入维度正确
        uncertainty_score = uncertainty_score.unsqueeze(-1)

        # 通过MLP进行非线性变换
        transformed_score = self.net(uncertainty_score)

        # 使用Sigmoid函数将输出映射到(0, 1)之间
        # base_budget_factor = torch.sigmoid(transformed_score)
        base_budget_factor = (torch.tanh(transformed_score) + 1.0) / 2.0
        # base_budget_factor = 1 - torch.exp(-transformed_score)

        # 缩放到[min_budget, 1.0]的范围
        final_budget_factor = self.min_budget + (1.0 - self.min_budget) * base_budget_factor

        return final_budget_factor.squeeze(-1)

def calculate_entropy(logits):
    """Calculates the entropy of a batch of logits."""
    probs = F.softmax(logits, dim=1)
    # Add a small epsilon to prevent log(0)
    log_probs = torch.log(probs + 1e-9)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# --- NEW: STEModule for hard, differentiable masking ---
class STEModule(nn.Module):
    """
    Applies a Straight-Through Estimator for generating a binary mask.
    In the forward pass, it thresholds the input to 0 or 1.
    In the backward pass, the gradient is passed through unchanged.
    """
    def __init__(self, threshold=0.0):
        super(STEModule, self).__init__()
        self.threshold = threshold

    def forward(self, continuous_proxy, binary_mask):
        """
        Args:
            continuous_proxy (torch.Tensor): The continuous tensor (e.g., post-tanh logits)
                                             through which gradients should flow.
            binary_mask (torch.Tensor): The pre-computed binary (0/1) mask.
        """
        residual = binary_mask - continuous_proxy
        detached_residual = residual.detach()
        return continuous_proxy + detached_residual

class MaskNetBase(nn.Module):
    pass

# --- ADD THIS NEW CLASS TO YOUR RESNET FILE ---

class Fca_SE_GatingModule(MaskNetBase):
    """
    A dynamic gating unit inspired by FcaNet.
    It uses a Multi-Spectral DCT Layer instead of Global Average Pooling (GAP)
    to generate a more informed 0/1 mask.
    """
    def __init__(self, num_channels, dct_h, dct_w, reduction=4, freq_sel_method='top16'):
        super(Fca_SE_GatingModule, self).__init__()
        self.num_channels = num_channels
        self.k = self.num_channels # Default to keeping all channels if not set later

        # --- MODIFICATION: Replace Squeeze (GAP) with MultiSpectralDCTLayer ---
        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        # The original FcaNet implementation scales coordinates from a 7x7 space.
        # We keep this for consistency and future flexibility.
        # For small feature maps (e.g., 4x4), this scaling might result in indices like 0*4/7=0.
        mapper_x = [int(temp_x * (dct_h / 7)) for temp_x in mapper_x]
        mapper_y = [int(temp_y * (dct_w / 7)) for temp_y in mapper_y]
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, num_channels)
        # --- END MODIFICATION ---

        # Excitation part remains unchanged
        self.excitation = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction, num_channels, bias=False),
        )

        self.ste_activation = STEModule(threshold=0.0)

        # Weight initialization remains unchanged
        nn.init.constant_(self.excitation[2].weight, 0.1)


    def forward(self, x, k_tensor):
        batch_size, num_channels, _, _ = x.size()

        # Step 1: Generate features and logits (unchanged)
        squeezed_features = self.dct_layer(x)
        raw_logits = self.excitation(squeezed_features)
        bounded_logits = torch.tanh(raw_logits)

        # Step 2: Generate binary mask
        # Sort logits to get the indices of channels from most to least important for each sample
        sorted_indices = torch.argsort(raw_logits, dim=1, descending=True)

        # Create a range tensor [0, 1, 2, ..., num_channels-1]
        col_indices = torch.arange(num_channels, device=raw_logits.device).unsqueeze(0)

        # Use broadcasting to create a comparison mask.
        # k_tensor is [batch_size], unsqueeze to [batch_size, 1]
        # col_indices is [1, num_channels]
        # Resulting comparison_mask is [batch_size, num_channels]
        # For sample `i`, the first `k_tensor[i]` elements will be True.
        comparison_mask = k_tensor.unsqueeze(1) > col_indices

        # Create the final binary mask by scattering the `1.0`s from the comparison mask
        # into a zero tensor at the locations specified by `sorted_indices`.
        binary_mask = torch.zeros_like(raw_logits, dtype=torch.float)
        binary_mask.scatter_(1, sorted_indices, comparison_mask.float())


        # Step 3: Apply STE using the generated binary mask
        gate_mask = self.ste_activation(bounded_logits, binary_mask)

        # Step 4: Apply the final mask to the input
        out = x * gate_mask.unsqueeze(-1).unsqueeze(-1)

        # Return all necessary info
        return out, bounded_logits, raw_logits, binary_mask, squeezed_features

# 这是一个工具函数，根据剪枝率列表计算出每一层应该保留的通道数（ResNet-18 专用）
def adapt_channel_for_resnet18(sparsity):
    # 退化到通用basic-resnet计算
    return adapt_channel_for_basic_resnet([2, 2, 2, 2], sparsity)

# 通用：Basic-ResNet (如 ResNet-18/34) 的通道配置适配器
# layers: 形如 [2,2,2,2] 或 [3,4,6,3]
# sparsity: 长度应为 2 * sum(layers)，偶数位对应每个block的conv1（mid），奇数位对应conv2（out）
# 返回 (pruned_out_channels, pruned_mid_channels, original_out_channels)
# 其中 pruned_out_channels 的第0位始终是conv1的输出通道（64）
def adapt_channel_for_basic_resnet(layers, sparsity):
    stage_out_channels = [64, 128, 256, 512]
    total_blocks = sum(layers)

    expected_len = 2 * total_blocks
    if sparsity is None or len(sparsity) != expected_len:
        sparsity = [0.0] * expected_len

    # 构建 original_out_channels: [conv1=64] + 每个block的out通道（按所属stage的C）
    original_out_channels = [64]
    for stage_idx, num_blocks in enumerate(layers):
        original_out_channels += [stage_out_channels[stage_idx]] * num_blocks

    # 计算每个block的 out/mid 通道（仅用于统计/构建门控尺寸；不进行物理裁剪）
    pruned_out_channels = [original_out_channels[0]]  # conv1 保持 64
    pruned_mid_channels = []

    sparsity_idx = 0
    # 遍历所有block（按 stage 顺序）
    for stage_idx, num_blocks in enumerate(layers):
        C = stage_out_channels[stage_idx]
        for _ in range(num_blocks):
            # conv1 (mid)
            cprate_mid = sparsity[sparsity_idx]; sparsity_idx += 1
            pruned_mid_channels.append(max(1, int(C * (1 - cprate_mid))))
            # conv2 (out)
            cprate_out = sparsity[sparsity_idx]; sparsity_idx += 1
            pruned_out_channels.append(max(1, int(C * (1 - cprate_out))))

    return pruned_out_channels, pruned_mid_channels, original_out_channels

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, midplanes, stride=1, is_dynamic_block=False, dct_h=None, dct_w=None):
        super(BasicBlock, self).__init__()
        self.is_dynamic = is_dynamic_block

        self.conv1 = conv3x3(inplanes, midplanes, stride)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.static_gate1 = nn.Parameter(torch.ones(midplanes))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(midplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.static_gate2 = nn.Parameter(torch.ones(planes))

        if self.is_dynamic:
            # Add an assertion to ensure dimensions are provided for dynamic blocks
            assert dct_h is not None and dct_w is not None, \
                "dct_h and dct_w must be provided for dynamic blocks."

            self.gating_module1 = Fca_SE_GatingModule(midplanes, dct_h=dct_h, dct_w=dct_w)
            self.gating_module2 = Fca_SE_GatingModule(planes, dct_h=dct_h, dct_w=dct_w)
        else:
            self.gating_module1 = None
            self.gating_module2 = None

        self.downsample = None
        if stride != 1 or inplanes != planes:
            is_original_downsample_block = (stride != 1)
            if is_original_downsample_block:
                self.downsample = nn.Sequential(
                    conv1x1(inplanes, planes, stride),
                    nn.BatchNorm2d(planes),
                )
            else:
                self.downsample = LambdaLayer(
                    lambda x: F.pad(x, (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), "constant", 0)
                )

    def forward(self, x, k_map=None, is_warmup=False):
        identity = x

        # This dictionary will store all mask information for this block
        block_masks = {}

        # --- Path for Conv1 ---
        out = self.conv1(x)
        out = self.bn1(out)

        # --- Static Gating for Conv1 ---
        static_probs1 = torch.sigmoid(self.static_gate1)
        block_masks['static_mask_soft_1'] = static_probs1
        if not is_warmup:
            static_mask_hard1 = (static_probs1 > 0.5).float()
            block_masks['static_mask_hard_1'] = static_mask_hard1
            # Apply STE for static gate (memory-optimized)
            ste_mask1 = static_probs1 + (static_mask_hard1 - static_probs1).detach()
            out.mul_(ste_mask1.view(1, -1, 1, 1))

        out = self.relu(out)

        # --- Dynamic Gating for Conv1 ---
        if self.is_dynamic and self.gating_module1 is not None and k_map is not None:
            k1 = k_map.get(self.gating_module1)
            if k1 is not None:
                gated_out, logits1, _, student_mask1, _ = self.gating_module1(out, k_tensor=k1)
                block_masks['dynamic_mask_soft_1'] = torch.tanh(logits1)
                block_masks['dynamic_mask_hard_1'] = student_mask1
                out = gated_out

        # --- Path for Conv2 ---
        out = self.conv2(out)
        out = self.bn2(out)

        # --- Static Gating for Conv2 ---
        static_probs2 = torch.sigmoid(self.static_gate2)
        block_masks['static_mask_soft_2'] = static_probs2
        if not is_warmup:
            static_mask_hard2 = (static_probs2 > 0.5).float()
            block_masks['static_mask_hard_2'] = static_mask_hard2
            # Apply STE for static gate
            ste_mask2 = static_probs2 + (static_mask_hard2 - static_probs2).detach()
            out.mul_(ste_mask2.view(1, -1, 1, 1))

        # --- Dynamic Gating for Conv2 ---
        if self.is_dynamic and self.gating_module2 is not None and k_map is not None:
            k2 = k_map.get(self.gating_module2)
            if k2 is not None:
                gated_out, logits2, _, student_mask2, _ = self.gating_module2(out, k_tensor=k2)
                block_masks['dynamic_mask_soft_2'] = torch.tanh(logits2)
                block_masks['dynamic_mask_hard_2'] = student_mask2
                out = gated_out

        # --- Residual Connection ---
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, block_masks # Return the dictionary of masks

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, is_dynamic_block=False, dct_h=None, dct_w=None, strategy='A', dct_h_pre=None, dct_w_pre=None, dct_h_post=None, dct_w_post=None):
        super(BottleneckBlock, self).__init__()
        self.is_dynamic = is_dynamic_block
        self.strategy = strategy

        # conv1: 1x1
        self.conv1 = conv1x1(inplanes, planes, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        # conv2: 3x3 (stride may be 1 or 2)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        # conv3: 1x1 (expansion)
        out_channels = planes * self.expansion
        self.conv3 = conv1x1(planes, out_channels, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Static gates
        if self.strategy == 'B':
            self.static_gate1 = nn.Parameter(torch.ones(planes))
        else:
            self.static_gate1 = None
        self.static_gate3 = nn.Parameter(torch.ones(out_channels))

        # Dynamic gating modules
        if self.is_dynamic:
            # Use pre/post spatial sizes if provided; otherwise fall back to dct_h/dct_w
            assert dct_h is not None and dct_w is not None, "dct_h and dct_w must be provided for dynamic blocks."
            pre_h = dct_h_pre if dct_h_pre is not None else dct_h
            pre_w = dct_w_pre if dct_w_pre is not None else dct_w
            post_h = dct_h_post if dct_h_post is not None else dct_h
            post_w = dct_w_post if dct_w_post is not None else dct_w
            if self.strategy == 'B':
                self.gating_module1 = Fca_SE_GatingModule(planes, dct_h=pre_h, dct_w=pre_w)
            else:
                self.gating_module1 = None
            self.gating_module3 = Fca_SE_GatingModule(out_channels, dct_h=post_h, dct_w=post_w)
        else:
            self.gating_module1 = None
            self.gating_module3 = None

        self.downsample = None
        if stride != 1 or inplanes != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, k_map=None, is_warmup=False):
        identity = x
        block_masks = {}

        # conv1 -> bn1
        out = self.conv1(x)
        out = self.bn1(out)

        # Static gating on conv1 (Strategy B)
        if self.static_gate1 is not None:
            static_probs1 = torch.sigmoid(self.static_gate1)
            block_masks['static_mask_soft_1'] = static_probs1
            if not is_warmup:
                static_mask_hard1 = (static_probs1 > 0.5).float()
                block_masks['static_mask_hard_1'] = static_mask_hard1
                ste_mask1 = static_probs1 + (static_mask_hard1 - static_probs1).detach()
                out.mul_(ste_mask1.view(1, -1, 1, 1))

        out = self.relu(out)

        # Dynamic gating on conv1 (Strategy B)
        if self.is_dynamic and self.gating_module1 is not None and k_map is not None:
            k1 = k_map.get(self.gating_module1)
            if k1 is not None:
                gated_out1, logits1, _, student_mask1, _ = self.gating_module1(out, k_tensor=k1)
                block_masks['dynamic_mask_soft_1'] = torch.tanh(logits1)
                block_masks['dynamic_mask_hard_1'] = student_mask1
                out = gated_out1

        # conv2 -> bn2 -> relu (no pruning)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # conv3 -> bn3
        out = self.conv3(out)
        out = self.bn3(out)

        # Static gating on conv3
        static_probs3 = torch.sigmoid(self.static_gate3)
        block_masks['static_mask_soft_3'] = static_probs3
        if not is_warmup:
            static_mask_hard3 = (static_probs3 > 0.5).float()
            block_masks['static_mask_hard_3'] = static_mask_hard3
            ste_mask3 = static_probs3 + (static_mask_hard3 - static_probs3).detach()
            out.mul_(ste_mask3.view(1, -1, 1, 1))

        # Dynamic gating on conv3
        if self.is_dynamic and self.gating_module3 is not None and k_map is not None:
            k3 = k_map.get(self.gating_module3)
            if k3 is not None:
                gated_out3, logits3, _, student_mask3, _ = self.gating_module3(out, k_tensor=k3)
                block_masks['dynamic_mask_soft_3'] = torch.tanh(logits3)
                block_masks['dynamic_mask_hard_3'] = student_mask3
                out = gated_out3

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out, block_masks


class ResNetForPruning(nn.Module):
    def __init__(self, block, layers, num_classes=10, sparsity=None, channel_config=None, bottleneck_strategy='A'):
        super(ResNetForPruning, self).__init__()
        self.bottleneck_strategy = bottleneck_strategy

        if channel_config:
            # print("[ResNet Builder] Building model from provided channel_config.")
            self.pruned_out_channels, self.pruned_mid_channels = channel_config
        elif sparsity is not None:
            print("[ResNet Builder] Building model from sparsity rates (First Iteration).")
            self.pruned_out_channels, self.pruned_mid_channels, self.original_out_channels = adapt_channel_for_basic_resnet(layers, sparsity)
        else:
            print("[ResNet Builder] No sparsity or channel_config. Building base Basic-ResNet.")
            self.pruned_out_channels, self.pruned_mid_channels, self.original_out_channels = adapt_channel_for_basic_resnet(layers, [0.0] * (2 * sum(layers)))

        # --- NEW: Define a map from original channels to feature map size for CIFAR-10 ---
        # Input 32x32 -> conv1(s1) -> 32x32 -> layer1(s1) -> 32x32 -> layer2(s2) -> 16x16 -> layer3(s2) -> 8x8 -> layer4(s2) -> 4x4
        # The keys are the ORIGINAL channel numbers for each stage.
        self.c2wh = {64: 32, 128: 16, 256: 8, 512: 4}
        # Ensure original_out_channels exists as base planes per block (pre-expansion)
        self.original_out_channels = [64] + [64]*layers[0] + [128]*layers[1] + [256]*layers[2] + [512]*layers[3]
        # Per-stage base planes and out channels (handles expansion=1 or 4)
        self.stage_base_planes = [64, 128, 256, 512]
        # self.stage_out_channels = [p * type(block).expansion if isinstance(block, type) else p * block.expansion for p in self.stage_base_planes]
        self.stage_out_channels = [p * block.expansion for p in self.stage_base_planes]
        # --- END NEW ---

        # --- NEW: Instantiate Auxiliary Heads and Budget Generators ---
        # one aux head per stage (after stage1/2/3 outputs)
        self.num_classes = num_classes
        self.aux_head1 = AuxiliaryHead(self.stage_out_channels[0], num_classes)
        self.aux_head2 = AuxiliaryHead(self.stage_out_channels[1], num_classes)
        self.aux_head3 = AuxiliaryHead(self.stage_out_channels[2], num_classes)

        self.budget_generator2 = BudgetFactorGenerator()
        self.budget_generator3 = BudgetFactorGenerator()
        self.budget_generator4 = BudgetFactorGenerator()

        # Collect them for easy access to parameters
        self.aux_heads = nn.ModuleList([self.aux_head1, self.aux_head2, self.aux_head3])
        self.budget_generators = nn.ModuleList([self.budget_generator2, self.budget_generator3, self.budget_generator4])

        self.channel_idx = 0
        self.mid_channel_idx = 0

        # Initial conv out channels: 64 for CIFAR-ResNet (independent of expansion)
        if getattr(block, 'expansion', 1) == 1:
            self.inplanes = self.pruned_out_channels[self.channel_idx]
        else:
            self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()

        self.channel_idx += 1

        self.layer1 = self._make_layer(block, layers[0], is_dynamic_layer=True)
        self.layer2 = self._make_layer(block, layers[1], stride=2, is_dynamic_layer=True)
        self.layer3 = self._make_layer(block, layers[2], stride=2, is_dynamic_layer=True)
        self.layer4 = self._make_layer(block, layers[3], stride=2, is_dynamic_layer=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Use current inplanes (output channels of last layer) to define FC input
        self.fc = nn.Linear(self.inplanes, num_classes)

        self.gating_modules = nn.ModuleList()
        for module in self.modules():
            if isinstance(module, Fca_SE_GatingModule):
                self.gating_modules.append(module)

        # --- MODIFICATION START: Collect all static gates for easy access ---
        self.static_gates = nn.ParameterList()
        for module in self.modules():
            if isinstance(module, BasicBlock):
                self.static_gates.append(module.static_gate1)
                self.static_gates.append(module.static_gate2)
            if isinstance(module, BottleneckBlock):
                if hasattr(module, 'static_gate1') and isinstance(module.static_gate1, torch.nn.Parameter):
                    self.static_gates.append(module.static_gate1)
                self.static_gates.append(module.static_gate3)


    def _make_layer(self, block, blocks, stride=1, is_dynamic_layer=False): # NEW: is_dynamic_layer flag
        layers = []

        # Determine DCT spatial size from stage base planes (pre-expansion)
        original_planes = self.original_out_channels[self.channel_idx]
        dct_size = self.c2wh.get(original_planes, None)

        if getattr(block, 'expansion', 1) == 1:
            # BasicBlock path (uses midplanes, two prunable points)
            planes = self.pruned_out_channels[self.channel_idx]
            midplanes = self.pruned_mid_channels[self.mid_channel_idx]
            layers.append(block(self.inplanes, planes, midplanes, stride,
                                 is_dynamic_block=is_dynamic_layer,
                                 dct_h=dct_size, dct_w=dct_size))
            self.inplanes = planes * block.expansion
            self.channel_idx += 1
            self.mid_channel_idx += 1
        else:
            # BottleneckBlock path
            # Here 'planes' means base planes (64/128/256/512)
            base_planes = original_planes
            post_size = dct_size
            # pre size: for first block with stride=2, use previous stage spatial; otherwise same as post
            if stride == 2 and base_planes != 64:
                prev_base = base_planes // 2
                pre_size = self.c2wh.get(prev_base, post_size)
            else:
                pre_size = post_size
            layers.append(block(self.inplanes, base_planes, stride,
                                 is_dynamic_block=is_dynamic_layer,
                                 dct_h=dct_size, dct_w=dct_size,
                                 strategy=getattr(self, 'bottleneck_strategy', 'A'),
                                 dct_h_pre=pre_size, dct_w_pre=pre_size,
                                 dct_h_post=post_size, dct_w_post=post_size))
            self.inplanes = base_planes * block.expansion
            self.channel_idx += 1
            # mid_channel_idx not used for bottleneck

        for _ in range(1, blocks):
            original_planes = self.original_out_channels[self.channel_idx]
            dct_size = self.c2wh.get(original_planes, None)
            if getattr(block, 'expansion', 1) == 1:
                planes = self.pruned_out_channels[self.channel_idx]
                midplanes = self.pruned_mid_channels[self.mid_channel_idx]
                layers.append(block(self.inplanes, planes, midplanes,
                                     is_dynamic_block=is_dynamic_layer,
                                     dct_h=dct_size, dct_w=dct_size))
                self.inplanes = planes * block.expansion
                self.channel_idx += 1
                self.mid_channel_idx += 1
            else:
                base_planes = original_planes
                post_size = dct_size
                pre_size = post_size  # subsequent blocks have stride=1, so pre==post
                layers.append(block(self.inplanes, base_planes,
                                     is_dynamic_block=is_dynamic_layer,
                                     dct_h=dct_size, dct_w=dct_size,
                                     strategy=getattr(self, 'bottleneck_strategy', 'A'),
                                     dct_h_pre=pre_size, dct_w_pre=pre_size,
                                     dct_h_post=post_size, dct_w_post=post_size))
                self.inplanes = base_planes * block.expansion
                self.channel_idx += 1

        return nn.Sequential(*layers)


    def forward(self, x, is_warmup=False):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        all_block_masks = [] # To collect masks from all blocks
        aux_logits_list = []
        budget_factors_list = []
        uncertainty_scores_list = []

        k_map1, k_map2, k_map3, k_map4 = None, None, None, None

        if not is_warmup:
            # Stage 1: Layer1 (Unconditional 80% dynamic budget)
            k_map1 = {}
            for block in self.layer1:
                if hasattr(block, 'gating_module1') and block.gating_module1 is not None:
                    fixed_k = int(round(block.gating_module1.num_channels * 0.8))
                    k_map1[block.gating_module1] = torch.full((batch_size,), fixed_k, device=x.device, dtype=torch.int)
                if hasattr(block, 'gating_module2') and block.gating_module2 is not None:
                    fixed_k = int(round(block.gating_module2.num_channels * 0.8))
                    k_map1[block.gating_module2] = torch.full((batch_size,), fixed_k, device=x.device, dtype=torch.int)
                if hasattr(block, 'gating_module3') and block.gating_module3 is not None:
                    fixed_k = int(round(block.gating_module3.num_channels * 0.8))
                    k_map1[block.gating_module3] = torch.full((batch_size,), fixed_k, device=x.device, dtype=torch.int)

        x1 = x
        for block in self.layer1:
            x1, block_masks = block(x1, k_map=k_map1, is_warmup=is_warmup)
            all_block_masks.append(block_masks)

        # AuxHead 1 and Uncertainty Calculation
        aux_logits1 = self.aux_head1(x1.detach())
        aux_logits_list.append(aux_logits1)
        uncertainty_score1 = calculate_entropy(aux_logits1)
        uncertainty_scores_list.append(uncertainty_score1)

        if not is_warmup:
            # Stage 2: Layer2 (Conditional Budget)
            max_entropy = math.log(self.num_classes)
            u1 = torch.clamp(uncertainty_score1 / max_entropy, 0.0, 1.0)
            budget_factor2 = torch.clamp(self.budget_generator2(u1), max=0.99)
            budget_factors_list.append(budget_factor2)
            k_map2 = {}
            for block in self.layer2:
                if hasattr(block, 'gating_module1') and block.gating_module1 is not None:
                    k_map2[block.gating_module1] = torch.round(budget_factor2 * block.gating_module1.num_channels).int()
                if hasattr(block, 'gating_module2') and block.gating_module2 is not None:
                    k_map2[block.gating_module2] = torch.round(budget_factor2 * block.gating_module2.num_channels).int()
                if hasattr(block, 'gating_module3') and block.gating_module3 is not None:
                    k_map2[block.gating_module3] = torch.round(budget_factor2 * block.gating_module3.num_channels).int()

        x2 = x1
        for block in self.layer2:
            x2, block_masks = block(x2, k_map=k_map2, is_warmup=is_warmup)
            all_block_masks.append(block_masks)

        # AuxHead 2 and Uncertainty Calculation
        aux_logits2 = self.aux_head2(x2.detach())
        aux_logits_list.append(aux_logits2)
        uncertainty_score2 = calculate_entropy(aux_logits2)
        uncertainty_scores_list.append(uncertainty_score2)

        if not is_warmup:
            # Stage 3: Layer3 (Conditional Budget)
            max_entropy = math.log(self.num_classes)
            u2 = torch.clamp(uncertainty_score2 / max_entropy, 0.0, 1.0)
            budget_factor3 = torch.clamp(self.budget_generator3(u2), max=0.99)
            budget_factors_list.append(budget_factor3)
            k_map3 = {}
            for block in self.layer3:
                if hasattr(block, 'gating_module1') and block.gating_module1 is not None:
                    k_map3[block.gating_module1] = torch.round(budget_factor3 * block.gating_module1.num_channels).int()
                if hasattr(block, 'gating_module2') and block.gating_module2 is not None:
                    k_map3[block.gating_module2] = torch.round(budget_factor3 * block.gating_module2.num_channels).int()
                if hasattr(block, 'gating_module3') and block.gating_module3 is not None:
                    k_map3[block.gating_module3] = torch.round(budget_factor3 * block.gating_module3.num_channels).int()

        x3 = x2
        for block in self.layer3:
            x3, block_masks = block(x3, k_map=k_map3, is_warmup=is_warmup)
            all_block_masks.append(block_masks)

        # AuxHead 3 and Uncertainty Calculation
        aux_logits3 = self.aux_head3(x3.detach())
        aux_logits_list.append(aux_logits3)
        uncertainty_score3 = calculate_entropy(aux_logits3)
        uncertainty_scores_list.append(uncertainty_score3)

        if not is_warmup:
            # Stage 4: Layer4 (Conditional Budget)
            max_entropy = math.log(self.num_classes)
            u3 = torch.clamp(uncertainty_score3 / max_entropy, 0.0, 1.0)
            budget_factor4 = torch.clamp(self.budget_generator4(u3), max=0.99)
            budget_factors_list.append(budget_factor4)
            k_map4 = {}
            for block in self.layer4:
                if hasattr(block, 'gating_module1') and block.gating_module1 is not None:
                    k_map4[block.gating_module1] = torch.round(budget_factor4 * block.gating_module1.num_channels).int()
                if hasattr(block, 'gating_module2') and block.gating_module2 is not None:
                    k_map4[block.gating_module2] = torch.round(budget_factor4 * block.gating_module2.num_channels).int()
                if hasattr(block, 'gating_module3') and block.gating_module3 is not None:
                    k_map4[block.gating_module3] = torch.round(budget_factor4 * block.gating_module3.num_channels).int()

        x4 = x3
        for block in self.layer4:
            x4, block_masks = block(x4, k_map=k_map4, is_warmup=is_warmup)
            all_block_masks.append(block_masks)

        # Final Classifier
        x_final = self.avgpool(x4)
        x_final = x_final.view(x_final.size(0), -1)
        main_logits = self.fc(x_final)

        # Return all necessary information for calculating losses
        return main_logits, aux_logits_list, budget_factors_list, uncertainty_scores_list, all_block_masks

def resnet18(sparsity=None, channel_config=None, **kwargs):
    return ResNetForPruning(BasicBlock, [2, 2, 2, 2], sparsity=sparsity, channel_config=channel_config, **kwargs)


# New: ResNet-34 builder (CIFAR-10, BasicBlock)
def resnet34(sparsity=None, channel_config=None, **kwargs):
    return ResNetForPruning(BasicBlock, [3, 4, 6, 3], sparsity=sparsity, channel_config=channel_config, **kwargs)


# New: ResNet-50 builder (CIFAR-10, Bottleneck; Strategy A)
def resnet50(sparsity=None, channel_config=None, **kwargs):
    return ResNetForPruning(BottleneckBlock, [3, 4, 6, 3], sparsity=sparsity, channel_config=channel_config, **kwargs)

def get_scalable_flops_thop(model: nn.Module):
    """
    Calculates the SCALABLE FLOPs of the main path (convs, bns, etc.) for each
    part of the model. This EXCLUDES the overhead of dynamic modules like gating_modules.


    This function analyzes the original, unpruned model structure.
    """
    # Create a temporary model clone to modify it for analysis without affecting the original.
    model_clone = copy.deepcopy(model)
    device = next(model_clone.parameters()).device

    # For profiling the main path, we don't need to modify the forward methods.
    # We just need to ensure the gating modules are None so they are not called.
    for module in model_clone.modules():
        if isinstance(module, BasicBlock):
            module.gating_module1 = None
            module.gating_module2 = None
        if isinstance(module, BottleneckBlock):
            if hasattr(module, 'gating_module1'):
                module.gating_module1 = None
            module.gating_module3 = None

    # 1. Profile conv1 + bn1
    input_conv1 = torch.randn(1, 3, 32, 32).to(device)
    flops_conv1_bn1, _ = profile(nn.Sequential(model_clone.conv1, model_clone.bn1), inputs=(input_conv1,), verbose=False)

    # 2. Profile fc layer
    last_layer_orig_channels = model.fc.in_features
    input_fc = torch.randn(1, last_layer_orig_channels).to(device)
    flops_fc, _ = profile(model_clone.fc, inputs=(input_fc,), verbose=False)

    # 3. Profile each of layers 1-4 using a wrapper
    # Build input shapes dynamically to support Basic (18/34) and Bottleneck (50)
    layer_in_channels = {
        'layer1': model_clone.conv1.out_channels,
        'layer2': model_clone.stage_out_channels[0],
        'layer3': model_clone.stage_out_channels[1],
        'layer4': model_clone.stage_out_channels[2],
    }
    # IMPORTANT: use pre-downsample spatial sizes as inputs for each layer,
    # so that stride-2 in the first block of the layer is properly accounted for
    layer_spatial_pre = {'layer1': (32, 32), 'layer2': (32, 32), 'layer3': (16, 16), 'layer4': (8, 8)}
    input_shapes = {
        name: (1, layer_in_channels[name], *layer_spatial_pre[name])
        for name in ['layer1', 'layer2', 'layer3', 'layer4']
    }
    layer_flops_dict = {}
    blocks_flops = {}
    blocks_input_shapes = {}

    # Define the wrapper class inside the function
    class ProfilingWrapper(nn.Module):
        def __init__(self, layer_to_wrap):
            super().__init__()
            self.layer_to_wrap = layer_to_wrap

        def forward(self, x):
            # Manually loop through the blocks in the layer
            current_tensor = x
            for block in self.layer_to_wrap:
                output_tuple = block(current_tensor, is_warmup=True)
                current_tensor = output_tuple[0]
            return current_tensor

    class SingleBlockWrapper(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block
        def forward(self, x):
            return self.block(x, is_warmup=True)[0]

    for name, shape in input_shapes.items():
        sub_module = getattr(model_clone, name)
        dummy_input = torch.randn(shape).to(device)

        # Wrap the nn.Sequential layer for profiling
        wrapped_layer = ProfilingWrapper(sub_module)
        flops, _ = profile(wrapped_layer, inputs=(dummy_input,), verbose=False)
        layer_flops_dict[name] = flops

        # Per-block profiling for debug
        per_block_flops = []
        per_block_input_shapes = []
        current = dummy_input
        for block in sub_module:
            per_block_input_shapes.append((int(current.shape[1]), int(current.shape[2]), int(current.shape[3])))
            bw = SingleBlockWrapper(block).to(device)
            f_b, _ = profile(bw, inputs=(current,), verbose=False)
            per_block_flops.append(f_b)
            with torch.no_grad():
                current = bw(current)
        blocks_flops[name] = per_block_flops
        blocks_input_shapes[name] = per_block_input_shapes

    return {
        'conv1_bn1': flops_conv1_bn1,
        'fc': flops_fc,
        'layers': layer_flops_dict,
        'input_shapes': {k: (shape[1], shape[2], shape[3]) for k, shape in input_shapes.items()},
        'blocks': blocks_flops,
        'block_input_shapes': blocks_input_shapes
    }

def get_authoritative_scalable_core_flops(model: nn.Module):
    """
    End-to-end authoritative main-path FLOPs (no overhead):
    - Disables gating modules
    - Skips aux heads and budget generators
    - Forces is_warmup=True for all blocks
    This should match Global Profile - Overhead if our decomposition is complete.
    """
    device = next(model.parameters()).device
    model_clone = copy.deepcopy(model)

    # Disable gating modules
    for module in model_clone.modules():
        if isinstance(module, BasicBlock):
            module.gating_module1 = None
            module.gating_module2 = None
        if isinstance(module, BottleneckBlock):
            if hasattr(module, 'gating_module1'):
                module.gating_module1 = None
            module.gating_module3 = None

    class CoreOnly(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.conv1 = base.conv1
            self.bn1 = base.bn1
            self.relu = base.relu
            self.maxpool = base.maxpool
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            self.avgpool = base.avgpool
            self.fc = base.fc
        def forward(self, x):
            x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
            for blk in self.layer1: x = blk(x, is_warmup=True)[0]
            for blk in self.layer2: x = blk(x, is_warmup=True)[0]
            for blk in self.layer3: x = blk(x, is_warmup=True)[0]
            for blk in self.layer4: x = blk(x, is_warmup=True)[0]
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    core = CoreOnly(model_clone).to(device)
    dummy = torch.randn(1, 3, 32, 32).to(device)
    flops_core, _ = profile(core, inputs=(dummy,), verbose=False)
    return flops_core

def get_overhead_flops_thop(model: nn.Module):
    """
    Calculates the fixed OVERHEAD FLOPs of all dynamic decision-making modules.
    """
    overhead_flops = {
        'gating_modules': 0,
        'static_mask_apply': 0,
        'aux_heads': 0,
        'budget_generators': 0
    }
    device = next(model.parameters()).device

    # Common spatial sizes per stage (CIFAR-10)
    stage_hw = {
        'layer1': (32, 32), 'layer2': (16, 16), 'layer3': (8, 8), 'layer4': (4, 4)
    }
    pre_stage_hw = {
        'layer1': (32, 32), 'layer2': (32, 32), 'layer3': (16, 16), 'layer4': (8, 8)
    }

    # 0. Static mask application cost (elementwise multiply outside gating modules)
    # Note: count once per block per existing static gate
    for layer_name, (H, W) in stage_hw.items():
        layer = getattr(model, layer_name)
        for block in layer:
            # BasicBlock/Bottleneck static gates (only count real Parameters)
            if hasattr(block, 'static_gate1') and isinstance(block.static_gate1, torch.nn.Parameter):
                C = block.static_gate1.numel()
                overhead_flops['static_mask_apply'] += C * H * W
            if hasattr(block, 'static_gate2') and isinstance(block.static_gate2, torch.nn.Parameter):
                C = block.static_gate2.numel()
                overhead_flops['static_mask_apply'] += C * H * W
            if hasattr(block, 'static_gate3') and isinstance(block.static_gate3, torch.nn.Parameter):
                C = block.static_gate3.numel()
                overhead_flops['static_mask_apply'] += C * H * W

    # 1. Gating Modules (their internal compute already includes dynamic mask application)
    for layer_name, (H, W) in stage_hw.items():
        layer = getattr(model, layer_name)
        for block in layer:
            if hasattr(block, 'gating_module1') and block.gating_module1 is not None:
                C = block.gating_module1.num_channels
                # For Bottleneck first block with stride=2, conv1 gating sees pre-stage spatial size
                if isinstance(block, BottleneckBlock):
                    try:
                        s = block.conv2.stride[0] if isinstance(block.conv2.stride, tuple) else block.conv2.stride
                    except Exception:
                        s = 1
                    HH, WW = pre_stage_hw[layer_name] if s == 2 else stage_hw[layer_name]
                else:
                    HH, WW = H, W
                dummy_input1 = torch.randn(1, C, HH, WW).to(device)
                dummy_k_tensor = torch.tensor([1]).to(device)
                flops, _ = profile(block.gating_module1, inputs=(dummy_input1, dummy_k_tensor), verbose=False)
                overhead_flops['gating_modules'] += flops
            if hasattr(block, 'gating_module2') and block.gating_module2 is not None:
                C = block.gating_module2.num_channels
                dummy_input2 = torch.randn(1, C, H, W).to(device)
                dummy_k_tensor = torch.tensor([1]).to(device)
                flops, _ = profile(block.gating_module2, inputs=(dummy_input2, dummy_k_tensor), verbose=False)
                overhead_flops['gating_modules'] += flops
            if hasattr(block, 'gating_module3') and block.gating_module3 is not None:
                C = block.gating_module3.num_channels
                dummy_input3 = torch.randn(1, C, H, W).to(device)
                dummy_k_tensor = torch.tensor([1]).to(device)
                flops, _ = profile(block.gating_module3, inputs=(dummy_input3, dummy_k_tensor), verbose=False)
                overhead_flops['gating_modules'] += flops

    # 2. Auxiliary Heads
    aux_specs = [
        ('aux_head1', model.stage_out_channels[0], 32, 32),
        ('aux_head2', model.stage_out_channels[1], 16, 16),
        ('aux_head3', model.stage_out_channels[2], 8, 8),
    ]
    for name, C, H, W in aux_specs:
        module = getattr(model, name)
        dummy_input = torch.randn(1, C, H, W).to(device)
        flops, _ = profile(module, inputs=(dummy_input,), verbose=False)
        overhead_flops['aux_heads'] += flops

    # 3. Budget Generators
    dummy_input_budget = torch.randn(1).to(device)
    for module in model.budget_generators:
        flops, _ = profile(module, inputs=(dummy_input_budget,), verbose=False)
        overhead_flops['budget_generators'] += flops

    return overhead_flops
