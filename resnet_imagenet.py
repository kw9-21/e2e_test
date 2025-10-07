import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from thop import profile
from fca_layer import get_freq_indices, MultiSpectralDCTLayer

# ---- Shared modules (compact versions) ----
class AuxiliaryHead(nn.Module):
    def __init__(self, in_channels, num_classes=1000, mid_channels_ratio=0.25):
        super().__init__()
        mid = max(16, int(in_channels * mid_channels_ratio))
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, num_classes, kernel_size=1, bias=True)
        )
    def forward(self, x):
        vote = self.classifier(x)
        return torch.mean(vote, dim=[2, 3])

class BudgetFactorGenerator(nn.Module):
    def __init__(self, min_budget=0.3, input_dim=1, hidden_dim=16):
        super().__init__()
        self.min_budget = min_budget
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, s):
        s = s.unsqueeze(-1)
        t = self.net(s)
        base = (torch.tanh(t) + 1.0) / 2.0
        return (self.min_budget + (1.0 - self.min_budget) * base).squeeze(-1)

def calculate_entropy(logits):
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-9)
    return -torch.sum(probs * log_probs, dim=1)

class STEModule(nn.Module):
    def __init__(self, threshold=0.0):
        super().__init__(); self.threshold = threshold
    def forward(self, continuous_proxy, binary_mask):
        return continuous_proxy + (binary_mask - continuous_proxy).detach()

class Fca_SE_GatingModule(nn.Module):
    def __init__(self, num_channels, dct_h, dct_w, reduction=4, freq_sel_method='top16'):
        super().__init__()
        self.num_channels = num_channels
        self.k = num_channels
        mx, my = get_freq_indices(freq_sel_method)
        mx = [int(x * (dct_h / 7)) for x in mx]
        my = [int(y * (dct_w / 7)) for y in my]
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mx, my, num_channels)
        self.excitation = nn.Sequential(
            nn.Linear(num_channels, max(1, num_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, num_channels // reduction), num_channels, bias=False)
        )
        nn.init.constant_(self.excitation[2].weight, 0.1)
        self.ste_activation = STEModule()
    def forward(self, x, k_tensor):
        b, c, _, _ = x.shape
        s = self.dct_layer(x)
        raw = self.excitation(s)
        bounded = torch.tanh(raw)
        idx = torch.argsort(raw, dim=1, descending=True)
        cols = torch.arange(c, device=raw.device).unsqueeze(0)
        cmp = k_tensor.unsqueeze(1) > cols
        mask = torch.zeros_like(raw, dtype=torch.float)
        mask.scatter_(1, idx, cmp.float())
        gate = self.ste_activation(bounded, mask)
        out = x * gate.unsqueeze(-1).unsqueeze(-1)
        return out, bounded, raw, mask, s

# ---- Utils ----
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# ---- Blocks ----
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, midplanes, stride=1, is_dynamic_block=False, dct_h=None, dct_w=None):
        super().__init__()
        self.is_dynamic = is_dynamic_block
        self.conv1 = conv3x3(inplanes, midplanes, stride)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.static_gate1 = nn.Parameter(torch.ones(midplanes))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(midplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.static_gate2 = nn.Parameter(torch.ones(planes))
        if self.is_dynamic:
            assert dct_h is not None and dct_w is not None
            self.gating_module1 = Fca_SE_GatingModule(midplanes, dct_h, dct_w)
            self.gating_module2 = Fca_SE_GatingModule(planes, dct_h, dct_w)
        else:
            self.gating_module1 = None; self.gating_module2 = None
        self.downsample = None
        out_c = planes
        if stride != 1 or inplanes != out_c:
            self.downsample = nn.Sequential(conv1x1(inplanes, out_c, stride), nn.BatchNorm2d(out_c))
    def forward(self, x, k_map=None, is_warmup=False):
        identity = x; masks = {}
        out = self.conv1(x); out = self.bn1(out)
        p1 = torch.sigmoid(self.static_gate1); masks['static_mask_soft_1']=p1
        if not is_warmup:
            h1 = (p1>0.5).float(); masks['static_mask_hard_1']=h1
            out.mul_((p1 + (h1-p1).detach()).view(1,-1,1,1))
        out = self.relu(out)
        if self.is_dynamic and self.gating_module1 is not None and k_map is not None:
            k1 = k_map.get(self.gating_module1)
            if k1 is not None:
                out, lg1, _, m1, _ = self.gating_module1(out, k_tensor=k1)
                masks['dynamic_mask_soft_1']=torch.tanh(lg1); masks['dynamic_mask_hard_1']=m1
        out = self.conv2(out); out = self.bn2(out)
        p2 = torch.sigmoid(self.static_gate2); masks['static_mask_soft_2']=p2
        if not is_warmup:
            h2 = (p2>0.5).float(); masks['static_mask_hard_2']=h2
            out.mul_((p2 + (h2-p2).detach()).view(1,-1,1,1))
        if self.is_dynamic and self.gating_module2 is not None and k_map is not None:
            k2 = k_map.get(self.gating_module2)
            if k2 is not None:
                out, lg2, _, m2, _ = self.gating_module2(out, k_tensor=k2)
                masks['dynamic_mask_soft_2']=torch.tanh(lg2); masks['dynamic_mask_hard_2']=m2
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out, masks

class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, is_dynamic_block=False, dct_h=None, dct_w=None,
                 strategy='A', dct_h_pre=None, dct_w_pre=None, dct_h_post=None, dct_w_post=None):
        super().__init__()
        self.is_dynamic = is_dynamic_block; self.strategy = strategy
        self.conv1 = conv1x1(inplanes, planes, stride=1); self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride); self.bn2 = nn.BatchNorm2d(planes)
        out_c = planes * self.expansion
        self.conv3 = conv1x1(planes, out_c, stride=1); self.bn3 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.static_gate1 = nn.Parameter(torch.ones(planes)) if self.strategy=='B' else None
        self.static_gate3 = nn.Parameter(torch.ones(out_c))
        if self.is_dynamic:
            assert dct_h is not None and dct_w is not None
            pre_h = dct_h_pre if dct_h_pre is not None else dct_h
            pre_w = dct_w_pre if dct_w_pre is not None else dct_w
            post_h = dct_h_post if dct_h_post is not None else dct_h
            post_w = dct_w_post if dct_w_post is not None else dct_w
            self.gating_module1 = Fca_SE_GatingModule(planes, pre_h, pre_w) if self.strategy=='B' else None
            self.gating_module3 = Fca_SE_GatingModule(out_c, post_h, post_w)
        else:
            self.gating_module1 = None; self.gating_module3 = None
        self.downsample = None
        if stride != 1 or inplanes != out_c:
            self.downsample = nn.Sequential(conv1x1(inplanes, out_c, stride=stride), nn.BatchNorm2d(out_c))
    def forward(self, x, k_map=None, is_warmup=False):
        identity = x; masks = {}
        out = self.conv1(x); out = self.bn1(out)
        if self.static_gate1 is not None:
            p1 = torch.sigmoid(self.static_gate1); masks['static_mask_soft_1']=p1
            if not is_warmup:
                h1=(p1>0.5).float(); masks['static_mask_hard_1']=h1
                out.mul_((p1+(h1-p1).detach()).view(1,-1,1,1))
        out = self.relu(out)
        if self.is_dynamic and self.gating_module1 is not None and k_map is not None:
            k1 = k_map.get(self.gating_module1)
            if k1 is not None:
                out, lg1, _, m1, _ = self.gating_module1(out, k_tensor=k1)
                masks['dynamic_mask_soft_1']=torch.tanh(lg1); masks['dynamic_mask_hard_1']=m1
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        p3 = torch.sigmoid(self.static_gate3); masks['static_mask_soft_3']=p3
        if not is_warmup:
            h3=(p3>0.5).float(); masks['static_mask_hard_3']=h3
            out.mul_((p3+(h3-p3).detach()).view(1,-1,1,1))
        if self.is_dynamic and self.gating_module3 is not None and k_map is not None:
            k3 = k_map.get(self.gating_module3)
            if k3 is not None:
                out, lg3, _, m3, _ = self.gating_module3(out, k_tensor=k3)
                masks['dynamic_mask_soft_3']=torch.tanh(lg3); masks['dynamic_mask_hard_3']=m3
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out, masks

# ---- Model ----
class ResNetForPruning(nn.Module):
    def __init__(self, block, layers, num_classes=1000, bottleneck_strategy='A'):
        super().__init__()
        self.bottleneck_strategy = bottleneck_strategy
        self.num_classes = num_classes
        self.stage_base_planes = [64, 128, 256, 512]
        self.stage_out_channels = [p * block.expansion for p in self.stage_base_planes]
        self.c2wh = {64:56, 128:28, 256:14, 512:7}
        self.original_out_channels = [64] + [64]*layers[0] + [128]*layers[1] + [256]*layers[2] + [512]*layers[3]
        # Stem for ImageNet
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Layers
        self.channel_idx = 1; self.mid_channel_idx = 0
        self.layer1 = self._make_layer(block, layers[0], is_dynamic_layer=True, stride=1)
        self.layer2 = self._make_layer(block, layers[1], is_dynamic_layer=True, stride=2)
        self.layer3 = self._make_layer(block, layers[2], is_dynamic_layer=True, stride=2)
        self.layer4 = self._make_layer(block, layers[3], is_dynamic_layer=True, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplanes, num_classes)
        # Aux and budgets
        self.aux_head1 = AuxiliaryHead(self.stage_out_channels[0], num_classes)
        self.aux_head2 = AuxiliaryHead(self.stage_out_channels[1], num_classes)
        self.aux_head3 = AuxiliaryHead(self.stage_out_channels[2], num_classes)
        self.aux_heads = nn.ModuleList([self.aux_head1, self.aux_head2, self.aux_head3])
        self.budget_generator2 = BudgetFactorGenerator()
        self.budget_generator3 = BudgetFactorGenerator()
        self.budget_generator4 = BudgetFactorGenerator()
        self.budget_generators = nn.ModuleList([self.budget_generator2, self.budget_generator3, self.budget_generator4])
        # Collections
        self.gating_modules = nn.ModuleList([m for m in self.modules() if isinstance(m, Fca_SE_GatingModule)])
        self.static_gates = nn.ParameterList()
        for m in self.modules():
            if isinstance(m, BasicBlock):
                self.static_gates.append(m.static_gate1); self.static_gates.append(m.static_gate2)
            if isinstance(m, BottleneckBlock):
                if getattr(m, 'static_gate1', None) is not None:
                    self.static_gates.append(m.static_gate1)
                self.static_gates.append(m.static_gate3)
    def _make_layer(self, block, blocks, stride=1, is_dynamic_layer=False):
        layers = []
        orig_planes = self.original_out_channels[self.channel_idx]
        dct_size = self.c2wh.get(orig_planes, None)
        if block.expansion == 1:
            planes = orig_planes
            mid = planes
            layers.append(block(self.inplanes, planes, mid, stride,
                                is_dynamic_block=is_dynamic_layer, dct_h=dct_size, dct_w=dct_size))
            self.inplanes = planes * block.expansion
            self.channel_idx += 1; self.mid_channel_idx += 1
        else:
            base_planes = orig_planes; post_size = dct_size
            if stride == 2 and base_planes != 64:
                pre_size = self.c2wh.get(base_planes//2, post_size)
            else:
                pre_size = post_size
            layers.append(block(self.inplanes, base_planes, stride,
                                is_dynamic_block=is_dynamic_layer, dct_h=dct_size, dct_w=dct_size,
                                strategy=getattr(self, 'bottleneck_strategy','A'),
                                dct_h_pre=pre_size, dct_w_pre=pre_size,
                                dct_h_post=post_size, dct_w_post=post_size))
            self.inplanes = base_planes * block.expansion
            self.channel_idx += 1
        for _ in range(1, blocks):
            orig_planes = self.original_out_channels[self.channel_idx]
            dct_size = self.c2wh.get(orig_planes, None)
            if block.expansion == 1:
                planes = orig_planes; mid = planes
                layers.append(block(self.inplanes, planes, mid,
                                    is_dynamic_block=is_dynamic_layer, dct_h=dct_size, dct_w=dct_size))
                self.inplanes = planes * block.expansion
                self.channel_idx += 1; self.mid_channel_idx += 1
            else:
                base_planes = orig_planes; post_size = dct_size; pre_size = post_size
                layers.append(block(self.inplanes, base_planes,
                                    is_dynamic_block=is_dynamic_layer, dct_h=dct_size, dct_w=dct_size,
                                    strategy=getattr(self, 'bottleneck_strategy','A'),
                                    dct_h_pre=pre_size, dct_w_pre=pre_size,
                                    dct_h_post=post_size, dct_w_post=post_size))
                self.inplanes = base_planes * block.expansion
                self.channel_idx += 1
        return nn.Sequential(*layers)
    def forward(self, x, is_warmup=False):
        b = x.size(0)
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        all_masks = []; aux_logits_list=[]; budget_factors_list=[]; uncertainty_scores_list=[]
        k1=k2=k3=k4=None
        if not is_warmup:
            k1 = {}
            for blk in self.layer1:
                for gm in ['gating_module1','gating_module2','gating_module3']:
                    if hasattr(blk, gm) and getattr(blk, gm) is not None:
                        g = getattr(blk, gm); k1[g]=torch.full((b,), int(round(g.num_channels*0.8)), device=x.device, dtype=torch.int)
        x1 = x
        for blk in self.layer1:
            x1, m = blk(x1, k_map=k1, is_warmup=is_warmup); all_masks.append(m)
        aux1 = self.aux_head1(x1.detach()); aux_logits_list.append(aux1)
        u1 = calculate_entropy(aux1); uncertainty_scores_list.append(u1)
        if not is_warmup:
            u1n = torch.clamp(u1 / math.log(self.num_classes), 0.0, 1.0)
            bf2 = torch.clamp(self.budget_generator2(u1n), max=0.99); budget_factors_list.append(bf2)
            k2 = {}
            for blk in self.layer2:
                for gm in ['gating_module1','gating_module2','gating_module3']:
                    if hasattr(blk, gm) and getattr(blk, gm) is not None:
                        g = getattr(blk, gm); k2[g]=torch.round(bf2*g.num_channels).int()
        x2 = x1
        for blk in self.layer2:
            x2, m = blk(x2, k_map=k2, is_warmup=is_warmup); all_masks.append(m)
        aux2 = self.aux_head2(x2.detach()); aux_logits_list.append(aux2)
        u2 = calculate_entropy(aux2); uncertainty_scores_list.append(u2)
        if not is_warmup:
            u2n = torch.clamp(u2 / math.log(self.num_classes), 0.0, 1.0)
            bf3 = torch.clamp(self.budget_generator3(u2n), max=0.99); budget_factors_list.append(bf3)
            k3 = {}
            for blk in self.layer3:
                for gm in ['gating_module1','gating_module2','gating_module3']:
                    if hasattr(blk, gm) and getattr(blk, gm) is not None:
                        g = getattr(blk, gm); k3[g]=torch.round(bf3*g.num_channels).int()
        x3 = x2
        for blk in self.layer3:
            x3, m = blk(x3, k_map=k3, is_warmup=is_warmup); all_masks.append(m)
        aux3 = self.aux_head3(x3.detach()); aux_logits_list.append(aux3)
        u3 = calculate_entropy(aux3); uncertainty_scores_list.append(u3)
        if not is_warmup:
            u3n = torch.clamp(u3 / math.log(self.num_classes), 0.0, 1.0)
            bf4 = torch.clamp(self.budget_generator4(u3n), max=0.99); budget_factors_list.append(bf4)
            k4 = {}
            for blk in self.layer4:
                for gm in ['gating_module1','gating_module2','gating_module3']:
                    if hasattr(blk, gm) and getattr(blk, gm) is not None:
                        g = getattr(blk, gm); k4[g]=torch.round(bf4*g.num_channels).int()
        x4 = x3
        for blk in self.layer4:
            x4, m = blk(x4, k_map=k4, is_warmup=is_warmup); all_masks.append(m)
        xf = self.avgpool(x4); xf = xf.view(xf.size(0), -1); main_logits = self.fc(xf)
        return main_logits, aux_logits_list, budget_factors_list, uncertainty_scores_list, all_masks

# ---- Builders ----
def resnet18(**kwargs):
    return ResNetForPruning(BasicBlock, [2,2,2,2], **kwargs)

def resnet34(**kwargs):
    return ResNetForPruning(BasicBlock, [3,4,6,3], **kwargs)

def resnet50(**kwargs):
    return ResNetForPruning(BottleneckBlock, [3,4,6,3], **kwargs)

# ---- FLOPs APIs ----
def get_scalable_flops_thop(model: nn.Module):
    m = copy.deepcopy(model); device = next(m.parameters()).device
    for mod in m.modules():
        if isinstance(mod, BasicBlock):
            mod.gating_module1=None; mod.gating_module2=None
        if isinstance(mod, BottleneckBlock):
            if hasattr(mod,'gating_module1'): mod.gating_module1=None
            mod.gating_module3=None
    input_conv1 = torch.randn(1,3,224,224).to(device)
    flops_conv1_bn1,_ = profile(nn.Sequential(m.conv1, m.bn1), inputs=(input_conv1,), verbose=False)
    last_in = m.fc.in_features
    flops_fc,_ = profile(m.fc, inputs=(torch.randn(1,last_in).to(device),), verbose=False)
    layer_in_channels = {
        'layer1': m.conv1.out_channels,
        'layer2': m.stage_out_channels[0],
        'layer3': m.stage_out_channels[1],
        'layer4': m.stage_out_channels[2],
    }
    layer_spatial_pre = {'layer1':(56,56), 'layer2':(56,56), 'layer3':(28,28), 'layer4':(14,14)}
    input_shapes = {n:(1, layer_in_channels[n], *layer_spatial_pre[n]) for n in ['layer1','layer2','layer3','layer4']}
    class LayerWrap(nn.Module):
        def __init__(self, layer): super().__init__(); self.layer=layer
        def forward(self,x):
            cur=x
            for b in self.layer: cur=b(cur, is_warmup=True)[0]
            return cur
    class BlockWrap(nn.Module):
        def __init__(self, blk): super().__init__(); self.blk=blk
        def forward(self,x): return self.blk(x, is_warmup=True)[0]
    layer_flops={}; blocks_flops={}; blocks_inshapes={}
    for name,shape in input_shapes.items():
        sub = getattr(m,name); dummy=torch.randn(shape).to(device)
        fl,_=profile(LayerWrap(sub), inputs=(dummy,), verbose=False)
        layer_flops[name]=fl
        per=[]; sh=[]; cur=dummy
        for blk in sub:
            sh.append((int(cur.shape[1]), int(cur.shape[2]), int(cur.shape[3])))
            bw=BlockWrap(blk).to(device); fb,_=profile(bw, inputs=(cur,), verbose=False)
            per.append(fb); cur=bw(cur)
        blocks_flops[name]=per; blocks_inshapes[name]=sh
    return {
        'conv1_bn1': flops_conv1_bn1,
        'fc': flops_fc,
        'layers': layer_flops,
        'input_shapes': {k:(s[1],s[2],s[3]) for k,s in input_shapes.items()},
        'blocks': blocks_flops,
        'block_input_shapes': blocks_inshapes
    }

def get_authoritative_scalable_core_flops(model: nn.Module):
    device = next(model.parameters()).device
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, BasicBlock):
            mod.gating_module1=None; mod.gating_module2=None
        if isinstance(mod, BottleneckBlock):
            if hasattr(mod,'gating_module1'): mod.gating_module1=None
            mod.gating_module3=None
    class Core(nn.Module):
        def __init__(self, b):
            super().__init__()
            self.conv1=b.conv1; self.bn1=b.bn1; self.relu=b.relu; self.maxpool=b.maxpool
            self.layer1=b.layer1; self.layer2=b.layer2; self.layer3=b.layer3; self.layer4=b.layer4
            self.avgpool=b.avgpool; self.fc=b.fc
        def forward(self,x):
            x=self.conv1(x); x=self.bn1(x); x=self.relu(x); x=self.maxpool(x)
            for blk in self.layer1: x=blk(x, is_warmup=True)[0]
            for blk in self.layer2: x=blk(x, is_warmup=True)[0]
            for blk in self.layer3: x=blk(x, is_warmup=True)[0]
            for blk in self.layer4: x=blk(x, is_warmup=True)[0]
            x=self.avgpool(x); x=x.view(x.size(0),-1); x=self.fc(x); return x
    core=Core(m).to(device)
    dummy=torch.randn(1,3,224,224).to(device)
    fl,_=profile(core, inputs=(dummy,), verbose=False)
    return fl

def get_overhead_flops_thop(model: nn.Module):
    overhead={'gating_modules':0,'static_mask_apply':0,'aux_heads':0,'budget_generators':0}
    device = next(model.parameters()).device
    stage_hw={'layer1':(56,56),'layer2':(28,28),'layer3':(14,14),'layer4':(7,7)}
    pre_stage_hw={'layer1':(56,56),'layer2':(56,56),'layer3':(28,28),'layer4':(14,14)}
    # static mask apply
    for lname,(H,W) in stage_hw.items():
        layer=getattr(model,lname)
        for blk in layer:
            for gname in ['static_gate1','static_gate2','static_gate3']:
                if hasattr(blk,gname) and isinstance(getattr(blk,gname), torch.nn.Parameter):
                    C=getattr(blk,gname).numel(); overhead['static_mask_apply']+=C*H*W
    # gating modules
    for lname,(H,W) in stage_hw.items():
        layer=getattr(model,lname)
        for blk in layer:
            if hasattr(blk,'gating_module1') and blk.gating_module1 is not None:
                C=blk.gating_module1.num_channels
                s = blk.conv2.stride[0] if isinstance(blk, BottleneckBlock) else 1
                HH,WW = pre_stage_hw[lname] if s==2 else stage_hw[lname]
                dummy=torch.randn(1,C,HH,WW).to(device); k=torch.tensor([1]).to(device)
                fl,_=profile(blk.gating_module1, inputs=(dummy,k), verbose=False)
                overhead['gating_modules']+=fl
            if hasattr(blk,'gating_module2') and blk.gating_module2 is not None:
                C=blk.gating_module2.num_channels
                dummy=torch.randn(1,C,H, W).to(device); k=torch.tensor([1]).to(device)
                fl,_=profile(blk.gating_module2, inputs=(dummy,k), verbose=False)
                overhead['gating_modules']+=fl
            if hasattr(blk,'gating_module3') and blk.gating_module3 is not None:
                C=blk.gating_module3.num_channels
                dummy=torch.randn(1,C,H, W).to(device); k=torch.tensor([1]).to(device)
                fl,_=profile(blk.gating_module3, inputs=(dummy,k), verbose=False)
                overhead['gating_modules']+=fl
    # aux heads
    for name,C,H,W in [('aux_head1', model.stage_out_channels[0],56,56),
                       ('aux_head2', model.stage_out_channels[1],28,28),
                       ('aux_head3', model.stage_out_channels[2],14,14)]:
        mod=getattr(model,name); dummy=torch.randn(1,C,H,W).to(device)
        fl,_=profile(mod, inputs=(dummy,), verbose=False)
        overhead['aux_heads']+=fl
    # budget generators
    di=torch.randn(1).to(device)
    for mod in model.budget_generators:
        fl,_=profile(mod, inputs=(di,), verbose=False)
        overhead['budget_generators']+=fl
    return overhead

