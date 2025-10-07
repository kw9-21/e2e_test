import os
import torch
import argparse
import numpy as np
import datetime
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from collections import OrderedDict
import copy
from thop import profile
import math
import resnet_cifar as resnet
from train_datadriven import finetune,  validate,calculate_hard_params
import utils
from tqdm import tqdm

def calculate_dynamic_budgets(model, rate_start, rate_end, logger):
    """
    Calculates the channel budget 'k' for each dynamic module and configures them.
    """
    logger.info("--- [Budget Calculator] Configuring channel budgets 'k' for dynamic modules... ---")
    dynamic_modules = []
    # Ensure a predictable order by iterating through the model structure
    if hasattr(model, 'layer4'):
        for block in model.layer4:
            if isinstance(block, resnet.BasicBlock) and block.is_dynamic:
                if block.gating_module1:
                    dynamic_modules.append(block.gating_module1)
                if block.gating_module2:
                    dynamic_modules.append(block.gating_module2)

    channel_budgets = {}
    if not dynamic_modules:
        logger.warning("[Budget Calculator] No dynamic gating modules found.")
        return channel_budgets

    p_ratios = np.linspace(rate_start, rate_end, len(dynamic_modules))

    for i, module in enumerate(dynamic_modules):
        total_channels = module.num_channels
        p_ratio = p_ratios[i]
        k = int(round(total_channels * p_ratio))

        # Configure the module for inference/validation
        module.k = k
        # Store the budget for teacher signal generation during training
        channel_budgets[module] = k
        logger.info(f"  - Module {i}: Total Ch={total_channels}, p_ratio={p_ratio:.3f} -> Budget k={module.k} SET.")

    logger.info("--- [Budget Calculator] Configuration complete. ---")
    return channel_budgets

def calculate_average_dynamic_flops(model, val_loader, logger, scalable_flops_dict, total_overhead_flops):
    """
    Calculates the average TOTAL FLOPs over the validation set using the unified
    budget factor approach, based on the final hard masks from the trained model.
    """
    logger.info("--- [Unified FLOPs Calculator] Starting final analysis over validation set...")

    model.eval()

    total_flops_accumulator = 0.0
    num_samples = 0

    # Pre-calculate fixed FLOPs part
    fixed_flops = scalable_flops_dict['conv1_bn1'] + scalable_flops_dict['fc'] + total_overhead_flops

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Final] Calculating Avg FLOPs")
        for images, _ in pbar:
            images = images.cuda()
            batch_size = images.shape[0]

            # Get all hard masks from the forward pass
            _, _, _, _, all_block_masks = model(images, is_warmup=False)

            # Calculate scalable FLOPs for this batch
            batch_scalable_flops = torch.tensor(0.0, device=images.device)

            # This logic must be kept in sync with the loss calculation
            hard_budgets_layers = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}

            # For DeiT FFN (MLP) budgets (hard), mirror heads logic
            hard_budgets_layers_ffn = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}

            # Robust mapping of sequential block masks -> layer name
            layer_block_counts = [len(model.layer1), len(model.layer2), len(model.layer3), len(model.layer4)]
            idx = 0
            for li, num_blocks in enumerate(layer_block_counts, start=1):
                layer_name = f'layer{li}'
                for _ in range(num_blocks):
                    block_masks = all_block_masks[idx]

                    # Collect available gate indices from keys
                    gate_indices = []
                    for gi in [1,2,3]:
                        if f'static_mask_hard_{gi}' in block_masks:
                            gate_indices.append(gi)
                    for gi in gate_indices:
                        s_key = f'static_mask_hard_{gi}'
                        d_key = f'dynamic_mask_hard_{gi}'
                        s_mask = block_masks[s_key].float()  # shape ~ [C]
                        # Build dynamic mask aligned to s_mask's shape
                        if layer_name == 'layer1':
                            d_mask = torch.ones_like(s_mask) * 0.8
                        else:
                            d_mask_raw = block_masks.get(d_key, None)
                            if d_mask_raw is None:
                                d_mask = torch.ones_like(s_mask)
                            else:
                                d_mask_raw = d_mask_raw.float()
                                # If dynamic mask is [B, C], average over batch to [C]
                                d_mask = d_mask_raw.mean(dim=0) if d_mask_raw.dim() > s_mask.dim() else d_mask_raw
                        # Reduce to scalar per gate to avoid channel-length mismatch across gates (e.g., gate1 vs gate3)
                        combined = (s_mask * d_mask).mean()
                        hard_budgets_layers[layer_name].append(combined)

                    # DeiT FFN budgets: use dedicated keys if present
                    if 'static_mask_hard_ffn' in block_masks:
                        s_ffn = block_masks['static_mask_hard_ffn'].float()
                        if layer_name == 'layer1':
                            d_ffn = torch.ones_like(s_ffn) * 0.8
                        else:
                            d_ffn_raw = block_masks.get('dynamic_mask_hard_ffn', None)
                            if d_ffn_raw is None:
                                d_ffn = torch.ones_like(s_ffn)
                            else:
                                d_ffn_raw = d_ffn_raw.float()
                                d_ffn = d_ffn_raw.mean(dim=0) if d_ffn_raw.dim() > s_ffn.dim() else d_ffn_raw
                        hard_budgets_layers_ffn[layer_name].append((s_ffn * d_ffn).mean())

                    idx += 1

            # Average budgets per layer and calculate FLOPs
            for i in range(1, 5):
                layer_name = f'layer{i}'
                if hard_budgets_layers[layer_name]:
                    # Average over blocks in the layer, then scale FLOPs for the full Attention module
                    avg_hard_budget = torch.stack(hard_budgets_layers[layer_name]).mean()
                    batch_scalable_flops += avg_hard_budget * scalable_flops_dict['layers'][layer_name]
                # If MLP scalable present, add FFN contribution similarly
                if 'layers_mlp' in scalable_flops_dict and hard_budgets_layers_ffn[layer_name]:
                    avg_hard_budget_ffn = torch.stack(hard_budgets_layers_ffn[layer_name]).mean()
                    batch_scalable_flops += avg_hard_budget_ffn * scalable_flops_dict['layers_mlp'][layer_name]

            total_flops_accumulator += (fixed_flops + batch_scalable_flops.item()) * batch_size
            num_samples += batch_size

    avg_total_flops = total_flops_accumulator / num_samples

    logger.info(f"--- [Unified FLOPs Calculator] Analysis complete.")
    logger.info(f"    - Average Total FLOPs: {avg_total_flops / 1e6:.3f} M")

    return avg_total_flops

def log_final_pruned_stats(trained_model, logger, val_loader, criterion,
                           total_flops_full_model, scalable_flops_dict, total_overhead_flops):
    """
    Analyzes the trained model using the new unified framework.
    """
    logger.info("\n" + "#"*60)
    logger.info("#####      ANALYZING FINAL PRUNED MODEL STATS (UNIFIED)      #####")
    logger.info("#"*60 + "\n")

    # --- Step 1: Baseline Stats (already calculated, just for logging) ---
    _, params_orig = profile(copy.deepcopy(trained_model), inputs=(torch.randn(1, 3, 32, 32).cuda(),), verbose=False)
    logger.info("--- [Step 1] Baseline Full Model Stats ---")
    logger.info(f"  - Original Model Parameters: {params_orig / 1e6:.3f} M")
    logger.info(f"  - Original Model Max FLOPs:  {total_flops_full_model / 1e6:.3f} M")

    # --- Step 2: Calculate Final Statically Pruned Parameters ---
    # We can use the existing helper function for this, it's accurate.
    params_pruned_static = calculate_hard_params(trained_model)
    logger.info("\n--- [Step 2] Final Static Structure Stats ---")
    logger.info(f"  - Statically Pruned Model Parameters: {params_pruned_static / 1e6:.3f} M")

    # --- Step 3: Calculate AVERAGE DYNAMIC FLOPs using the trained model ---
    logger.info("\n--- [Step 3] Calculating Average Dynamic FLOPs... ---")
    flops_pruned_dynamic_avg = calculate_average_dynamic_flops(
        trained_model, val_loader, logger, scalable_flops_dict, total_overhead_flops
    )

    # --- Step 4: Final summary ---
    logger.info("\n--- [Step 4] Pruning Analysis Summary ---")
    param_reduction_rate = (1 - params_pruned_static / params_orig) * 100 if params_orig > 0 else 0
    total_flops_reduction_rate = (1 - flops_pruned_dynamic_avg / total_flops_full_model) * 100 if total_flops_full_model > 0 else 0

    logger.info(f"  - Parameters: {params_orig / 1e6:.3f} M -> {params_pruned_static / 1e6:.3f} M (Reduced by {param_reduction_rate:.2f}%)")
    logger.info(f"  - FLOPs:      {total_flops_full_model / 1e9:.3f} G -> {flops_pruned_dynamic_avg / 1e9:.3f} G (Avg, Reduced by {total_flops_reduction_rate:.2f}%)")

    logger.info("\n--- [Step 5] Final Performance Validation ---")
    validate(epoch=-1, val_loader=val_loader, model=trained_model, criterion=criterion, logger=logger, is_warmup=False)

    logger.info("\n" + "#"*60)
    logger.info("#####      ANALYSIS COMPLETE      #####")
    logger.info("#"*60 + "\n")

def calculate_max_prunable_params(model):
    """
    Calculates the total number of parameters controlled by all static gates
    in the model, representing the theoretical maximum for the parameter loss proxy.
    """
    max_params_proxy = 0.0
    for module in model.modules():
        if isinstance(module, resnet.BasicBlock):
            # Cost for static_gate1
            cost1 = module.conv1.in_channels * (module.conv1.kernel_size[0] ** 2)
            num_channels1 = module.static_gate1.numel()
            max_params_proxy += cost1 * num_channels1

            # Cost for static_gate2
            cost2 = module.conv2.in_channels * (module.conv2.kernel_size[0] ** 2)
            num_channels2 = module.static_gate2.numel()
            max_params_proxy += cost2 * num_channels2
    return max_params_proxy

def main():
    parser = argparse.ArgumentParser(description='End-to-End Pruning Trainer V3.0')

    # Basic parameters
    parser.add_argument('--lr', type=float, default=0.05, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--save-dir', type=str, default='./result_e2e_datadriven', help='directory to save results')
    parser.add_argument('--dataset-path', type=str, default='/home/dancer/wxr_code/e2e_prune/data', help='path to CIFAR-10 dataset')
    parser.add_argument('--gpu-id', type=str, default='4', help='GPU id to use')

    # E2E Core parameters
    parser.add_argument('--alpha-final', type=float, default=0.2, help='final weight for mask loss')
    parser.add_argument('--w-params-final', type=float, default=1.0, help='Final weight for the target-driven parameter loss.')
    parser.add_argument('--w-params-anneal-epochs', type=int, default=75, help='Epochs to anneal w_params from 0 to its final value.')
    parser.add_argument('--target-static-params-ratio', type=float, default=0.5, help='Target ratio of prunable parameters to keep.')
    parser.add_argument('--warmup-epochs', type=int, default=50, help='epochs for model warmup before applying losses')
    parser.add_argument('--annealing-epochs-pct', type=float, default=0.7, help='percentage of epochs for annealing beta')
    parser.add_argument('--dynamic-rate-start', type=float, default=0.7, help='start activation rate for dynamic masks')
    parser.add_argument('--dynamic-rate-end', type=float, default=0.7, help='end activation rate for dynamic masks')
    parser.add_argument('--gate-threshold', type=float, default=0.1,help='threshold for physical pruning of static gates')
    parser.add_argument('--backbone', type=str, default='resnet18', help='backbone: resnet18 or resnet34 or resnet50')
    parser.add_argument('--bottleneck-strategy', type=str, default='A', choices=['A','B'], help='For ResNet-50: A=prune conv3 only; B=prune conv1 and conv3')

    # Dataset and dataloader related args
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'], help='dataset to use')
    parser.add_argument('--img-size', type=int, default=None, help='input size (default: 32 for CIFAR, 224 for ImageNet)')
    parser.add_argument('--num-workers', type=int, default=4, help='number of dataloader workers')
    parser.add_argument('--imagenet-train-dir', type=str, default='/home/dancer/PublicData/ImageNet/train', help='ImageNet train dir (ImageFolder)')
    parser.add_argument('--imagenet-val-dir', type=str, default='/home/dancer/PublicData/ImageNet/val', help='ImageNet val dir (ImageFolder)')


    parser.add_argument('--target-flops-ratio', type=float, default=0.5, help='Target ratio of the final model FLOPs to the layer2-4 model FLOPs')
    args = parser.parse_args()

    # --- 1. Setup ---
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logger = utils.get_logger(os.path.join(args.save_dir, f'logger_{now}.log'))
    logger.info(f"Args: {args}")

    # --- 2. Data ---
    dataset = args.dataset.lower()
    bb = args.backbone.lower()

    # Determine default img size if not specified
    if args.img_size is None:
        img_size = 224 if dataset == 'imagenet' else 32
    else:
        img_size = args.img_size

    if dataset in ['cifar10', 'cifar100']:
        # Choose stats: for DeiT, follow ImageNet stats per WDPruning; otherwise CIFAR stats
        if bb.startswith('deit'):
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            if dataset == 'cifar10':
                mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            else:
                mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        num_classes = 10 if dataset == 'cifar10' else 100
        DatasetCls = CIFAR10 if dataset == 'cifar10' else CIFAR100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_set = DatasetCls(root=args.dataset_path, train=True, download=True, transform=transform_train)
        val_set = DatasetCls(root=args.dataset_path, train=False, download=True, transform=transform_test)
    elif dataset == 'imagenet':
        num_classes = 1000
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.1, value='random'),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(int(img_size / 0.875)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_set = ImageFolder(args.imagenet_train_dir, transform=transform_train)
        val_set = ImageFolder(args.imagenet_val_dir, transform=transform_test)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # --- 3. Model & Budget Initialization ---
    logger.info(f"==> Building full-size {args.backbone} for training and target calculation...")

    if bb.startswith('deit'):
        from deit.builder import build_deit
        from deit import flops as deit_flops
        # Build original model for FLOPs and the training model separately (no weight sharing required here)
        original_model_for_flops = build_deit(bb, num_classes=num_classes, img_size=img_size)
        # --- Step A: Global authoritative FLOPs ---
        total_flops_full_model, _ = profile(copy.deepcopy(original_model_for_flops), inputs=(torch.randn(1, 3, img_size, img_size).cuda(),), verbose=False)
        # --- Step B: Decomposition via THOP on slices (max) ---
        scalable_flops_dict = deit_flops.get_scalable_flops_thop(original_model_for_flops, img_size=img_size)
        overhead_flops_dict = deit_flops.get_overhead_flops_thop(original_model_for_flops, img_size=img_size)
        # --- Extra: Authoritative core-only (no overhead) ---
        core_only_flops = deit_flops.get_authoritative_scalable_core_flops(original_model_for_flops, img_size=img_size)
        # Build training model
        model = build_deit(bb, num_classes=num_classes, img_size=img_size)
    else:
        # Select backbone module by dataset
        import importlib
        resnet_module = importlib.import_module('resnet_imagenet' if dataset == 'imagenet' else 'resnet_cifar')
        # Bind to global name used by helper functions
        global resnet
        resnet = resnet_module
        if bb == 'resnet34':
            model = resnet.resnet34(num_classes=num_classes, bottleneck_strategy=args.bottleneck_strategy).cuda()
        elif bb == 'resnet50':
            model = resnet.resnet50(num_classes=num_classes, bottleneck_strategy=args.bottleneck_strategy).cuda()
        else:
            model = resnet.resnet18(num_classes=num_classes, bottleneck_strategy=args.bottleneck_strategy).cuda()
        logger.info("==> Accurately calculating FLOPs breakdown from a clean, original model...")
        if bb == 'resnet34':
            original_model_for_flops = resnet.resnet34(num_classes=num_classes, bottleneck_strategy=args.bottleneck_strategy).cuda()
        elif bb == 'resnet50':
            original_model_for_flops = resnet.resnet50(num_classes=num_classes, bottleneck_strategy=args.bottleneck_strategy).cuda()
        else:
            original_model_for_flops = resnet.resnet18(num_classes=num_classes, bottleneck_strategy=args.bottleneck_strategy).cuda()
        # --- Step A: Calculate the single, authoritative total FLOPs using global profile ---
        total_flops_full_model, _ = profile(copy.deepcopy(original_model_for_flops), inputs=(torch.randn(1, 3, img_size, img_size).cuda(),), verbose=False)
        # --- Step B: Decompose the model into parts for loss calculation ---
        scalable_flops_dict = resnet.get_scalable_flops_thop(original_model_for_flops)
        overhead_flops_dict = resnet.get_overhead_flops_thop(original_model_for_flops)
        # Extra: Authoritative core-only (no overhead) by end-to-end wrapper
        core_only_flops = resnet.get_authoritative_scalable_core_flops(original_model_for_flops)

    # Align overhead to authoritative value by adding residual bucket if needed
    overhead_authoritative = total_flops_full_model - core_only_flops
    current_overhead_sum = sum(overhead_flops_dict.values())
    residual_misc = overhead_authoritative - current_overhead_sum
    if residual_misc > 0:
        overhead_flops_dict['misc_overhead'] = residual_misc
    total_overhead_flops = sum(overhead_flops_dict.values())

    # --- Step C: Sanity Check ---
    # Training-meaningful scalable capacity (Attention module + MLP linear)
    scalable_only_total = (
        sum(scalable_flops_dict['layers'].values())
        + (sum(scalable_flops_dict.get('layers_mlp', {}).values()) if 'layers_mlp' in scalable_flops_dict else 0)
    )
    fixed_total = scalable_flops_dict['conv1_bn1'] + scalable_flops_dict['fc']
    total_flops_by_sum = scalable_only_total + fixed_total + total_overhead_flops

    logger.info("==> FLOPs Sanity Check:")
    logger.info(f"  - Total FLOPs by Global Profile (Authoritative): {total_flops_full_model / 1e6:.3f} M")
    logger.info(f"  - Total FLOPs by Sum of Parts:                 {total_flops_by_sum / 1e6:.3f} M")
    discrepancy = abs(total_flops_by_sum - total_flops_full_model) / total_flops_full_model * 100
    logger.info(f"  - Discrepancy: {discrepancy:.2f}%")
    if discrepancy > 5:
        logger.warning("  - Sanity Check Warning: Discrepancy > 5%.")

    # --- Debug breakdown logging ---
    logger.info("==> [DEBUG] FLOPs Breakdown Details:")
    # 0) Authoritative core-only vs global-minus-overhead
    logger.info(f"  - [CoreOnly e2e]    {core_only_flops/1e6:.3f} M")
    logger.info(f"  - [Global-Overhead] {(total_flops_full_model - total_overhead_flops)/1e6:.3f} M")
    logger.info(f"  - [CoreOnly - (Fixed+ScalableOnly)] {(core_only_flops - (fixed_total + scalable_only_total))/1e6:.3f} M")
    # 1) Inputs per layer (C,H,W)
    if 'input_shapes' in scalable_flops_dict:
        for lname in ['layer1','layer2','layer3','layer4']:
            c,h,w = scalable_flops_dict['input_shapes'][lname]
            logger.info(f"  - Input to {lname}: (C,H,W)=({c},{h},{w})")
    # 2) Scalable parts
    # If MLP layers are present, log them as well
    if 'layers_mlp' in scalable_flops_dict:
        for lname, v in scalable_flops_dict['layers_mlp'].items():
            logger.info(f"  - {lname} [MLP] FLOPs: {v / 1e6:.3f} M")

    logger.info(f"  - conv1+bn1 FLOPs: {scalable_flops_dict['conv1_bn1'] / 1e6:.3f} M")
    for lname, v in scalable_flops_dict['layers'].items():
        logger.info(f"  - {lname} [AttnModule] FLOPs:     {v / 1e6:.3f} M")
    logger.info(f"  - fc FLOPs:         {scalable_flops_dict['fc'] / 1e6:.3f} M")
    logger.info(f"  - Scalable-only capacity total: {scalable_only_total / 1e6:.3f} M")
    logger.info(f"  - Fixed total (conv1+bn1 + fc): {fixed_total / 1e6:.3f} M")
    # 2.1) Per-block details (debug)
    if 'blocks' in scalable_flops_dict and 'block_input_shapes' in scalable_flops_dict:
        for lname in ['layer1','layer2','layer3','layer4']:
            if lname in scalable_flops_dict['blocks']:
                blks = scalable_flops_dict['blocks'][lname]
                shapes = scalable_flops_dict['block_input_shapes'][lname]
                logger.info(f"  - {lname} has {len(blks)} blocks; per-block FLOPs (M): {[round(b/1e6,3) for b in blks]}")
                logger.info(f"    per-block inputs (C,H,W): {shapes}")
                logger.info(f"    {lname} sum by blocks (M): {sum(blks)/1e6:.3f}")
    # 3) Overhead parts
    logger.info(f"  - Overhead:gating   {overhead_flops_dict.get('gating_modules',0) / 1e6:.3f} M")
    logger.info(f"  - Overhead:static   {overhead_flops_dict.get('static_mask_apply',0) / 1e6:.3f} M")
    logger.info(f"  - Overhead:aux      {overhead_flops_dict.get('aux_heads',0) / 1e6:.3f} M")
    logger.info(f"  - Overhead:budget   {overhead_flops_dict.get('budget_generators',0) / 1e6:.3f} M")
    if 'misc_overhead' in overhead_flops_dict:
        logger.info(f"  - Overhead:misc     {overhead_flops_dict['misc_overhead'] / 1e6:.3f} M")
    logger.info(f"  - Overhead total:   {total_overhead_flops / 1e6:.3f} M")
    logger.info(f"  - Sum (Scalable+Fixed+Overhead): {total_flops_by_sum / 1e6:.3f} M")

    # --- Step D: Calculate the final target for the TOTAL FLOPs ---
    target_total_flops = total_flops_full_model * args.target_flops_ratio
    fixed_flops = total_overhead_flops + scalable_flops_dict['conv1_bn1'] + scalable_flops_dict['fc']
    target_total_flops_train = target_total_flops - fixed_flops

    # Guard: if target scalable is non-positive (fixed costs exceed target), clamp to a small floor
    if target_total_flops_train <= 0:
        sum_layer_max = (
            sum(scalable_flops_dict['layers'].values())
            + (sum(scalable_flops_dict.get('layers_mlp', {}).values()) if 'layers_mlp' in scalable_flops_dict else 0)
        )
        floor = max(1e-6, 0.05 * sum_layer_max)  # 5% of scalable capacity as a safe floor
        logger.warning(
            "[FLOPs Target] Computed target scalable FLOPs is non-positive: "
            f"{target_total_flops_train / 1e6:.2f} M. Fixed FLOPs (overhead+conv1_bn1+fc)="
            f"{fixed_flops / 1e6:.2f} M exceeds total target {target_total_flops / 1e6:.2f} M. "
            "Clamping training target to a small positive floor. Consider using a larger --target-flops-ratio."
        )
        target_total_flops_train = floor

    logger.info("==> Final FLOPs Targets:")
    logger.info(f"  - Max Theoretical FLOPs: {total_flops_full_model / 1e6:.2f} M")
    logger.info(f"  - Target Total FLOPs ({args.target_flops_ratio*100}%): {target_total_flops / 1e6:.2f} M")
    logger.info(f"  - Target Scalable FLOPs for Training: {target_total_flops_train / 1e6:.2f} M")

    del original_model_for_flops

    _, total_params_full_model = profile(copy.deepcopy(model), inputs=(torch.randn(1, 3, img_size, img_size).cuda(),), verbose=False)
    final_target_params = total_params_full_model * args.target_static_params_ratio
    logger.info(f"Full model total parameters: {total_params_full_model / 1e6:.3f} M")
    logger.info(f"Target total parameters ({args.target_static_params_ratio*100}%): {final_target_params / 1e6:.3f} M")
    logger.info("==> Main model for training is ready.")
    logger.info("==> Building main model for training...")
    if bb.startswith('deit'):
        # model already built in DeiT branch above; keep as-is
        pass
    else:
        if bb == 'resnet34':
            model = resnet.resnet34(num_classes=num_classes, bottleneck_strategy=args.bottleneck_strategy).cuda()
        elif bb == 'resnet50':
            model = resnet.resnet50(num_classes=num_classes, bottleneck_strategy=args.bottleneck_strategy).cuda()
        else:
            model = resnet.resnet18(num_classes=num_classes, bottleneck_strategy=args.bottleneck_strategy).cuda()

    logger.info("==> Separating parameters into four groups for optimization...")

    main_params = []
    gating_params = []
    aux_params = []
    budget_params = []

    for name, param in model.named_parameters():
        if 'aux_head' in name:
            aux_params.append(param)
        elif 'budget_generator' in name:
            budget_params.append(param)
        elif 'gating_module' in name or 'static_gate' in name:
            gating_params.append(param)
        else:
            main_params.append(param)

    # Log the number of parameter tensors in each group for verification
    logger.info(f" - Main backbone params: {len(main_params)} tensors.")
    logger.info(f" - Gating (logit & static) params: {len(gating_params)} tensors.")
    logger.info(f" - Auxiliary head params: {len(aux_params)} tensors.")
    logger.info(f" - Budget generator params: {len(budget_params)} tensors.")

    # 1. Optimizer for the main network backbone (SGD is a good choice)
    main_optimizer = optim.SGD(main_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # 2. Optimizer for gating modules (logits) and static gates (Adam is suitable)
    gating_optimizer = optim.Adam(gating_params, lr=0.01)
    # 3. Optimizer for the new Auxiliary Heads (Adam is a good choice for fast convergence)
    aux_optimizer = optim.Adam(aux_params, lr=0.001, weight_decay=1e-4)
    # 4. Optimizer for the new Budget Factor Generators (Adam with a high LR for quick adaptation)
    budget_optimizer = optim.Adam(budget_params, lr=0.01)

    logger.info("==> Four optimizers created successfully.")
    lr_warmup_epochs = 5
    warmup_start_lr = 0.004

    total_steps = len(train_loader) * args.epochs
    warmup_steps = lr_warmup_epochs * len(train_loader)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup: from warmup_start_lr to the base learning rate (args.lr)
            progress = float(current_step) / float(max(1, warmup_steps))
            return (warmup_start_lr + (args.lr - warmup_start_lr) * progress) / args.lr
        else:
            # Cosine annealing after warmup
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(main_optimizer, lr_lambda=lr_lambda)
    logger.info(f"==> LR scheduler set: {lr_warmup_epochs}-epoch linear warmup + Cosine Annealing.")

    # --- 5. Execute End-to-End Training ---
    w_aux = 0.1
    w_flops = 2.0
    trained_model = finetune(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        main_optimizer=main_optimizer,
        gating_optimizer=gating_optimizer,
        aux_optimizer=aux_optimizer,
        budget_optimizer=budget_optimizer,
        scheduler=scheduler,
        logger=logger,
        result_dir=args.save_dir,
        w_aux=w_aux,
        w_flops=w_flops,
        w_ranking=0.2,
        ranking_margin=0.05,
        w_params_final=args.w_params_final,
        w_params_anneal_epochs=args.w_params_anneal_epochs,
        final_target_params=final_target_params,
        warmup_epochs=args.warmup_epochs,
        target_total_flops=target_total_flops,
        target_total_flops_train=target_total_flops_train,
        scalable_flops_dict=scalable_flops_dict,
        total_overhead_flops=total_overhead_flops
    )

    criterion = nn.CrossEntropyLoss().cuda()
    log_final_pruned_stats(
        trained_model=trained_model,
        logger=logger,
        val_loader=val_loader,
        criterion=criterion,
        total_flops_full_model=total_flops_full_model,
        scalable_flops_dict=scalable_flops_dict,
        total_overhead_flops=total_overhead_flops
    )

    logger.info("End-to-end pruning process finished successfully.")

if __name__ == '__main__':
    main()