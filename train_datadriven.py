import os
import numpy as np
import time, datetime
import argparse
import copy
from thop import profile
from collections import OrderedDict
import torch.optim as optim
import math
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.models as models
import resnet_cifar as resnet_cifar
from torchvision.datasets import CIFAR10
import utils
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt

def calculate_flops_loss(model, all_block_masks, budget_factors_list, scalable_flops_dict, target_scalable_flops):
    """
    【V4 - 最终清晰版】
    Calculates the FLOPs regularization loss with a clear, layer-wise logic 
    that correctly models mask interaction and provides clean gradient paths.

    - HARD loss (for value): Calculated layer-by-layer. For each layer, it computes an
      average combined hard budget by averaging the element-wise product of 
      static and dynamic hard masks over all convs in that layer. This accurately
      reflects channel overlap.
      
    - SOFT loss (for gradient): Also calculated layer-by-layer. For each layer, it
      computes a combined soft budget by multiplying the average soft static budget
      (from static_gates) and the average soft dynamic budget (from budget_generators).
      This provides independent, differentiable paths for both pruning mechanisms.
    """
    device = next(model.parameters()).device
    
    # --- 初始化累加器 ---
    actual_scalable_flops_soft = torch.tensor(0.0, device=device)
    actual_scalable_flops_hard = torch.tensor(0.0, device=device)

    # --- 准备每层的软动态预算 (来自 Budget Generators) ---
    # 这是可微的，用于 soft loss
    dynamic_budgets_soft = {
        'layer1': torch.tensor(0.8, device=device), # 固定值
        'layer2': torch.mean(budget_factors_list[0]) if len(budget_factors_list) > 0 else torch.tensor(1.0, device=device),
        'layer3': torch.mean(budget_factors_list[1]) if len(budget_factors_list) > 1 else torch.tensor(1.0, device=device),
        'layer4': torch.mean(budget_factors_list[2]) if len(budget_factors_list) > 2 else torch.tensor(1.0, device=device)
    }

    # --- 逐层计算 FLOPs ---
    block_idx = 0
    for i in range(1, 5): # 遍历 layer1 到 layer4
        layer_name = f'layer{i}'
        layer = getattr(model, layer_name)
        max_layer_flops = scalable_flops_dict['layers'][layer_name]

        # --- 为当前层收集所有相关的软/硬预算 ---
        
        # 用于 Soft Loss (梯度)
        static_budgets_soft_convs = [] # 收集该层每个conv的软静态预算

        # 用于 Hard Loss (值)
        combined_budgets_hard_convs = [] # 收集该层每个conv的综合硬预算

        # 遍历该层的所有 block
        for block in layer:
            # DeiT path: block has head-wise static gate; CNN path: gate indices 1..3
            if hasattr(block, 'static_gate_heads') and isinstance(block.static_gate_heads, torch.nn.Parameter):
                # Soft static budget (per block: mean over heads)
                static_budgets_soft_convs.append(torch.mean(torch.sigmoid(block.static_gate_heads.detach())))
                # Hard combined budget from masks collected in all_block_masks
                s_mask_hard = all_block_masks[block_idx]['static_mask_hard_1']  # (H)
                if layer_name == 'layer1':
                    d_mask_hard = 0.8  # fixed during warmup-like policy for layer1
                else:
                    # mean over batch -> (H)
                    d_mask_hard = torch.mean(all_block_masks[block_idx]['dynamic_mask_hard_1'], dim=0)
                combined_hard_ratio = torch.mean(s_mask_hard * d_mask_hard)
                combined_budgets_hard_convs.append(combined_hard_ratio)
            else:
                # CNN path: 收集该 block 的可用门控索引（Basic: 1,2; Bottleneck: 3）
                gate_indices = []
                for gi in [1,2,3]:
                    name = f'static_gate{gi}'
                    if hasattr(block, name) and isinstance(getattr(block, name), torch.nn.Parameter):
                        gate_indices.append(gi)
                for gi in gate_indices:
                    static_gate = getattr(block, f'static_gate{gi}')
                    if isinstance(static_gate, torch.nn.Parameter):
                        static_budgets_soft_convs.append(torch.mean(torch.sigmoid(static_gate.detach())))
                    s_key = f'static_mask_hard_{gi}'
                    d_key = f'dynamic_mask_hard_{gi}'
                    s_mask_hard = all_block_masks[block_idx][s_key]
                    if layer_name == 'layer1':
                        d_mask_hard = 0.8
                    else:
                        d_mask_hard = torch.mean(all_block_masks[block_idx].get(d_key, torch.ones_like(s_mask_hard)), dim=0)
                    combined_hard_ratio = torch.mean(s_mask_hard * d_mask_hard)
                    combined_budgets_hard_convs.append(combined_hard_ratio)

            block_idx += 1

        # --- 计算当前层的 Soft/Hard FLOPs ---
        # 如果是 DeiT 分支（存在 FFN 门控），则分别计算注意力与 MLP 两部分并相加
        if any(hasattr(b, 'static_gate_heads') for b in layer):
            # Heads (Attention linear)
            avg_static_budget_soft_heads = torch.mean(torch.stack(static_budgets_soft_convs))
            avg_dynamic_budget_soft_layer = dynamic_budgets_soft[layer_name]
            combined_soft_heads = avg_static_budget_soft_heads * avg_dynamic_budget_soft_layer
            actual_scalable_flops_soft += combined_soft_heads * max_layer_flops

            avg_combined_budget_hard_heads = torch.mean(torch.stack(combined_budgets_hard_convs))
            actual_scalable_flops_hard += avg_combined_budget_hard_heads * max_layer_flops

            # FFN (MLP linear)
            # 收集该层的 FFN 软/硬预算（在上面的遍历中同步收集）
            if 'layers_mlp' in scalable_flops_dict:
                # 软静态预算：每 block 的 static_gate_ffn 概率求均值
                static_budgets_soft_ffn = []
                combined_budgets_hard_ffn = []
                # 重新遍历该层的 blocks 来取出 FFN 的 mask（使用 all_block_masks 索引）
                # 注意：block_idx 当前指向下一 block，因此减去层内块数以对齐读取
                start_idx = block_idx - len(layer)
                for j, block in enumerate(layer):
                    if hasattr(block, 'static_gate_ffn') and isinstance(block.static_gate_ffn, torch.nn.Parameter):
                        static_budgets_soft_ffn.append(torch.mean(torch.sigmoid(block.static_gate_ffn.detach())))
                        s_mask_hard_ffn = all_block_masks[start_idx + j]['static_mask_hard_ffn']
                        if layer_name == 'layer1':
                            d_mask_hard_ffn = 0.8
                        else:
                            d_mask_hard_ffn = torch.mean(all_block_masks[start_idx + j]['dynamic_mask_hard_ffn'], dim=0)
                        combined_budgets_hard_ffn.append(torch.mean(s_mask_hard_ffn * d_mask_hard_ffn))
                if static_budgets_soft_ffn:
                    avg_static_budget_soft_ffn = torch.mean(torch.stack(static_budgets_soft_ffn))
                    combined_soft_ffn = avg_static_budget_soft_ffn * avg_dynamic_budget_soft_layer
                    max_layer_flops_mlp = scalable_flops_dict['layers_mlp'][layer_name]
                    actual_scalable_flops_soft += combined_soft_ffn * max_layer_flops_mlp

                if combined_budgets_hard_ffn:
                    avg_combined_budget_hard_ffn = torch.mean(torch.stack(combined_budgets_hard_ffn))
                    max_layer_flops_mlp = scalable_flops_dict['layers_mlp'][layer_name]
                    actual_scalable_flops_hard += avg_combined_budget_hard_ffn * max_layer_flops_mlp
        else:
            # CNN/其他路径：原有逻辑（单路）
            avg_static_budget_soft_layer = torch.mean(torch.stack(static_budgets_soft_convs))
            avg_dynamic_budget_soft_layer = dynamic_budgets_soft[layer_name]
            combined_soft_budget_layer = avg_static_budget_soft_layer * avg_dynamic_budget_soft_layer
            actual_scalable_flops_soft += combined_soft_budget_layer * max_layer_flops

            avg_combined_budget_hard_layer = torch.mean(torch.stack(combined_budgets_hard_convs))
            actual_scalable_flops_hard += avg_combined_budget_hard_layer * max_layer_flops

    # --- 计算最终的STE损失 ---
    loss_soft = (actual_scalable_flops_soft / target_scalable_flops - 1)**2
    loss_hard = (actual_scalable_flops_hard / target_scalable_flops - 1)**2
    
    loss_flops_ste = loss_soft + (loss_hard - loss_soft).detach()
    
    return loss_flops_ste, actual_scalable_flops_hard.item()

def log_static_gate_heatmap(model, epoch, save_dir, logger):
    """
    Visualizes the absolute values of all static gates in the model using heatmaps.
    Creates a single figure with one subplot per ResNet layer (layer1-4)
    and saves it to a file. All heatmaps share a common color bar for
    consistent value representation across epochs and layers.
    """
    logger.info(f"[Visualizer] Generating static gate heatmap for epoch {epoch}...")
    
    # --- 1. Create Figure and Subplots ---
    # Create 4 subplots, one for each layer of the ResNet
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle(f'Static Gate Absolute Values - Epoch {epoch}', fontsize=20)

    # Move model to CPU for data extraction
    # model.cpu()

    im = None # To store the mappable object for the color bar

    # --- 2. Iterate Through Layers to Collect Data and Plot ---
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    for i, layer_name in enumerate(layer_names):
        ax = axes[i]
        layer = getattr(model, layer_name)

        gate_values_for_this_layer = []
        row_labels = []
        num_blocks = 0

        # Collect gate values from each block in the current layer
        for bidx, block in enumerate(layer):
            gates = []
            for gid in [1, 2, 3]:
                name = f'static_gate{gid}'
                if hasattr(block, name) and isinstance(getattr(block, name), torch.nn.Parameter):
                    arr = torch.sigmoid(getattr(block, name).data).cpu().numpy()
                    gates.append(arr)
                    row_labels.append(f'B{bidx+1}-g{gid}({len(arr)})')
            if gates:
                num_blocks += 1
                gate_values_for_this_layer.extend(gates)

        if not gate_values_for_this_layer:
            continue # Skip if no gates found in this layer

        # Stack the collected 1D arrays into a 2D matrix for the heatmap
        # Note: In Bottleneck (strategy B), gate1 (planes) and gate3 (planes*expansion) have different lengths.
        # We pad shorter vectors with zeros to the max length within this layer for visualization consistency.
        max_len = max(len(arr) for arr in gate_values_for_this_layer)
        padded_rows = []
        for arr in gate_values_for_this_layer:
            row = np.zeros((max_len,), dtype=np.float32)
            row[:len(arr)] = arr
            padded_rows.append(row)
        heatmap_data = np.vstack(padded_rows)

        # --- 3. Plot the Heatmap for the Current Layer ---
        # Use a fixed value range (vmin, vmax) for consistency across all plots
        im = ax.imshow(heatmap_data, cmap='viridis', vmin=0, vmax=1.0, aspect='auto')

        # Diagnostics: show distinct rounded values to detect anomalies (e.g., warmup should be ~{0, 0.73})
        uniq_vals = np.unique(np.round(heatmap_data, 2))
        logger.info(f"[Visualizer][{layer_name}] unique(rounded) values: {uniq_vals[:10]} ... total={len(uniq_vals)}")

        # 3. 计算最小值和最大值
        min_val = heatmap_data.min()
        max_val = heatmap_data.max()

        # 4. 创建要显示的文本字符串
        range_text = f"Range: [{min_val:.2f}, {max_val:.2f}]"

        # 5. 将文本添加到子图的右上角
        ax.text(0.98, 0.95,  # x, y 坐标 (在右上角)
                range_text,  # 要显示的文本
                transform=ax.transAxes,  # 使用坐标轴相对坐标系
                fontsize=12,
                color='white',
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5)) # 添加一个半透明的黑色背景框

        # --- 4. Set Titles and Labels for Clarity ---
        ax.set_title(f'{layer_name.capitalize()} Gates (Shape: {heatmap_data.shape})', fontsize=14)
        ax.set_xlabel('Channel Index', fontsize=12)
        ax.set_ylabel('Block & Gate', fontsize=12)

        # Create meaningful labels for the y-axis
        y_ticks = np.arange(len(gate_values_for_this_layer))
        y_labels = row_labels if 'row_labels' in locals() and len(row_labels)==len(gate_values_for_this_layer) else [f'Entry{j+1}' for j in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

    # --- 5. Create a Single, Shared Color Bar for the Entire Figure ---
    if im:
        # Position the color bar to the right of all subplots
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)

    # --- 6. Adjust Layout and Save the Figure ---
    # Adjust layout to prevent titles/labels from overlapping
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Create the save directory if it doesn't exist
    vis_save_dir = os.path.join(save_dir, 'static_gate_heatmaps')
    os.makedirs(vis_save_dir, exist_ok=True)
    
    # Save the figure
    save_path = os.path.join(vis_save_dir, f'gate_heatmap_epoch_{epoch:03d}.png')
    plt.savefig(save_path)
    plt.close(fig) # Close the figure to free up memory

    logger.info(f"[Visualizer] Saved static gate heatmap to {save_path}")


def log_uncertainty_heatmap(epoch, batch_idx, uncertainty_scores_list, save_dir, logger, num_classes=10):
    """
    Visualizes the uncertainty scores (entropy) for a batch of samples at different layers.
    Creates a single figure with one subplot per layer and saves it.
    """
    logger.info(f"[Visualizer] Generating uncertainty heatmap for epoch {epoch}, batch {batch_idx}...")
    
    # --- 1. Create Figure and Subplots ---
    num_layers = len(uncertainty_scores_list)
    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 4 * num_layers), squeeze=False)
    fig.suptitle(f'Uncertainty (Entropy) Scores - Epoch {epoch}, Batch {batch_idx}', fontsize=16)
    axes = axes.flatten() # Ensure axes is always a flat array

    # --- 2. Iterate Through Layers to Collect Data and Plot ---
    for i, scores_tensor in enumerate(uncertainty_scores_list):
        ax = axes[i]
        
        # Prepare data for plotting
        scores_np = scores_tensor.detach().cpu().numpy()
        # Reshape from (batch_size,) to (batch_size, 1) for imshow
        heatmap_data = scores_np.reshape(-1, 1)
        
        # --- 3. Plot the Heatmap for the Current Layer ---
        # Use a fixed value range based on theoretical max entropy for comparability
        max_entropy = math.log(num_classes)
        im = ax.imshow(heatmap_data, cmap='plasma', vmin=0, vmax=max_entropy, aspect='auto')
        
        # --- 4. Calculate and Annotate Range ---
        min_val = heatmap_data.min()
        max_val = heatmap_data.max()
        range_text = f"Range: [{min_val:.3f}, {max_val:.3f}]"
        
        ax.text(0.98, 0.95, range_text,
                transform=ax.transAxes, fontsize=10, color='white',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

        # --- 5. Set Titles and Labels ---
        ax.set_title(f'Layer {i+1} Uncertainty (Shape: {heatmap_data.shape})', fontsize=12)
        ax.set_xlabel('Uncertainty Score', fontsize=10)
        ax.set_ylabel('Sample Index in Batch', fontsize=10)
        ax.set_xticks([]) # Hide x-axis ticks as it's just one column

    # --- 6. Create a Single, Shared Color Bar ---
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02, label='Entropy')

    # --- 7. Adjust Layout and Save the Figure ---
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    vis_save_dir = os.path.join(save_dir, 'uncertainty_heatmaps')
    os.makedirs(vis_save_dir, exist_ok=True)
    
    save_path = os.path.join(vis_save_dir, f'uncertainty_epoch_{epoch:03d}_batch_{batch_idx:03d}.png')
    plt.savefig(save_path)
    plt.close(fig)

    logger.info(f"[Visualizer] Saved uncertainty heatmap to {save_path}")

def log_uncertainty_budget_heatmap(epoch, batch_idx, uncertainty_scores_list, budget_factors_list, save_dir, logger, num_classes=10):
    """
    Visualizes uncertainty scores (entropy) and their corresponding generated budget factors
    for a batch of samples at different layers.
    Creates a single figure with one row per decision point, showing entropy on the left
    and budget factor on the right.
    """
    logger.info(f"[Visualizer] Generating uncertainty-budget heatmap for epoch {epoch}, batch {batch_idx}...")
    
    # --- 1. Create Figure and Subplots ---
    # Number of decision points is the number of uncertainty scores calculated
    num_decision_points = len(uncertainty_scores_list)
    
    # We create `num_decision_points` rows and 2 columns (Entropy | Budget)
    fig, axes = plt.subplots(num_decision_points, 2, figsize=(12, 4 * num_decision_points), squeeze=False)
    fig.suptitle(f'Uncertainty to Budget Mapping - Epoch {epoch}, Batch {batch_idx}', fontsize=16)

    # --- 2. Iterate Through Decision Points to Plot ---
    for i in range(num_decision_points):
        ax_uncertainty = axes[i, 0] # Left subplot for uncertainty
        ax_budget = axes[i, 1]      # Right subplot for budget factor

        # --- 3. Plot Uncertainty Heatmap (Left Column) ---
        scores_tensor = uncertainty_scores_list[i]
        scores_np = scores_tensor.detach().cpu().numpy().reshape(-1, 1)
        
        # Use a fixed value range based on theoretical max entropy
        max_entropy = math.log(num_classes)
        im_uncertainty = ax_uncertainty.imshow(scores_np, cmap='plasma', vmin=0, vmax=max_entropy, aspect='auto')
        
        # Annotate range
        min_val_u = scores_np.min()
        max_val_u = scores_np.max()
        range_text_u = f"Range: [{min_val_u:.3f}, {max_val_u:.3f}]"
        ax_uncertainty.text(0.98, 0.95, range_text_u, transform=ax_uncertainty.transAxes, fontsize=10, color='white',
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

        # Set titles and labels
        ax_uncertainty.set_title(f'Layer {i+1} Uncertainty (Entropy)', fontsize=12)
        ax_uncertainty.set_ylabel('Sample Index in Batch', fontsize=10)
        ax_uncertainty.set_xticks([])

        # --- 4. Plot Budget Factor Heatmap (Right Column) ---
        budget_tensor = budget_factors_list[i]
        budget_np = budget_tensor.detach().cpu().numpy().reshape(-1, 1)
        
        # Use a fixed value range [0, 1] for budget factors
        im_budget = ax_budget.imshow(budget_np, cmap='viridis', vmin=0, vmax=1.0, aspect='auto')
        
        # Annotate range
        min_val_b = budget_np.min()
        max_val_b = budget_np.max()
        range_text_b = f"Range: [{min_val_b:.3f}, {max_val_b:.3f}]"
        ax_budget.text(0.98, 0.95, range_text_b, transform=ax_budget.transAxes, fontsize=10, color='white',
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

        # Set titles and labels
        ax_budget.set_title(f'Layer {i+2} Budget Factor', fontsize=12)
        ax_budget.set_yticks([]) # Hide y-axis ticks as they are shared with the left plot
        ax_budget.set_xticks([])

    # --- 5. Create Shared Color Bars ---
    # Place color bar for uncertainty on the left
    fig.colorbar(im_uncertainty, ax=axes[:, 0].ravel().tolist(), shrink=0.8, pad=0.02, label='Entropy', location='left')
    # Place color bar for budget factor on the right
    fig.colorbar(im_budget, ax=axes[:, 1].ravel().tolist(), shrink=0.8, pad=0.02, label='Budget Factor',location='left')

    # --- 6. Adjust Layout and Save the Figure ---
    fig.tight_layout(rect=[0.05, 0.03, 0.95, 0.95]) # Adjust rect to make space for color bars
    
    vis_save_dir = os.path.join(save_dir, 'uncertainty_budget_heatmaps') # New directory
    os.makedirs(vis_save_dir, exist_ok=True)
    
    save_path = os.path.join(vis_save_dir, f'uncertainty_budget_epoch_{epoch:03d}_batch_{batch_idx:03d}.png')
    plt.savefig(save_path)
    plt.close(fig)

    logger.info(f"[Visualizer] Saved uncertainty-budget heatmap to {save_path}")

def log_model_structure_comparison(original_model, pruned_model, logger):
    """
    Compares the structure of the original and pruned models side-by-side.
    """
    logger.info("\n" + "="*80)
    logger.info(" " * 25 + "Model Structure Comparison")
    logger.info("="*80)
    logger.info(f"{'Layer Name':<30} | {'Original Shape':<20} | {'Pruned Shape':<20} | {'Reduction'}")
    logger.info("-"*80)

    original_modules = dict(original_model.named_modules())
    
    total_params_orig = 0
    total_params_pruned = 0

    for name, pruned_module in pruned_model.named_modules():
        if isinstance(pruned_module, (nn.Conv2d, nn.Linear)):
            if name in original_modules:
                original_module = original_modules[name]
                orig_shape = tuple(original_module.weight.shape)
                pruned_shape = tuple(pruned_module.weight.shape)
                
                orig_params = original_module.weight.numel()
                pruned_params = pruned_module.weight.numel()
                total_params_orig += orig_params
                total_params_pruned += pruned_params
                
                reduction_str = ""
                if orig_shape != pruned_shape:
                    if isinstance(pruned_module, nn.Conv2d):
                        reduction = orig_shape[0] - pruned_shape[0]
                        reduction_str = f"(-{reduction} out_ch)"
                    elif isinstance(pruned_module, nn.Linear):
                        reduction = orig_shape[1] - pruned_shape[1]
                        reduction_str = f"(-{reduction} in_feat)"

                logger.info(f"{name:<30} | {str(orig_shape):<20} | {str(pruned_shape):<20} | {reduction_str}")

    logger.info("-"*80)
    logger.info(f"Total Conv/FC Params (Original): {total_params_orig/1e6:.2f}M")
    logger.info(f"Total Conv/FC Params (Pruned):   {total_params_pruned/1e6:.2f}M")
    reduction_percent = (1 - total_params_pruned / total_params_orig) * 100
    logger.info(f"Parameter Reduction: {reduction_percent:.2f}%")
    logger.info("="*80 + "\n")


def validate(epoch, val_loader, model, criterion, logger, is_warmup=False):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            # logits = model(images)
            logits, _, _,_,_ = model(images,is_warmup=is_warmup)
            loss = criterion(logits, target)
            pred1, pred5 = utils.accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)
            batch_time.update(time.time() - end)
            end = time.time()
        # logger.info(' * Validation: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        if epoch == -1: # A special value to indicate final validation
            logger.info(' * Final Performance Validation: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        else:
            logger.info(' * Validation (Epoch {epoch}): Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(epoch=epoch, top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg

def calculate_target_channel_config(base_config, sparsity_rates):
    """
    Calculates the target channel configuration based on a base configuration
    and a list of sparsity rates for the current iteration.
    """
    base_out_channels, base_mid_channels = base_config
    
    # The sparsity_rates list has 16 elements, corresponding to the 8 blocks' conv1 and conv2
    target_out_channels = [base_out_channels[0]] # conv1 is not pruned
    target_mid_channels = []

    sparsity_idx = 0
    for i in range(8): # 8 blocks in ResNet-18
        # Mid channels (conv1 of the block)
        base_mid = base_mid_channels[i]
        mid_cprate = sparsity_rates[sparsity_idx]
        target_mid_channels.append(max(1, int(round(base_mid * (1 - mid_cprate)))))
        sparsity_idx += 1

        # Out channels (conv2 of the block)
        base_out = base_out_channels[i+1] # base_out_channels[0] is for conv1
        out_cprate = sparsity_rates[sparsity_idx]
        target_out_channels.append(max(1, int(round(base_out * (1 - out_cprate)))))
        sparsity_idx += 1
        
    return (target_out_channels, target_mid_channels)

def _calculate_params_unified(model, mode='soft'):
    """
    A unified function to calculate parameter counts, configurable for either
    'soft' (differentiable proxy) or 'hard' (non-differentiable value) mode.
    This avoids expensive model recreation and profiling in the training loop.
    """
    assert mode in ['soft', 'hard'], "Mode must be either 'soft' or 'hard'"
    
    device = next(model.parameters()).device
    total_params = torch.tensor(0.0, device=device)

    # Branch for DeiT (Transformer) models: precise per-head and per-FFN-unit parameter counting
    if hasattr(model, 'blocks') and hasattr(model, 'embed_dim') and hasattr(model, 'num_heads') and not hasattr(model, 'conv1'):
        D = torch.tensor(float(model.embed_dim), device=device)  # model dimension
        H = torch.tensor(float(model.num_heads), device=device)  # num heads
        depth = len(model.blocks)

        kept_params = torch.tensor(0.0, device=device)
        max_prunable_total = torch.tensor(0.0, device=device)

        for blk in model.blocks:
            # ---- Heads (per-head) ----
            # Per-head prunable weights: q,k,v (D*d each) + proj (d*D) => 4*D*d, where d=D/H
            # Per-head prunable biases: q,k,v biases (d each); proj bias is treated as fixed (not head-prunable)
            d = (D / H)
            per_head_weights = 4.0 * D * d
            per_head_biases = 3.0 * d
            per_head_total = per_head_weights + per_head_biases  # = d*(4D+3)
            max_heads_prunable = H * per_head_total  # = 4*D^2 + 3*D

            # kept ratio from static head gate only (dynamic is sample-dependent; params are structural)
            if isinstance(blk.static_gate_heads, torch.nn.Parameter):
                if mode == 'soft':
                    keep_ratio_heads = torch.mean(torch.sigmoid(blk.static_gate_heads))
                else:
                    probs = torch.sigmoid(blk.static_gate_heads.data)
                    keep_ratio_heads = torch.mean((probs > 0.5).float())
            else:
                keep_ratio_heads = torch.tensor(1.0, device=device)

            kept_params += keep_ratio_heads * max_heads_prunable
            max_prunable_total += max_heads_prunable

            # ---- FFN units (per-unit) ----
            # Per-unit prunable weights/bias: fc1 column (D weights + 1 bias) + fc2 row (D weights)
            hidden_dim = torch.tensor(float(blk.mlp.fc1.out_features), device=device)
            per_unit = 2.0 * D + 1.0
            max_ffn_prunable = hidden_dim * per_unit

            if isinstance(blk.static_gate_ffn, torch.nn.Parameter):
                if mode == 'soft':
                    keep_ratio_ffn = torch.mean(torch.sigmoid(blk.static_gate_ffn))
                else:
                    probs_ffn = torch.sigmoid(blk.static_gate_ffn.data)
                    keep_ratio_ffn = torch.mean((probs_ffn > 0.5).float())
            else:
                keep_ratio_ffn = torch.tensor(1.0, device=device)

            kept_params += keep_ratio_ffn * max_ffn_prunable
            max_prunable_total += max_ffn_prunable

        # Fixed params = total - max_prunable_total
        total_params_full = torch.tensor(float(sum(p.numel() for p in model.parameters())), device=device)
        fixed_params = total_params_full - max_prunable_total
        total_params = fixed_params + kept_params
        return total_params

    # 1. Conv1 and BN1 (fixed structure)
    total_params += model.conv1.weight.numel()
    total_params += model.bn1.weight.numel() + model.bn1.bias.numel()

    in_channels = torch.tensor(float(model.conv1.out_channels), device=device)

    # 2. Layers 1-4
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for block in layer:
            # --- Determine channel counts based on block type and mode ---
            if hasattr(block, 'static_gate1') and hasattr(block, 'static_gate2'):
                # BasicBlock path
                if mode == 'soft':
                    mid_channels = torch.sum(torch.sigmoid(block.static_gate1))
                    out_channels = torch.sum(torch.sigmoid(block.static_gate2))
                else: # hard
                    probs1 = torch.sigmoid(block.static_gate1.data)
                    mid_channels = torch.sum((probs1 > 0.5).float())
                    probs2 = torch.sigmoid(block.static_gate2.data)
                    out_channels = torch.sum((probs2 > 0.5).float())

                # Conv1 + BN1
                k1_sq = block.conv1.kernel_size[0] ** 2
                total_params += in_channels * mid_channels * k1_sq
                total_params += 2 * mid_channels
                # Conv2 + BN2
                k2_sq = block.conv2.kernel_size[0] ** 2
                total_params += mid_channels * out_channels * k2_sq
                total_params += 2 * out_channels
                # Downsample
                if block.downsample is not None:
                    ds_conv = block.downsample[0]
                    ds_k_sq = ds_conv.kernel_size[0] ** 2
                    total_params += in_channels * out_channels * ds_k_sq
                    total_params += 2 * out_channels
                # Next in_channels
                in_channels = out_channels
            elif hasattr(block, 'static_gate3'):
                # BottleneckBlock path
                planes = block.conv1.out_channels
                if hasattr(block, 'static_gate1') and isinstance(block.static_gate1, torch.nn.Parameter):
                    # Strategy B: prune conv1 (mid) and conv3 (out)
                    if mode == 'soft':
                        mid_channels = torch.sum(torch.sigmoid(block.static_gate1))
                        out_channels = torch.sum(torch.sigmoid(block.static_gate3))
                    else:
                        probs1 = torch.sigmoid(block.static_gate1.data)
                        mid_channels = torch.sum((probs1 > 0.5).float())
                        probs3 = torch.sigmoid(block.static_gate3.data)
                        out_channels = torch.sum((probs3 > 0.5).float())
                    # conv1 1x1 + bn1
                    total_params += in_channels * mid_channels * 1
                    total_params += 2 * mid_channels
                    # conv2 3x3 + bn2
                    k2_sq = block.conv2.kernel_size[0] ** 2
                    total_params += mid_channels * mid_channels * k2_sq
                    total_params += 2 * mid_channels
                    # conv3 1x1 + bn3 (pruned on output channels)
                    total_params += mid_channels * out_channels * 1
                    total_params += 2 * out_channels
                    # downsample (to out_channels)
                    if block.downsample is not None:
                        ds_conv = block.downsample[0]
                        ds_k_sq = ds_conv.kernel_size[0] ** 2
                        total_params += in_channels * out_channels * ds_k_sq
                        total_params += 2 * out_channels
                    in_channels = out_channels
                else:
                    # Strategy A: only conv3 prunable
                    if mode == 'soft':
                        out_channels = torch.sum(torch.sigmoid(block.static_gate3))
                    else:
                        probs3 = torch.sigmoid(block.static_gate3.data)
                        out_channels = torch.sum((probs3 > 0.5).float())
                    # conv1 1x1 + bn1
                    total_params += in_channels * planes * 1
                    total_params += 2 * planes
                    # conv2 3x3 + bn2
                    k2_sq = block.conv2.kernel_size[0] ** 2
                    total_params += planes * planes * k2_sq
                    total_params += 2 * planes
                    # conv3 1x1 + bn3
                    total_params += planes * out_channels * 1
                    total_params += 2 * out_channels
                    # downsample
                    if block.downsample is not None:
                        ds_conv = block.downsample[0]
                        ds_k_sq = ds_conv.kernel_size[0] ** 2
                        total_params += in_channels * out_channels * ds_k_sq
                        total_params += 2 * out_channels
                    in_channels = out_channels
            else:
                # Fallback: treat as fixed (no pruning)
                # Use module's defined channels as scalars
                planes = block.conv1.out_channels
                k2_sq = block.conv2.kernel_size[0] ** 2
                out_ch = block.conv2.out_channels if hasattr(block.conv2, 'out_channels') else planes
                total_params += in_channels * planes * 1 + 2 * planes
                total_params += planes * out_ch * k2_sq + 2 * out_ch
                if block.downsample is not None:
                    ds_conv = block.downsample[0]
                    ds_k_sq = ds_conv.kernel_size[0] ** 2
                    total_params += in_channels * out_ch * ds_k_sq + 2 * out_ch
                in_channels = torch.tensor(float(out_ch), device=device)

    # 3. FC layer
    total_params += in_channels * model.fc.out_features + model.fc.bias.numel()

    return total_params


def calculate_hard_params(model):
    """
    Calculates the precise number of parameters for the current hard-gated model
    by calling the unified calculator in 'hard' mode.
    Returns a standard float (no gradients).
    """
    hard_params_tensor = _calculate_params_unified(model, mode='hard')
    return hard_params_tensor.item()

def calculate_soft_params_proxy(model):
    """
    Calculates a differentiable proxy for the total number of parameters
    by calling the unified calculator in 'soft' mode.
    Returns a differentiable PyTorch tensor.
    """
    return _calculate_params_unified(model, mode='soft')

def calculate_params_loss(model, target_params_total):
    """
    Calculates the regularization loss based on the deviation of the current
    parameter count from a target value. Uses STE.
    The forward loss value is based on a hard, precise parameter count.
    The backward gradient is based on a soft, differentiable proxy.
    """
    # --- Step 1: Calculate loss based on soft probabilities (for gradient) ---
    actual_params_proxy_soft = calculate_soft_params_proxy(model)
    loss_soft = (actual_params_proxy_soft / target_params_total - 1)**2
    
    # --- Step 2: Calculate loss based on hard mask (for value) ---
    # This part does not have gradients.
    actual_params_hard = calculate_hard_params(model)
    loss_hard = (torch.tensor(actual_params_hard, device=loss_soft.device) / target_params_total - 1)**2
    
    # --- Step 3: Apply STE ---
    loss_params_ste = loss_soft + (loss_hard - loss_soft).detach()
    
    return loss_params_ste


def calculate_hardest_ranking_loss(scores, budgets, margin):
    """
    计算基于单向最难样本采样的Pair-wise Ranking Loss。
    对于批次中的每个样本i，找到比它简单的样本中预算最高的那个j，
    并确保 budget[i] > budget[j] + margin。

    Args:
        scores (torch.Tensor): 形状为 [N] 的难易度分数张量。
        budgets (torch.Tensor): 形状为 [N] 的预算因子张量。
        margin (float): 期望的预算间隔。

    Returns:
        torch.Tensor: 一个标量的损失值。
    """
    # 1. 高效构建所有样本对的比较矩阵
    # [N, 1] - [1, N] -> [N, N]
    scores_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    
    # 2. 找到“正样本对”：(i, j) 其中 score[i] > score[j]
    # positive_mask[i, j] is True if score[i] > score[j]
    positive_mask = scores_diff > 0
    
    # 3. 筛选出有效的锚点：那些至少存在一个比它更简单的样本的锚点
    # valid_anchors_mask 形状为 [N]，如果第i行至少有一个True，则为True
    valid_anchors_mask = torch.any(positive_mask, dim=1)
    
    # 如果没有有效的锚点对，直接返回0损失
    if not torch.any(valid_anchors_mask):
        return torch.tensor(0.0, device=scores.device)

    # 4. 挖掘最难的正样本
    # 我们想在 positive_mask 为 True 的地方找到 budgets[j] 的最大值
    # [1, N] -> [N, N]
    budgets_j = budgets.unsqueeze(0).expand(budgets.size(0), -1)
    
    # 将无效的候选者（非正样本对）的预算设置为-inf，这样它们就不会被max选中
    masked_budgets_j = budgets_j.clone()
    masked_budgets_j[~positive_mask] = -torch.inf
    
    # 沿着j的维度（dim=1）找到每个锚点i对应的最难正样本的预算
    hardest_positive_budgets, _ = torch.max(masked_budgets_j, dim=1)
    
    # 5. 计算Ranking Loss
    # 我们只对那些有效的锚点计算损失
    anchor_budgets = budgets[valid_anchors_mask]
    hardest_budgets_for_anchors = hardest_positive_budgets[valid_anchors_mask]
    
    # Hinge Loss: max(0, margin - (budget_anchor - budget_hardest_positive))
    loss = torch.relu(margin - (anchor_budgets - hardest_budgets_for_anchors))
    
    # 返回所有有效锚点对的平均损失
    return loss.mean()

def finetune(model: nn.Module,
             train_loader: DataLoader,
             val_loader: DataLoader,
             epochs: int,
             main_optimizer,
             gating_optimizer,
             aux_optimizer,
             budget_optimizer,
             scheduler,
             logger,
             result_dir: str,
             w_aux: float,
             w_flops: float,
             w_ranking: float,          
             ranking_margin: float,     
             w_params_final: float,
             w_params_anneal_epochs: int,
             final_target_params: float,
             warmup_epochs: int,
             # --- NEW FLOPs PARAMS ---
             target_total_flops: float,
             target_total_flops_train: float,
             scalable_flops_dict: dict,
             total_overhead_flops: float):
    """
    Performs end-to-end training with the new unified FLOPs loss.
    """
    logger.info("\n" + "#"*60)
    logger.info("#####      STARTING UNIFIED ADAPTIVE TRAINING      #####")
    logger.info(f"Total Epochs: {epochs}, Warmup Epochs: {warmup_epochs}")
    logger.info(f"Loss Weights: w_aux={w_aux}, w_flops={w_flops}, w_params_final={w_params_final}")
    logger.info(f"Target Total FLOPs: {target_total_flops / 1e6:.2f} MFLOPs")
    logger.info(f"Target Total FLOPs (Train): {target_total_flops_train / 1e6:.2f} MFLOPs")
    logger.info("#"*60 + "\n")
    
    aux_loss_weights = [1.0, 1.0, 1.0]
    cudnn.benchmark = True
    cudnn.enabled = True
    criterion = nn.CrossEntropyLoss().cuda()
    best_top1_acc = 0.0

    target_vis_epochs = [5, 35, 75, 105, 135, 165, 195]
    target_vis_uncertainty_budget_batches = [
        (epoch, batch) for epoch in [5, 15, 35, 50, 75, 100,115,135, 150,175, 195] for batch in [0, 1]
    ]

    for epoch in range(epochs):
        model.train()
        
        # Parameter loss weight annealing
        anneal_start_epoch = warmup_epochs
        anneal_end_epoch = warmup_epochs + w_params_anneal_epochs
        current_w_params = 0.0
        if epoch >= anneal_end_epoch:
            current_w_params = w_params_final
        elif epoch >= anneal_start_epoch:
            progress = (epoch - anneal_start_epoch) / w_params_anneal_epochs
            current_w_params = progress * w_params_final
        
        w_flops_anneal_epochs = w_params_anneal_epochs
        current_w_flops = 0.0
        if epoch >= anneal_end_epoch:
            current_w_flops = w_flops
        elif epoch >= anneal_start_epoch:
            progress = (epoch - anneal_start_epoch) / w_flops_anneal_epochs
            current_w_flops = progress * w_flops

        is_warmup_phase = epoch < warmup_epochs
        phase_log = "[WARMUP]" if is_warmup_phase else "[JOINT TRAINING]"
        logger.info(f"[Epoch {epoch+1}/{epochs}] {phase_log} LR: {main_optimizer.param_groups[0]['lr']:.6f} | w_params: {current_w_params:.4f}")
        
        pbar = tqdm(train_loader, desc=f"Train E{epoch+1}")

        for i, (images, target) in enumerate(pbar):
            images, target = images.cuda(), target.cuda()
            
            # 1. Forward Pass
            main_logits, aux_logits_list, budget_factors_list, uncertainty_scores_list, all_block_masks = model(images, is_warmup=is_warmup_phase)
            
            if (epoch, i) in target_vis_uncertainty_budget_batches:
                if is_warmup_phase:
                    log_uncertainty_heatmap(
                        epoch=epoch,
                        batch_idx=i,
                        uncertainty_scores_list=uncertainty_scores_list,
                        save_dir=result_dir,
                        logger=logger
                    )
                else:
                    log_uncertainty_budget_heatmap(
                        epoch=epoch,
                        batch_idx=i,
                        uncertainty_scores_list=uncertainty_scores_list,
                        budget_factors_list=budget_factors_list, # <-- Pass the new list
                        save_dir=result_dir,
                        logger=logger
                    )

            # 2. Calculate Losses
            loss_main = criterion(main_logits, target)
            
            individual_aux_losses = []
            if aux_logits_list:
                for aux_logits in aux_logits_list:
                    individual_aux_losses.append(criterion(aux_logits, target))

            if is_warmup_phase:
                loss_flops = torch.tensor(0.0, device=images.device)
                loss_params = torch.tensor(0.0, device=images.device)
                display_flops_val = 0.0
            else:
                # NEW: Unified FLOPs loss calculation
                loss_flops, display_flops_val = calculate_flops_loss(
                    model, all_block_masks, budget_factors_list,scalable_flops_dict, target_total_flops_train
                )
                # Parameter loss calculation (remains the same)
                loss_params = calculate_params_loss(model, final_target_params)

            # 3. Zero All Gradients
            main_optimizer.zero_grad()
            gating_optimizer.zero_grad()
            aux_optimizer.zero_grad()
            budget_optimizer.zero_grad()

            if is_warmup_phase:
                # In warmup, only main and aux losses are active
                loss_main.backward(retain_graph=True) # Retain graph for subsequent backward calls
                
                if individual_aux_losses:
                    # Backward for each aux loss separately
                    for idx, aux_loss in enumerate(individual_aux_losses):
                        # The last backward call does not need retain_graph=True
                        is_last_backward = (idx == len(individual_aux_losses) - 1)
                        weighted_aux_loss = w_aux * aux_loss_weights[idx] * aux_loss
                        weighted_aux_loss.backward(retain_graph=not is_last_backward)
            else: # Joint Training Phase
                loss_params = calculate_params_loss(model, final_target_params) # NEW
                loss_ranking_total = torch.tensor(0.0, device=images.device)
                if budget_factors_list: # 确保列表不为空
                    # 遍历每个决策点，计算并累加ranking loss
                    for scores, budgets in zip(uncertainty_scores_list, budget_factors_list):
                        loss_ranking_total += calculate_hardest_ranking_loss(
                            scores=scores,
                            budgets=budgets,
                            margin=ranking_margin
                        )

                # --- Backprop Step 1: For Budget Generator, Gating, and Backbone ---
                loss_for_main_path = loss_main + current_w_flops * loss_flops+ w_ranking * loss_ranking_total
                loss_for_main_path.backward(retain_graph=True)

                # --- Backprop Step 2: For Static Gates ---
                # loss_for_static_gates = w_params * loss_params
                loss_for_static_gates = current_w_params * loss_params # NEW
                loss_for_static_gates.backward(retain_graph=True)

                # --- Backprop Step 3: For Auxiliary Heads ---
                if individual_aux_losses:
                    for idx, aux_loss in enumerate(individual_aux_losses):
                        is_last_backward = (idx == len(individual_aux_losses) - 1)
                        weighted_aux_loss = w_aux * aux_loss_weights[idx] * aux_loss
                        weighted_aux_loss.backward(retain_graph=not is_last_backward)

            # 5. Clip Gradients and Step Optimizers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            main_optimizer.step()
            aux_optimizer.step()
            
            if not is_warmup_phase:
                gating_optimizer.step()
                budget_optimizer.step()
            
            scheduler.step()
            
            display_aux_loss = sum(l.item() for l in individual_aux_losses) if individual_aux_losses else 0.0
            pbar.set_postfix(loss_main=loss_main.item(), loss_aux=display_aux_loss, loss_flops=loss_flops.item(), loss_params=loss_params.item())
        
        # --- End of Batch Loop ---
        if epoch in target_vis_epochs:
            # Diagnostics in warmup: static gates should remain ~1.0 (sigmoid ~0.73)
            if is_warmup_phase:
                mins = []
                maxs = []
                for m in model.modules():
                    for gname in ['static_gate1','static_gate2','static_gate3']:
                        if hasattr(m, gname) and isinstance(getattr(m, gname), torch.nn.Parameter):
                            p = getattr(m, gname).detach().cpu()
                            mins.append(float(p.min()))
                            maxs.append(float(p.max()))
                if mins and maxs:
                    logger.info(f"[WarmupDiag] static_gate param ranges: min={min(mins):.4f}, max={max(maxs):.4f} (expect ~1.0)")
            log_static_gate_heatmap(model=model, epoch=epoch, save_dir=result_dir, logger=logger)

        # validate(epoch, val_loader, model, criterion, logger, is_warmup=is_warmup_phase)
        # Validate after each epoch (common for both phases)
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion, logger, is_warmup=is_warmup_phase)

        is_best = valid_top1_acc > best_top1_acc
        if is_best:
            best_top1_acc = valid_top1_acc
            logger.info(f"==> New best accuracy: {best_top1_acc:.2f}% at epoch {epoch+1}")

        # Save checkpoint (common for both phases)
        checkpoint_path = os.path.join(result_dir, f'e2e_1')
        os.makedirs(checkpoint_path, exist_ok=True)
        utils.save_checkpoint({
            'epoch': epoch, 'state_dict': model.state_dict(), 'best_top1_acc': best_top1_acc,
            'main_optimizer': main_optimizer.state_dict(), 'mask_optimizer': gating_optimizer.state_dict(),
        }, is_best, checkpoint_path)

        # Proactively release cached memory to mitigate fragmentation before next epoch
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    logger.info(f"End-to-end training completed. Best validation accuracy: {best_top1_acc:.3f}%")
    return model # Return the fully trained model for post-processing