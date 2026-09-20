"""
Architecture optimization utilities for hardware-aware model optimization in medical imaging.

This module provides comprehensive implementations of modern neural network optimization
techniques specifically designed for clinical deployment scenarios. Focuses on reducing
computational overhead, memory usage, and inference latency while maintaining diagnostic
accuracy for the PneumoniaMNIST binary classification task.

Key optimization strategies:
    - Interpolation Removal: Eliminates computational overhead from resolution upscaling
    - Depthwise Separable Convolutions: Reduces parameters and FLOPs significantly
    - Grouped Convolutions: Parallel channel processing for improved efficiency
    - Inverted Residual Blocks: Mobile-optimized residual architectures
    - Low-Rank Factorization: Matrix decomposition for parameter reduction
    - Channel Optimization: Memory layout and activation optimizations
    - Parameter Sharing: Weight reuse across similar layer configurations
"""

import copy
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# =====================================================================
# REUSABLE OPTIMIZED BUILDING BLOCKS
# =====================================================================

class NativeResolutionWrapper(nn.Module):
    """Wrap a ResNetBaseline-style model to bypass input interpolation.

    The forward pass calls the underlying backbone directly at native
    resolution, eliminating the bilinear upscaling overhead identified as the
    dominant inefficiency in the baseline analysis.
    """

    def __init__(self, base_model: nn.Module, native_size: int) -> None:
        super().__init__()
        # Reuse the underlying backbone if available; otherwise wrap as-is.
        self.model = getattr(base_model, "model", base_model)
        self.input_size = native_size
        self.target_size = native_size
        self.num_classes = getattr(base_model, "num_classes", None)
        self.architecture_name = f"{getattr(base_model, 'architecture_name', 'Backbone')}-Native{native_size}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No interpolation: process the image at its native resolution.
        return self.model(x)


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable replacement for a standard Conv2d.

    Factorizes a spatial convolution into a per-channel depthwise convolution
    followed by a 1x1 pointwise convolution, drastically reducing FLOPs and
    parameters while preserving output shape (residual compatible).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 bias: bool = False) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=in_channels, bias=bias,
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.act(self.bn(self.depthwise(x))))


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block (expand -> depthwise -> project)."""

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # Pointwise expansion.
            layers += [
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            # Depthwise convolution.
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Linear pointwise projection (no activation).
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class LowRankLinear(nn.Module):
    """Low-rank factorization of a Linear layer: W approx = second @ first."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True) -> None:
        super().__init__()
        self.first = nn.Linear(in_features, rank, bias=False)
        self.second = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))


def _replace_module(root: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    """Replace a (possibly nested) submodule addressed by its dotted name."""
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def create_optimized_model(base_model: nn.Module, optimizations: Dict[str, Any]) -> nn.Module:
    """
    Apply selected optimization strategies in order to create a clinically-optimized model.

    Args:
        base_model: Original ResNet model to optimize for clinical deployment
        optimizations: Dictionary specifying which optimizations to apply with parameters:
            - 'interpolation_removal': bool - Remove upscaling overhead (recommended: True)
            - 'depthwise_separable': bool - Apply depthwise separable convolutions
            - 'grouped_conv': bool - Use grouped convolutions for parallel processing
            - 'channel_optimization': bool - Optimize memory layout and activations
            - 'inverted_residuals': bool - Replace blocks with inverted residuals
            - 'lowrank_factorization': bool - Apply matrix factorization to linear layers
            - 'parameter_sharing': bool - Share weights between similar layers
            
    Returns:
        Optimized model with selected techniques applied, ready for clinical deployment
        
    Example:
        >>> base_model = create_baseline_model()
        >>> optimization_config = {
        ...     'interpolation_removal': True,
        ...     'depthwise_separable': True,
        ...     'channel_optimization': True
        ... }
        >>> optimized_model = create_optimized_model(base_model, optimization_config)
        >>> print("Clinical deployment model ready")
    """
    model = copy.deepcopy(base_model)
  
    print("Starting clinical model optimization pipeline...")
    
    # Apply optimizations from coarse structural changes down to fine parameter tweaks:
    #   1. Architectural change (input resolution) first, so downstream ops see native size.
    #   2. Block/layer redesigns (inverted residuals, depthwise, grouped conv).
    #   3. Parameter-level compression (low-rank factorization of linear layers).
    #   4. Hardware/memory-layout opts (channels_last, in-place activations).
    #   5. Parameter sharing last, tying weights of whatever layers remain.
    optimization_order = [
        'interpolation_removal',   # architectural change: native-resolution input
        'inverted_residuals',      # block redesign (expand -> depthwise -> project)
        'depthwise_separable',     # layer modification: factorized convolutions
        'grouped_conv',            # layer modification: parallel channel groups
        'lowrank_factorization',   # parameter opt: low-rank linear decomposition
        'channel_optimization',    # hardware opt: channels_last + in-place ReLU
        'parameter_sharing',       # parameter opt: tie identical-shape weights
    ]
    
    # Optimization function mapping - connects optimization names to their implementation
    # IMPORTANT: Make sure to experiment with different input parameters for each optimization function, if performance is suboptimal
    optimization_functions = {
        'interpolation_removal': lambda m: apply_interpolation_removal_optimization(m),
        'depthwise_separable': lambda m: apply_depthwise_separable_optimization(m),
        'grouped_conv': lambda m: apply_grouped_convolution_optimization(m),
        'channel_optimization': lambda m: apply_channel_optimization(m),
        'inverted_residuals': lambda m: apply_inverted_residual_optimization(m),
        'lowrank_factorization': lambda m: apply_lowrank_factorization(m),
        'parameter_sharing': lambda m: apply_parameter_sharing(m)
    }
    
    # Smart iteration through the defined optimization order
    applied_optimizations = []
    for opt_name in optimization_order:
        # Check if this optimization is requested and available
        if optimizations.get(opt_name, False) and opt_name in optimization_functions:
            print(f"   Applying {opt_name.replace('_', ' ')} optimization...")
            try:
                # Apply the optimization using the mapped function
                model = optimization_functions[opt_name](model)
                applied_optimizations.append(opt_name)
            except Exception as e:
                print(f"   ERROR: {opt_name} optimization failed: {e}")
        elif opt_name not in optimization_functions:
            print(f"   WARNING: Unknown optimization: {opt_name}")
    
    # Report results
    if applied_optimizations:
        print(f"Applied optimizations in order: {' → '.join(applied_optimizations)}")
    else:
        print("No optimizations were applied")
        
    return model

# --------------------------------------
# INTERPOLATION REMOVAL (NATIVE RESOLUTION)
# --------------------------------------

# TODO: Implement this optimization method, if selected in your optimization strategy
def apply_interpolation_removal_optimization(model: nn.Module, native_size: int = 64) -> nn.Module:
    """
    Remove interpolation overhead by processing images at native resolution.
    
    Args:
        model: Model with interpolation capability (e.g., ResNetBaseline)
        native_size: Native input resolution to process (64 for clinical deployment)
        
    Returns:
        Optimized model that processes at native resolution without interpolation

    Note: 
        In `data_loader.py`, we would also want to replace ImageNet stats with chest 
        X-ray specific to check if accuracy improves, but you can skip this for simplicity 
        as normalization affects accuracy/sensitivity and not operational efficiency.
        
    Example:
        >>> baseline_model = create_baseline_model()
        >>> optimized_model = apply_interpolation_removal_optimization(baseline_model, 64)
        >>> # Model now processes 64x64 images directly without upscaling
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)

    print(f"Applying native resolution optimization ({native_size}x{native_size})...")
    
    # TODO: Update the existing model class to bypasses interpolation and processes images at native resolution.
    # HINT: The ResNetBaseline model automatically interpolates input images from 64x64 to 224x224 
    # before passing them to the underlying ResNet. One option is to create a wrapper that:
    # 1. Stores the original model architecture and metadata
    # 2. Updates the input_size attribute to reflect native processing  
    # 3. In the forward pass, bypasses the interpolation step entirely
    # 4. Directly calls the underlying ResNet model (model.model if it's a ResNetBaseline)
    #
    # See the ResNetBaseline.forward() method to understand how interpolation currently works.

    if hasattr(optimized_model, "target_size") and hasattr(optimized_model, "model"):
        # Fast path: the baseline only interpolates when height != target_size.
        # Setting target_size to the native resolution disables interpolation with
        # zero wrapping overhead while preserving all trained weights.
        optimized_model.target_size = native_size
        optimized_model.input_size = native_size
        optimized_model.architecture_name = (
            f"{getattr(optimized_model, 'architecture_name', 'ResNet')}-Native{native_size}"
        )
    else:
        # Robust fallback for arbitrary models: wrap to call the backbone directly.
        optimized_model = NativeResolutionWrapper(optimized_model, native_size)

    # Report optimization status and provide deployment guidance
    print("INTERPOLATION REMOVAL completed.")
    
    return optimized_model

# --------------------------------------
# DEPTHWISE SEPARABLE CONVOLUTION MODULES
# --------------------------------------

# TODO: Implement this optimization method, if selected in your optimization strategy
def apply_depthwise_separable_optimization(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
    min_channels: int = 16,
    preserve_residuals: bool = True
) -> nn.Module:
    """
    Convert suitable Conv2d layers to DepthwiseSeparableConv2d for clinical efficiency.
    
    Systematically replaces standard convolutions with depthwise separable alternatives
    to reduce computational cost and memory usage while preserving diagnostic accuracy.
    Essential for deploying medical imaging models on resource-constrained devices.
    
    Args:
        model: Input model to optimize for clinical deployment
        layer_names: Specific layer names to convert (None = convert all suitable layers)
        min_channels: Minimum input/output channels required for conversion
        preserve_residuals: Use residual-compatible configurations for ResNet models
        
    Returns:
        Optimized model with depthwise separable convolutions applied
        
    Note:
        Only converts layers that benefit from depthwise separation (kernel_size > 1,
        sufficient channels, not already grouped). Preserves ResNet compatibility by
        maintaining residual connection requirements.
        
    Example:
        >>> model = create_baseline_model()
        >>> optimized_model = apply_depthwise_separable_optimization(
        ...     model, min_channels=32
        ... )
        >>> # Suitable Conv2d layers now use depthwise separable convolutions
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    replacements = 0  # Track number of successful replacements

    print("Applying depthwise separable convolution optimization...")

    # TODO: Update the model to use depthwise separable convolution instead of convolution. 
    # HINT: To transform a conv2d into depthwise separable, you need to convolve each channel with its own kernel (groups=in_channels) for depthwise, 
    # and then combine information across channels processed by depthwise layer to define the pointwise layer.
    # Note that a conv2d block is also composed by activation and batchnorm in ResNet - Do you want to keep both, either, or none in?
    # Also, think about how the residuals are handled.
    # See https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/ for an intuitive explanation and code template.

    # Collect conversion targets first to avoid mutating during iteration.
    candidates: List[Tuple[str, nn.Conv2d]] = []
    for name, module in optimized_model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if layer_names is not None and name not in layer_names:
            continue
        # Only spatial, non-grouped convs with enough channels benefit and stay
        # residual-compatible (output shape is preserved by construction).
        if (module.kernel_size[0] > 1
                and module.groups == 1
                and module.in_channels >= min_channels
                and module.out_channels >= min_channels):
            candidates.append((name, module))

    for name, conv in candidates:
        replacement = DepthwiseSeparableConv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size[0],
            stride=conv.stride[0],
            padding=conv.padding[0],
            dilation=conv.dilation[0],
            bias=conv.bias is not None,
        )
        _replace_module(optimized_model, name, replacement)
        replacements += 1

    # Report optimization status
    if replacements > 0:
        print(f"DEPTHWISE SEPARABLE completed: Successfully applied to layers with {replacements} replacements")
    else:
        print("WARNING: DEPTHWISE SEPARABLE not applied: No suitable layers found for replacement")

    return optimized_model

# --------------------------------------
# GROUPED CONVOLUTION MODULES
# --------------------------------------

# TODO: Implement this optimization method, if selected in your optimization strategy
def apply_grouped_convolution_optimization(
    model: nn.Module,
    groups: int = 2,
    min_channels: int = 32,
    layer_names: Optional[List[str]] = None,
    do_depthwise: Optional[bool] = False,
) -> nn.Module:
    """
    Convert suitable Conv2d layers to grouped convolutions for parallel efficiency.
    
    Args:
        model: Input model to optimize
        groups: Number of groups for grouped convolution (typically 2-8)
        min_channels: Minimum channels required for conversion
        layer_names: Specific layers to convert (None = all suitable layers)
        do_depthwise: Whether to apply depthwise grouping (groups=in_channels)
        
    Returns:
        Model with grouped convolutions applied for enhanced efficiency
        
    Note:
        Grouped convolutions can be highly efficient on certain hardware backends, 
        especially when used with memory formats like channels_last and mixed precision (AMP)
        
    Example:
        >>> model = create_baseline_model()
        >>> optimized_model = apply_grouped_convolution_optimization(
        ...     model, groups=4, min_channels=64
        ... )
        >>> # Suitable layers now use 4-group parallel processing
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    # Track number of successful and skipped replacements
    replacements = 0
    skipped = 0

    print(f"Applying grouped convolution optimization (groups={groups})...")

    # TODO: Convert suitable Conv2d layers to grouped convolutions.
    # HINT: Grouped convolution divides input channels into independent groups and applies separate 
    # convolutions to each group. To make this happen, you need to ensure that the later is suitable for this transformation.
    # See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html for how to use the group parameter.

    candidates: List[Tuple[str, nn.Conv2d]] = []
    for name, module in optimized_model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if layer_names is not None and name not in layer_names:
            continue
        if module.kernel_size[0] > 1 and module.groups == 1 \
                and module.in_channels >= min_channels and module.out_channels >= min_channels:
            candidates.append((name, module))

    for name, conv in candidates:
        # Depthwise grouping uses groups=in_channels; otherwise use the requested count.
        target_groups = conv.in_channels if do_depthwise else groups
        # Grouped conv requires both channel counts divisible by the group count.
        if conv.in_channels % target_groups != 0 or conv.out_channels % target_groups != 0:
            skipped += 1
            continue

        grouped = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=target_groups,
            bias=conv.bias is not None,
        )
        _replace_module(optimized_model, name, grouped)
        replacements += 1

    # Report optimization status and provide deployment tipes
    if replacements > 0:
        print(f"GROUPED CONV completed: Successfully applied to layers with {replacements} replacements. Skipped {skipped} layers.")
        print("\nDEPLOYMENT TIP: For some hardware (like NVIDIA GPUs), grouped convolutions may require specific memory formats (channels_last) and mixed precision to achieve maximum throughput.")
    else:
        print("WARNING: GROUPED CONV not applied: No suitable layers found for replacement")

    return optimized_model

# --------------------------------------
# INVERTED RESIDUAL BLOCKS
# --------------------------------------

# TODO: Implement this optimization method, if selected in your optimization strategy
def apply_inverted_residual_optimization(
    model: nn.Module,
    target_layers: Optional[List[str]] = None,
    expand_ratio: int = 6
) -> nn.Module:
    """
    Replace suitable blocks with mobile-optimized InvertedResidual blocks.

    Args:
        model: Original model for mobile optimization
        target_layers: Specific layer names to convert (None = auto-detect suitable blocks)
        expand_ratio: Channel expansion factor for inverted residuals (6 is optimal)
        
    Returns:
        Model with mobile-optimized inverted residual blocks
        
    Note:
        This optimization targets BasicBlock structures and converts them to mobile-friendly
        inverted residuals. Most effective for deployment on edge devices and mobile platforms
        common in point-of-care medical applications.
        
    Example:
        >>> model = create_baseline_model()
        >>> mobile_model = apply_inverted_residual_optimization(
        ...     model, expand_ratio=6
        ... )
        >>> # Suitable blocks now use mobile-optimized inverted residuals
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    replacements = 0  # Track number of successful replacements

    print(f"Applying mobile inverted residual optimization...")
    
    # TODO: Replaces suitable blocks in the model with InvertedResidual blocks.
    # HINT: Inverted residuals use an expand→depthwise→project pattern as used in MobileNetV2.
    # The "inverted" aspect means we expand channels first (unlike standard residuals that compress).
    # Architecture flow: input → [expand] → depthwise → project → [+residual]
    #
    # Check the MobileNetV2 code at https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py 
    # for a code template, and consider whether to use ReLU or ReLU6 and batchnorm.

    from torchvision.models.resnet import BasicBlock

    candidates: List[Tuple[str, BasicBlock]] = []
    for name, module in optimized_model.named_modules():
        if not isinstance(module, BasicBlock):
            continue
        if target_layers is not None and name not in target_layers:
            continue
        candidates.append((name, module))

    for name, block in candidates:
        # Infer the block's I/O geometry directly from its convolutions.
        in_channels = block.conv1.in_channels
        out_channels = block.conv2.out_channels
        stride = block.conv1.stride[0]
        replacement = InvertedResidual(
            inp=in_channels,
            oup=out_channels,
            stride=stride,
            expand_ratio=expand_ratio,
        )
        _replace_module(optimized_model, name, replacement)
        replacements += 1

    # Report optimization status
    if replacements > 0:
        print(f"INVERTED RESIDUALS completed: Successfully applied to layers with {replacements} replacements")
    else:
        print("WARNING: INVERTED RESIDUALS not applied: No suitable layers found for replacement")

    return optimized_model

# --------------------------------------
# LOW-RANK FACTORIZATION MODULES
# --------------------------------------

# TODO: Implement this optimization method, if selected in your optimization strategy
def apply_lowrank_factorization(
    model: nn.Module,
    min_params: int = 10_000,
    rank_ratio: float = 0.25
) -> nn.Module:
    """
    Apply low-rank factorization to large linear layers for parameter reduction.
    
    Args:
        model: Input model to optimize for clinical deployment
        min_params: Minimum parameter count to consider for factorization
        rank_ratio: Fraction of minimum dimension to use as factorization rank
    
    Returns:
        Model with low-rank factorized linear layers for reduced memory usage
        
    Note:
        Only factorizes layers with sufficient parameters to benefit from compression.
        Rank selection balances compression ratio with accuracy preservation - lower
        ranks provide more compression but may impact diagnostic performance.
        
    Example:
        >>> model = create_baseline_model()
        >>> compressed_model = apply_lowrank_factorization(
        ...     model, min_params=5000, rank_ratio=0.5
        ... )
        >>> # Large linear layers now use low-rank factorization
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    replacements = 0  # Track number of successful replacements

    print("Applying low-rank factorization optimization...")

    # TODO: Factorizes large linear layers into low-rank approximations.
    # HINT: Low-rank factorization decomposes a large weight matrix W into two smaller matrices U and V
    # such that W ≈ U @ V. This dramatically reduces parameters while maintaining representational capacity.
    # Remember that higher rank = better approximation but less compression
    #
    # See https://arikpoz.github.io/posts/2025-04-29-low-rank-factorization-in-pytorch-compressing-neural-networks-with-linear-algebra/ 
    # for explanation and code template, and consider how to initialize parameters with respect to the new rank.

    candidates: List[Tuple[str, nn.Linear]] = []
    for name, module in optimized_model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.numel() >= min_params:
            candidates.append((name, module))

    for name, linear in candidates:
        in_features = linear.in_features
        out_features = linear.out_features
        rank = max(1, int(min(in_features, out_features) * rank_ratio))

        # Only factorize when it actually reduces parameters.
        dense_params = in_features * out_features
        factorized_params = rank * (in_features + out_features)
        if factorized_params >= dense_params:
            continue

        replacement = LowRankLinear(in_features, out_features, rank, bias=linear.bias is not None)

        # Initialize the factors from a truncated SVD of the original weight so the
        # optimized layer starts as an accurate approximation of the trained layer.
        with torch.no_grad():
            weight = linear.weight.detach().float()  # [out, in]
            u, s, vh = torch.linalg.svd(weight, full_matrices=False)
            u_r = u[:, :rank]
            s_r = s[:rank]
            vh_r = vh[:rank, :]
            sqrt_s = torch.sqrt(s_r)
            # first: [rank, in], second: [out, rank]  =>  second @ first ≈ weight
            replacement.first.weight.copy_((sqrt_s.unsqueeze(1) * vh_r).to(linear.weight.dtype))
            replacement.second.weight.copy_((u_r * sqrt_s.unsqueeze(0)).to(linear.weight.dtype))
            if linear.bias is not None:
                replacement.second.bias.copy_(linear.bias.detach())

        _replace_module(optimized_model, name, replacement)
        replacements += 1

    # Report optimization status
    if replacements > 0:
        print(f"LOW RANK FACTORIZATION completed: Successfully applied to layers with {replacements} replacements")
    else:
        print("WARNING: LOW RANK FACTORIZATION not applied: No suitable layers found for replacement")

    return optimized_model

# --------------------------------------
# CHANNEL OPTIMIZATION FUNCTIONS
# --------------------------------------

# TODO: Implement this optimization method, if selected in your optimization strategy
def apply_channel_optimization(
    model: nn.Module,
    enable_channels_last: bool = True,
    enable_inplace_relu: bool = True
) -> nn.Module:
    """
    Apply channel-level optimizations for enhanced hardware efficiency.

    Implements memory layout and activation optimizations to improve hardware utilization
    and reduce memory bandwidth requirements.

    Args:
        model: Model to optimize for hardware efficiency
        enable_channels_last: E.g., you'd use NHWC memory layout for faster GPU convolutions
        enable_inplace_relu: Convert ReLU layers to in-place for memory savings
    
    Returns:
        Hardware-optimized model with improved memory efficiency
        
    Note:
        The 'channels last' memory format can significantly improve convolution performance on certain hardware 
        (e.g., modern GPUs with specialized cores) but requires input tensors to be converted...
        
    Example:
        >>> model = create_baseline_model()
        >>> optimized_model = apply_channel_optimization(model)
        >>> # Remember to convert inputs: input.to(memory_format=torch.channels_last)
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    
    print("Applying channel-level hardware optimizations...")
    
    # TODO: Applies channel-level optimizations such as memory format changes
    # and in-place ReLU conversions for better hardware efficiency.
    # HINT: See https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html for a tutorial on channels last organization,
    # and note how input needs to be handled for it. 
    # Also, consider ensuring activations are in place by reviewing https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948/2 
    # for more details.

    if enable_inplace_relu:
        # In-place ReLU/ReLU6 avoids allocating a new activation tensor, cutting
        # memory bandwidth with no effect on outputs.
        converted = 0
        for module in optimized_model.modules():
            if isinstance(module, (nn.ReLU, nn.ReLU6)) and not module.inplace:
                module.inplace = True
                converted += 1
        print(f"   Converted {converted} activation(s) to in-place")

    if enable_channels_last:
        # channels_last (NHWC) layout lets modern GPUs use faster convolution kernels.
        # Inputs must also be converted at inference time via
        # input.to(memory_format=torch.channels_last).
        optimized_model = optimized_model.to(memory_format=torch.channels_last)
        print("   Converted model to channels_last memory format")

    # Report optimization status
    print("CHANNEL OPTIMIZATION completed")

    return optimized_model

# --------------------------------------
# PARAMETER SHARING FUNCTIONS
# --------------------------------------

# TODO: Implement this optimization method, if selected in your optimization strategy
def apply_parameter_sharing(
    model: nn.Module,
    sharing_groups: Optional[List[List[str]]] = None,
    layer_types: Optional[List[Type[nn.Module]]] = None
) -> nn.Module:
    """
    Apply parameter sharing between layers to reduce memory and improve efficiency.

    Shares weight parameters between layers with identical shapes to reduce memory
    footprint and potentially improve generalization. 

    Args:
        model: Model to optimize through parameter sharing
        sharing_groups: Manual specification of layer groups to share parameters.
                       If None, automatically groups layers with identical weight shapes.
        layer_types: Types of layers to consider for parameter sharing 
                    (defaults to Conv2d for maximum impact)
    
    Returns:
        Memory-optimized model with parameter sharing applied
        
    Note:
        Parameter sharing can improve model generalization by enforcing weight
        consistency across similar layers. Most effective when applied to layers
        with identical computational roles and sufficient parameter count.
        
    Example:
        >>> model = create_baseline_model()
        >>> shared_model = apply_parameter_sharing(model)
        >>> # Layers with identical shapes now share parameters
    """    
    # Default to Conv2d layers (largest parameter count and memory footprint)
    if layer_types is None:
        layer_types = [nn.Conv2d]

    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    # Track number of sharing layers and shared parameters
    total_shared = 0
    total_parameters_shared = 0
    
    print("Applying parameter sharing optimization...")

    # TODO: Shares parameters between layers in specified groups to reduce memory and computation.
    # HINT: Parameter sharing involves assigning the same `nn.Parameter` instance to multiple layers
    #
    # See https://stackoverflow.com/questions/57929299/how-to-share-weights-between-modules-in-pytorch 
    # for some inspiration.

    # Resolve every module by name once for quick lookup.
    module_by_name = dict(optimized_model.named_modules())

    def _is_shareable(mod: nn.Module) -> bool:
        return any(isinstance(mod, t) for t in layer_types)

    # Build the groups of layers that will share a single parameter set.
    groups: List[List[nn.Module]] = []
    if sharing_groups is not None:
        # Honor caller-specified groupings (referenced by layer name).
        for group_names in sharing_groups:
            members = [module_by_name[n] for n in group_names if n in module_by_name]
            if len(members) > 1:
                groups.append(members)
    else:
        # Auto-group layers that have identical weight shape and conv geometry so
        # the tied parameter is valid for every member.
        shape_groups: Dict[Tuple, List[nn.Module]] = {}
        for module in optimized_model.modules():
            if not _is_shareable(module) or not hasattr(module, "weight"):
                continue
            key = (
                tuple(module.weight.shape),
                getattr(module, "stride", None),
                getattr(module, "padding", None),
                getattr(module, "dilation", None),
                getattr(module, "groups", None),
            )
            shape_groups.setdefault(key, []).append(module)
        groups = [members for members in shape_groups.values() if len(members) > 1]

    # Tie each group's later layers to the first member's parameters.
    for members in groups:
        base = members[0]
        for member in members[1:]:
            member.weight = base.weight
            if getattr(member, "bias", None) is not None and getattr(base, "bias", None) is not None:
                member.bias = base.bias
            total_shared += 1
            total_parameters_shared += base.weight.numel()
   
    # Report optimization status
    if total_shared > 0:
        print(f"PARAMETER SHARING completed - Successfully shared parameters for {total_shared} layers")
        print(f"   Total parameters shared: {total_parameters_shared:,}")
    else:
        print("WARNING: PARAMETER SHARING failed - No suitable layer groups found for optimization")
    
    return optimized_model