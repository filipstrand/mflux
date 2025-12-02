# 🎯 mflux Torch 依赖优化提案

## 📊 实际数据验证

### Torch 真实占用空间

**下载大小 (Wheel 文件)**:
- macOS ARM64: **~75 MB** (出乎意料的小！)
- 包含 CPU-only 运算库

**安装后磁盘占用**:
- Wheel 解压后: ~150-200 MB
- 加上依赖 (numpy, typing-extensions 等): ~250-300 MB
- 如果包含 CUDA 版本: 2-3 GB ⚠️

**关键发现**:
- ✅ macOS ARM64 的 torch 默认就是 **CPU-only** 版本
- ✅ 已经相对精简 (~300 MB 总占用)
- ✅ 不需要特殊的"精简版"

---

## 🔍 mflux 的 torch 使用分析总结

### 实际使用的功能 (非常有限)

```python
# 1. 加载权重文件
torch.load("model.pt", map_location="cpu")  # 仅 1 处使用

# 2. 基础张量操作
tensor.to(torch.float16)      # 类型转换
tensor.detach().cpu()         # 移到 CPU
tensor.numpy()                # 转 numpy
torch.Tensor 类型判断          # 类型检查

# 3. 简单数学运算
torch.all(tensor)             # 布尔运算
torch.split(tensor, sizes)    # 分割
torch.chunk(tensor, chunks)   # 分块

# 4. transformers 模型加载
Qwen3VLForConditionalGeneration.from_pretrained(
    dtype=torch.bfloat16  # 仅用于指定数据类型
)
```

**不使用的功能**:
- ❌ 神经网络层 (nn.Module, nn.Linear 等)
- ❌ 自动微分 (requires_grad, backward)
- ❌ 优化器 (Adam, SGD 等)
- ❌ GPU/CUDA 运算
- ❌ 分布式训练
- ❌ TorchScript/JIT
- ❌ 数据加载器 (DataLoader)

---

## 💡 优化方案对比

| 方案 | 节省空间 | 兼容性 | 实施难度 | 推荐度 |
|------|----------|--------|----------|--------|
| **方案 0: 保持现状** | 0 MB | 100% | - | ⭐⭐⭐ |
| **方案 1: 拆分可选依赖** | 0-300 MB* | 100% | 低 | ⭐⭐⭐⭐⭐ |
| **方案 2: 延迟导入 + 降级提示** | 0-300 MB* | 95% | 低 | ⭐⭐⭐⭐ |
| **方案 3: 用 numpy 替换** | ~300 MB | 80% | 高 | ⭐⭐ |
| **方案 4: 完全移除 torch** | ~300 MB | 60% | 很高 | ⭐ |

*取决于用户是否安装可选功能

---

## 🎯 推荐方案: 拆分可选依赖

### 核心理念
- **不牺牲功能**，只是让用户选择需要什么
- **最小化默认安装**，提供增量安装选项
- **向后兼容**，现有用户不受影响

### 实施细节

#### 1. 修改 `pyproject.toml`

```toml
[project]
name = "mflux"
dependencies = [
    "accelerate>=0.31.0",
    "huggingface-hub>=0.24.5,<1.0",
    "mlx>=0.27.0,<0.31.0",
    "numpy>=2.0.1,<3.0",
    "safetensors>=0.4.4,<1.0",
    # ... 其他核心依赖

    # ❌ 移除强制的 torch 依赖
    # "torch>=2.3.1,<3.0",
]

[project.optional-dependencies]
# 基础权重转换（支持大部分模型）
weights = [
    "torch>=2.3.1,<3.0; python_version<'3.13'",
    "torch>=2.8.0,<3.0; python_version>='3.13'",
]

# VLM 模型支持 (FIBO-VLM, Qwen-VL)
vlm = [
    "torch>=2.3.1,<3.0; python_version<'3.13'",
    "torch>=2.8.0,<3.0; python_version>='3.13'",
    "transformers>=4.57,<5.0",
]

# Depth Pro 模型支持
depth = [
    "torch>=2.3.1,<3.0; python_version<'3.13'",
    "torch>=2.8.0,<3.0; python_version>='3.13'",
]

# LoRA 权重转换
lora = [
    "torch>=2.3.1,<3.0; python_version<'3.13'",
    "torch>=2.8.0,<3.0; python_version>='3.13'",
]

# 完整功能（向后兼容）
all = [
    "mflux[weights,vlm,depth,lora]",
]

# 开发依赖
dev = [
    "mflux[all]",
    "pytest>=8.3.0,<9.0",
    # ...
]
```

#### 2. 添加运行时检查

创建 `src/mflux/compat/torch_check.py`:

```python
"""
Torch compatibility and optional dependency checking.
"""

_TORCH_AVAILABLE = None
_TORCH_ERROR = None


def is_torch_available() -> bool:
    """Check if torch is available."""
    global _TORCH_AVAILABLE, _TORCH_ERROR
    if _TORCH_AVAILABLE is not None:
        return _TORCH_AVAILABLE

    try:
        import torch
        _TORCH_AVAILABLE = True
        return True
    except ImportError as e:
        _TORCH_ERROR = e
        _TORCH_AVAILABLE = False
        return False


def require_torch(feature_name: str = "this feature"):
    """
    Raise a helpful error if torch is not available.

    Args:
        feature_name: Name of the feature requiring torch

    Raises:
        ImportError: With installation instructions
    """
    if not is_torch_available():
        raise ImportError(
            f"\n{'='*70}\n"
            f"❌ {feature_name} requires PyTorch, but it's not installed.\n\n"
            f"To install PyTorch support:\n"
            f"  pip install mflux[weights]      # Basic weight conversion\n"
            f"  pip install mflux[vlm]          # VLM models (FIBO-VLM, Qwen)\n"
            f"  pip install mflux[lora]         # LoRA conversion\n"
            f"  pip install mflux[all]          # All features\n"
            f"\nOr install torch directly:\n"
            f"  pip install torch\n"
            f"{'='*70}\n"
        ) from _TORCH_ERROR


def optional_import_torch():
    """
    Optionally import torch with graceful fallback.

    Returns:
        torch module or None
    """
    if is_torch_available():
        import torch
        return torch
    return None
```

#### 3. 修改权重处理文件

**示例: `qwen_weight_handler.py`**

```python
import mlx.core as mx
from safetensors.mlx import load_file as mlx_load_file
from safetensors.torch import load_file as torch_load_file

from mflux.compat.torch_check import require_torch, optional_import_torch

class QwenWeightHandler:
    @staticmethod
    def _load_safetensors_shards(path: Path, loading_mode: str = "multi_glob"):
        # ... existing code ...

        # 当需要 torch fallback 时检查
        try:
            file_weights = mlx_load_file(str(file_path))
        except Exception:
            # 需要 torch 作为后备
            require_torch("Qwen weight loading (torch fallback)")
            torch = optional_import_torch()

            torch_weights = torch_load_file(str(file_path))
            file_weights = {}
            for name, tensor in torch_weights.items():
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                file_weights[name] = mx.array(tensor.numpy())

        # ... rest of code ...
```

**示例: `fibo_vlm_weight_handler.py`**

```python
import mlx.core as mx

from mflux.compat.torch_check import require_torch, optional_import_torch

class FIBOVLMWeightHandler:
    @staticmethod
    def load_vlm_regular_weights(repo_id: str = "briaai/FIBO-vlm", ...):
        # 明确要求 torch
        require_torch("FIBO-VLM model loading")

        torch = optional_import_torch()
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        # ... rest of code ...
```

**示例: `lora_converter.py`**

```python
import mlx.core as mx
from safetensors import safe_open

from mflux.compat.torch_check import require_torch, optional_import_torch

class LoRAConverter:
    @staticmethod
    def load_weights(lora_path: str) -> dict:
        require_torch("LoRA weight conversion")

        torch = optional_import_torch()
        state_dict = LoRAConverter._load_pytorch_weights(lora_path)
        # ... rest of code ...
```

#### 4. 更新文档

**README.md 添加安装选项**:

```markdown
## 📦 Installation

### Basic Installation (MLX models only)
```bash
pip install mflux
```

### With Weight Conversion Support
```bash
# For most models (FLUX, FIBO, Qwen, etc.)
pip install mflux[weights]

# For VLM models (FIBO-VLM, Qwen-VL)
pip install mflux[vlm]

# For LoRA conversion
pip install mflux[lora]

# Full installation (all features)
pip install mflux[all]
```

### 💾 Disk Space Requirements

| Installation Type | Disk Space | Supported Features |
|-------------------|------------|-------------------|
| Basic (`mflux`) | ~200 MB | Pre-converted MLX models |
| With weights (`mflux[weights]`) | ~500 MB | Most weight conversions |
| VLM support (`mflux[vlm]`) | ~1.5 GB | Vision-language models |
| Full (`mflux[all]`) | ~1.5 GB | All features |
```

---

## 📈 预期收益

### 对用户的好处

1. **更快的安装** (基础安装)
   - 下载: 减少 ~75 MB
   - 安装时间: 减少 ~30%

2. **更小的 Docker 镜像**
   ```dockerfile
   # 基础镜像: 只用 MLX 推理
   RUN pip install mflux
   # 节省 ~300 MB
   ```

3. **更清晰的依赖**
   - 用户知道每个功能需要什么
   - 避免不必要的依赖

4. **向后兼容**
   - 现有用户可以继续使用 `pip install mflux[all]`
   - 新用户可以选择精简安装

### 对项目的好处

1. **更模块化的架构**
   - 清晰的功能边界
   - 更容易测试

2. **更容易移植**
   - 可以在不支持 torch 的平台运行基础功能
   - 为未来替换 torch 铺路

3. **更好的错误提示**
   - 用户知道缺少什么依赖
   - 清晰的安装指引

---

## 🚀 实施路线图

### Phase 1: 准备阶段 (不破坏现有功能)
- [ ] 创建 `torch_check.py` 兼容层
- [ ] 添加运行时检查到所有 torch 使用点
- [ ] 更新测试确保兼容性
- [ ] 在 CI 中测试可选依赖场景

### Phase 2: 发布过渡版本
- [ ] 更新 `pyproject.toml`，torch 仍在 dependencies 中
- [ ] 在文档中添加关于可选依赖的说明
- [ ] 发布说明中告知用户即将的变化

### Phase 3: 正式拆分 (Breaking Change)
- [ ] 将 torch 移到 optional-dependencies
- [ ] 更新所有文档和示例
- [ ] 发布主版本更新 (例如 0.13.0 → 0.14.0)

### Phase 4: 持续优化
- [ ] 监控用户反馈
- [ ] 考虑进一步优化 (numpy 替换等)
- [ ] 添加自动安装提示

---

## ⚠️ 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 用户不知道安装哪个版本 | 中 | 清晰的文档 + 友好的错误提示 |
| CI/CD 需要更新 | 低 | 使用 `[all]` 在 CI 中 |
| 向后不兼容 | 高 | 主版本更新 + 详细的迁移指南 |
| 增加维护复杂度 | 中 | 良好的测试覆盖 |

---

## 🎬 示例场景

### 场景 1: 只想用预转换的 MLX 模型

```bash
pip install mflux
mflux-generate --model username/flux-schnell-mlx --prompt "test"
# ✅ 工作正常，只用 ~200 MB
```

### 场景 2: 需要转换 HuggingFace 权重

```bash
pip install mflux[weights]
mflux-save --model black-forest-labs/FLUX.1-schnell
# ✅ 可以转换权重
```

### 场景 3: 使用 VLM 功能

```bash
pip install mflux[vlm]
mflux-generate-fibo --prompt "描述这张图片" --image photo.jpg
# ✅ 完整 VLM 功能
```

### 场景 4: 开发者 (需要所有功能)

```bash
pip install mflux[all]
# 或者
pip install -e ".[all]"
# ✅ 完整功能，向后兼容
```

---

## 📝 结论

虽然 torch 本身只占 ~300 MB (macOS ARM64)，但通过**可选依赖**的方式：

1. ✅ **给用户选择权** - 根据需求安装
2. ✅ **优化安装体验** - 基础安装更快
3. ✅ **提高代码质量** - 模块化、清晰的依赖关系
4. ✅ **向后兼容** - 不破坏现有工作流
5. ✅ **为未来铺路** - 可以逐步用其他方案替换

**建议**: 采用此方案，在下一个主版本更新 (如 v0.13.0) 中实施。
