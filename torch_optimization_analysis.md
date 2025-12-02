# mflux Torch 依赖优化分析

## 📊 当前状况

### Torch 安装占用空间
- **完整 PyTorch (CPU+CUDA)**: 约 2-3 GB
- **CPU-only PyTorch**: 约 **200-600 MB**
- **Torch 精简版 (如果可行)**: 可能 < 100 MB

### mflux 中 torch 的实际使用

通过代码审计，mflux 仅使用了以下 **极简** torch 功能：

| 功能 | 使用场景 | 文件 |
|------|---------|------|
| `torch.load()` | 加载 .pt 权重文件 | `weight_handler_depth_pro.py:24` |
| `torch.Tensor` 类型检查 | 判断数据类型 | 多处 |
| 数据类型 (`bfloat16`, `float16`, `float32`) | 类型转换 | 多处权重处理 |
| `tensor.to()` | 数据类型转换 | 权重转换 |
| `tensor.numpy()` | 转换为 numpy | 权重导出到 MLX |
| `tensor.detach().cpu()` | 移到 CPU | 权重处理 |
| `torch.all()` | 布尔运算 | `lora_converter.py:202` |
| `torch.split()` / `torch.chunk()` | 张量分割 | `lora_converter.py:221,229` |
| `torch.tensor()` | 创建张量 | `qwen2vl_processor.py:157,159` |

**关键发现**：
- ❌ 不使用自动微分 (autograd)
- ❌ 不使用神经网络模块 (nn.Module)
- ❌ 不使用优化器
- ❌ 不使用 CUDA/GPU 功能
- ✅ 仅使用基础的张量操作和类型转换

---

## 💡 优化方案

### 🎯 方案 1: 使用 CPU-only Torch (推荐)

**优势**：
- ✅ 减少 **1.5-2 GB** 磁盘空间（无 CUDA）
- ✅ 完全兼容现有代码
- ✅ 实施简单，无需修改代码

**实施方式**：

```toml
# pyproject.toml
dependencies = [
    # 替换现有的 torch 依赖为 CPU-only 版本
    "torch>=2.3.1,<3.0; python_version<'3.13' and platform_machine=='arm64'",
    "torch>=2.8.0,<3.0; python_version>='3.13' and platform_machine=='arm64'",
]
```

**安装命令**：
```bash
# 使用 CPU-only 索引
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**磁盘占用**：~200-400 MB

---

### 🎯 方案 2: 条件依赖 + 延迟导入

将 torch 相关功能拆分为可选依赖：

```toml
[project]
dependencies = [
    # 移除 torch 作为核心依赖
    # ... 其他依赖
]

[project.optional-dependencies]
# 基础权重转换（大多数场景）
torch-lite = [
    "torch>=2.3.1,<3.0; python_version<'3.13'",
]

# VLM 模型支持（需要 transformers + torch）
vlm = [
    "torch>=2.3.1,<3.0",
    "transformers>=4.57,<5.0",
]

# 完整功能
full = [
    "mflux[torch-lite,vlm]",
]
```

**好处**：
- 用户可以根据需求安装：
  - `pip install mflux` - 不包含 torch，但大多数功能不可用
  - `pip install mflux[torch-lite]` - 包含 CPU-only torch
  - `pip install mflux[vlm]` - 包含 VLM 支持

---

### 🎯 方案 3: 自定义精简 Torch 封装 (激进)

创建一个极简的 torch 兼容层，仅实现 mflux 需要的功能：

```python
# src/mflux/compat/torch_minimal.py

"""
Minimal torch compatibility layer for weight conversion only.
Falls back to full torch if available.
"""

try:
    import torch as _torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    class TorchMinimal:
        """Minimal torch implementation using numpy + safetensors"""

        class bfloat16:
            pass

        class float16:
            pass

        class float32:
            pass

        class Tensor:
            def __init__(self, data):
                self.data = np.array(data)

            def numpy(self):
                return self.data

            def to(self, dtype):
                # 简化的类型转换
                if dtype == TorchMinimal.float16:
                    return TorchMinimal.Tensor(self.data.astype(np.float16))
                elif dtype == TorchMinimal.float32:
                    return TorchMinimal.Tensor(self.data.astype(np.float32))
                return self

            def cpu(self):
                return self

            def detach(self):
                return self

        @staticmethod
        def load(path, map_location=None):
            # 使用 safetensors 或 numpy 加载
            # 这需要确保权重文件是 safetensors 格式
            raise NotImplementedError("Use safetensors format instead of .pt files")

        @staticmethod
        def tensor(data):
            return TorchMinimal.Tensor(data)

        @staticmethod
        def all(tensor):
            return np.all(tensor.data)

        @staticmethod
        def split(tensor, sizes, dim=0):
            return [TorchMinimal.Tensor(t) for t in np.split(tensor.data, sizes, axis=dim)]

        @staticmethod
        def chunk(tensor, chunks, dim=0):
            return [TorchMinimal.Tensor(t) for t in np.array_split(tensor.data, chunks, axis=dim)]

    _torch = TorchMinimal()

# Export
torch = _torch
```

**优缺点**：
- ✅ 可以完全移除 torch 依赖（如果用户不需要 VLM）
- ✅ 磁盘占用接近 0
- ❌ 需要大量测试
- ❌ 维护成本高
- ❌ depth_pro 的 `torch.load()` 仍需完整 torch

---

### 🎯 方案 4: 重构权重加载流程（最彻底）

**目标**：完全移除 torch 依赖

**实施步骤**：

1. **统一使用 safetensors 格式**
   - 移除 `torch.load()` 的使用
   - 要求所有权重为 safetensors 格式
   - 对于 depth_pro，提供转换脚本

2. **用 numpy 替代 torch 基础操作**
   ```python
   # 之前
   if tensor.dtype == torch.bfloat16:
       tensor = tensor.to(torch.float16)
   arr = mx.array(tensor.numpy())

   # 之后
   if tensor.dtype == 'bfloat16':
       tensor = tensor.astype(np.float16)
   arr = mx.array(tensor)
   ```

3. **处理 transformers 依赖**
   - `fibo_vlm_weight_handler.py` 是唯一使用 transformers 的地方
   - 将其设为可选依赖
   - 或者提供预转换的权重

**代码修改示例**：

```python
# src/mflux/models/depth_pro/weights/weight_handler_depth_pro.py

# 之前
pt_weights = torch.load(model_path, map_location="cpu")

# 之后 - 使用 safetensors
from safetensors.numpy import load_file as numpy_load_file
weights = numpy_load_file(model_path)
```

**好处**：
- ✅ **完全移除 ~600MB torch 依赖**
- ✅ 更轻量的安装
- ✅ 更快的导入时间
- ❌ 需要重构多个文件
- ❌ 需要用户转换现有 .pt 权重文件

---

## 📋 推荐实施路线

### 阶段 1: 立即可行 (不修改代码)
1. 切换到 CPU-only torch
2. 更新安装文档

**预计节省**: 1.5-2 GB

### 阶段 2: 中期优化 (小幅修改)
1. 将 transformers 依赖设为可选
2. 实现延迟导入
3. 添加可选依赖组

**预计额外节省**: 取决于用户选择

### 阶段 3: 长期目标 (需要重构)
1. 统一使用 safetensors
2. 用 numpy 替代 torch 基础操作
3. 提供权重转换工具

**预计额外节省**: 完全移除 600MB torch

---

## 🧪 验证 torch 实际大小

```bash
# 检查已安装的 torch 大小
du -sh $(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")

# 比较不同安装方式
pip install torch  # 完整版
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only

# 检查 wheel 大小
pip download torch --no-deps --dest .
ls -lh torch*.whl
```

---

## 💭 建议

**对于 macOS 移植**，我建议采用 **方案 1 (CPU-only)** + **方案 2 (可选依赖)** 的组合：

1. ✅ 默认安装 CPU-only torch（节省空间）
2. ✅ 将 VLM 功能（fibo_vlm）设为可选依赖
3. ✅ 在文档中说明精简安装方式
4. 🔄 逐步重构为 safetensors-only（长期目标）

这样可以：
- 立即减少 **60-70%** 的磁盘占用
- 保持完全兼容性
- 为用户提供选择权
- 为未来完全移除 torch 铺路
