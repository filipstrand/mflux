# 代码迁移示例

这个文档展示如何将现有代码迁移到使用可选的 torch 依赖。

## 示例 1: qwen_weight_handler.py

### 修改前

```python
import torch
from safetensors.torch import load_file as torch_load_file

class QwenWeightHandler:
    @staticmethod
    def _load_safetensors_shards(path: Path, loading_mode: str = "multi_glob"):
        # ...
        try:
            file_weights = mlx_load_file(str(file_path))
        except Exception:
            # If MLX can't load directly, try with torch and convert
            torch_weights = torch_load_file(str(file_path))
            file_weights = {}
            for name, tensor in torch_weights.items():
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                file_weights[name] = mx.array(tensor.numpy())
```

### 修改后

```python
from safetensors.torch import load_file as torch_load_file

from mflux.compat.torch_check import optional_import_torch, require_torch

class QwenWeightHandler:
    @staticmethod
    def _load_safetensors_shards(path: Path, loading_mode: str = "multi_glob"):
        # ...
        try:
            file_weights = mlx_load_file(str(file_path))
        except Exception:
            # If MLX can't load directly, try with torch and convert
            # Check torch availability with helpful error message
            require_torch("Qwen weight conversion (torch fallback)")
            torch = optional_import_torch()

            torch_weights = torch_load_file(str(file_path))
            file_weights = {}
            for name, tensor in torch_weights.items():
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                file_weights[name] = mx.array(tensor.numpy())
```

**改动说明**:
- ❌ 移除顶部的 `import torch`
- ✅ 添加 `from mflux.compat.torch_check import require_torch, optional_import_torch`
- ✅ 在使用前调用 `require_torch()` 检查依赖
- ✅ 使用 `optional_import_torch()` 动态导入

---

## 示例 2: fibo_vlm_weight_handler.py

### 修改前

```python
import torch
from transformers import Qwen3VLForConditionalGeneration

class FIBOVLMWeightHandler:
    @staticmethod
    def load_vlm_regular_weights(repo_id: str = "briaai/FIBO-vlm", ...):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        # ...
```

### 修改后

```python
from mflux.compat.torch_check import optional_import_torch, require_torch

class FIBOVLMWeightHandler:
    @staticmethod
    def load_vlm_regular_weights(repo_id: str = "briaai/FIBO-vlm", ...):
        # VLM models require torch
        require_torch("FIBO-VLM model loading")
        torch = optional_import_torch()

        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        # ...
```

**改动说明**:
- ❌ 移除顶部的 `import torch`
- ✅ 在函数开始时检查 torch 可用性
- ✅ 清晰的错误信息告诉用户需要 `mflux[vlm]`

---

## 示例 3: lora_converter.py

### 修改前

```python
import torch

class LoRAConverter:
    @staticmethod
    def load_weights(lora_path: str) -> dict:
        state_dict = LoRAConverter._load_pytorch_weights(lora_path)
        state_dict = LoRAConverter._convert_weights_to_diffusers(state_dict)
        state_dict = LoRAConverter._convert_to_mlx(state_dict)
        # ...

    @staticmethod
    def _convert_to_mlx(torch_dict: dict):
        mlx_dict = {}
        for key, value in torch_dict.items():
            if isinstance(value, torch.Tensor):
                mlx_dict[key] = mx.array(value.detach().cpu())
            else:
                mlx_dict[key] = value
        return mlx_dict
```

### 修改后

```python
from mflux.compat.torch_check import optional_import_torch, require_torch

class LoRAConverter:
    @staticmethod
    def load_weights(lora_path: str) -> dict:
        # LoRA conversion requires torch
        require_torch("LoRA weight conversion")

        state_dict = LoRAConverter._load_pytorch_weights(lora_path)
        state_dict = LoRAConverter._convert_weights_to_diffusers(state_dict)
        state_dict = LoRAConverter._convert_to_mlx(state_dict)
        # ...

    @staticmethod
    def _convert_to_mlx(torch_dict: dict):
        torch = optional_import_torch()

        mlx_dict = {}
        for key, value in torch_dict.items():
            if isinstance(value, torch.Tensor):
                mlx_dict[key] = mx.array(value.detach().cpu())
            else:
                mlx_dict[key] = value
        return mlx_dict
```

**改动说明**:
- ❌ 移除顶部的 `import torch`
- ✅ 在入口函数检查依赖
- ✅ 在需要使用 `torch.Tensor` 的地方动态导入

---

## 示例 4: qwen2vl_processor.py (条件导入)

### 修改前

```python
def apply_chat_template(self, ...):
    # ...
    if return_tensors == "pt":
        import torch

        if isinstance(formatted, dict):
            result = {}
            for key, value in formatted.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.numpy()
```

### 修改后

```python
from mflux.compat.torch_check import optional_import_torch, require_torch

def apply_chat_template(self, ...):
    # ...
    if return_tensors == "pt":
        require_torch("Qwen2VL processor with return_tensors='pt'")
        torch = optional_import_torch()

        if isinstance(formatted, dict):
            result = {}
            for key, value in formatted.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.numpy()
```

**改动说明**:
- ✅ 保持原有的条件导入逻辑
- ✅ 添加显式的依赖检查
- ✅ 只在实际使用时才要求安装

---

## 示例 5: depth_pro 权重处理

### 修改前

```python
import torch

class WeightHandlerDepthPro:
    @staticmethod
    def load_weights() -> "WeightHandlerDepthPro":
        model_path = WeightHandlerDepthPro._download_or_get_cached_weights()
        pt_weights = torch.load(model_path, map_location="cpu")
        weights = WeightHandlerDepthPro._to_mlx_weights(pt_weights)
        # ...
```

### 修改后

```python
from mflux.compat.torch_check import optional_import_torch, require_torch

class WeightHandlerDepthPro:
    @staticmethod
    def load_weights() -> "WeightHandlerDepthPro":
        require_torch("Depth Pro model loading")
        torch = optional_import_torch()

        model_path = WeightHandlerDepthPro._download_or_get_cached_weights()
        pt_weights = torch.load(model_path, map_location="cpu")
        weights = WeightHandlerDepthPro._to_mlx_weights(pt_weights)
        # ...
```

---

## 用户体验示例

### 场景 1: 用户没有安装 torch

```bash
$ pip install mflux  # 不包含 torch
$ mflux-generate-fibo --prompt "test"
```

**输出**:
```
======================================================================
❌ FIBO-VLM model loading requires PyTorch, but it's not installed.

To enable this feature, install PyTorch support:

  # For basic weight conversion (most models)
  pip install 'mflux[weights]'

  # For VLM models (FIBO-VLM, Qwen-VL)
  pip install 'mflux[vlm]'

  # For LoRA conversion
  pip install 'mflux[lora]'

  # For all features
  pip install 'mflux[all]'

Or install PyTorch directly:
  pip install torch
======================================================================
```

### 场景 2: 用户安装了 torch

```bash
$ pip install 'mflux[vlm]'
$ mflux-generate-fibo --prompt "test"
✅ 正常运行
```

---

## 测试策略

### 单元测试

```python
# tests/test_torch_compat.py

import pytest
from mflux.compat.torch_check import is_torch_available, require_torch, optional_import_torch


def test_torch_availability():
    """Test torch availability check."""
    available = is_torch_available()
    assert isinstance(available, bool)


def test_optional_import():
    """Test optional import returns torch or None."""
    torch = optional_import_torch()
    if is_torch_available():
        assert torch is not None
        assert hasattr(torch, 'Tensor')
    else:
        assert torch is None


def test_require_torch_with_torch():
    """Test require_torch doesn't raise when torch is available."""
    if is_torch_available():
        require_torch("test feature")  # Should not raise


def test_require_torch_without_torch():
    """Test require_torch raises helpful error when torch is not available."""
    if not is_torch_available():
        with pytest.raises(ImportError) as exc_info:
            require_torch("test feature")
        assert "test feature" in str(exc_info.value)
        assert "pip install" in str(exc_info.value)
```

### CI 配置

```yaml
# .github/workflows/test.yml

jobs:
  test-without-torch:
    name: Test without torch
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -e .  # 基础安装
      - run: pytest tests/ -k "not requires_torch"

  test-with-torch:
    name: Test with torch
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -e ".[all]"  # 完整安装
      - run: pytest tests/
```

---

## 迁移检查清单

- [ ] 创建 `src/mflux/compat/torch_check.py`
- [ ] 更新所有使用 torch 的文件:
  - [ ] `qwen_weight_handler.py`
  - [ ] `fibo_weight_handler.py`
  - [ ] `fibo_vlm_weight_handler.py`
  - [ ] `lora_converter.py`
  - [ ] `weight_handler_depth_pro.py`
  - [ ] `qwen_vision_language_processor.py`
  - [ ] `qwen2vl_processor.py`
- [ ] 更新 `pyproject.toml`
- [ ] 添加测试
- [ ] 更新文档
- [ ] 测试所有安装场景
- [ ] 发布说明

---

## 预期影响

### 兼容性
- ✅ **100% 向后兼容** (通过 `mflux[all]`)
- ✅ 现有用户不受影响
- ✅ CI/CD 只需添加 `[all]`

### 性能
- ✅ 导入时间可能略微增加 (动态导入)
- ✅ 运行时性能无影响

### 维护
- ➕ 需要维护兼容层代码
- ➕ 需要测试多种安装场景
- ➖ 代码更加模块化，职责更清晰
