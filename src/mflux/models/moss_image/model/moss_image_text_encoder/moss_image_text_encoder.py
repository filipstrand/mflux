from pathlib import Path

import mlx.core as mx
import torch
from transformers import AutoModel, AutoTokenizer


class MOSSImageTextEncoder:
    def __init__(self, model_path: str | Path):
        self.path = Path(model_path)
        text_path = self.path / "text_encoder" if (self.path / "text_encoder").exists() else self.path
        tok_path = self.path / "tokenizer" if (self.path / "tokenizer").exists() else text_path

        # 智能共享回退：若当前量化目录未单独存放 16GB text_encoder，回退到本地 BF16 基准或远端默认
        bf16_fallback = Path("/Users/vanch/models/mlx/MOSS-Image-SFT-mlx-bf16")
        if not (text_path / "config.json").exists():
            if (bf16_fallback / "text_encoder").exists():
                print(f"🔗 自动复用共享文本编码器: {bf16_fallback / text_encoder}")
                text_path = bf16_fallback / "text_encoder"
            else:
                # 若本地未缓存，可直接指定使用基准 HF 仓库的 text_encoder
                text_path = "vanch007/MOSS-Image-SFT-mlx-bf16"

        if not (tok_path / "tokenizer.json").exists():
            if (bf16_fallback / "tokenizer").exists():
                tok_path = bf16_fallback / "tokenizer"
            else:
                tok_path = text_path

        print(f"📦 Loading MOSS Text Encoder from: {text_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(tok_path))
        self.model = AutoModel.from_pretrained(
            str(text_path),
            subfolder="text_encoder" if isinstance(text_path, str) and not Path(text_path).exists() else None,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        print("✅ MOSS Text Encoder loaded successfully!")

    @torch.no_grad()
    def encode_prompt(self, prompt: str, max_sequence_length: int = 4096) -> mx.array:
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=max_sequence_length,
        ).to(self.device)

        outputs = self.model(**inputs, use_cache=False, return_dict=True)
        hidden_state = outputs.last_hidden_state[0]
        hidden_np = hidden_state.float().cpu().numpy()

        return mx.expand_dims(mx.array(hidden_np), axis=0)