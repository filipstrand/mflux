from mlx import nn
import mlx.core as mx

from mflux.tokenizer.clip_tokenizer import TokenizerCLIP


class CLIPEmbeddings(nn.Module):

    def __init__(self, dims: int):
        super().__init__()
        self.position_embedding = nn.Embedding(num_embeddings=TokenizerCLIP.MAX_TOKEN_LENGTH, dims=dims)
        self.token_embedding = nn.Embedding(num_embeddings=49408, dims=dims)

    def forward(self, tokens: mx.array) -> mx.array:
        seq_length = tokens.shape[-1]
        position_ids = mx.arange(0, seq_length).reshape(1, seq_length)
        inputs_embeds = self.token_embedding(tokens)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings
        return embeddings
