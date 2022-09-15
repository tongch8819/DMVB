import torch
import torch.nn as nn

batch, sentence_length, embedding_dim = 2,3,4
# embedding = torch.randn(batch, sentence_length, embedding_dim)
embedding = torch.tensor([
    [[1,1,1,1], [2,2,2,2], [3,3,3,3]],
    [[1,1,1,1], [2,2,2,2], [3,3,3,3]],
    [[1,1,1,1], [2,2,2,2], [3,3,3,3]]
    ], dtype=torch.float32)

layer_norm = nn.LayerNorm(embedding_dim)
norm_a = layer_norm(embedding)

batch_norm = nn.BatchNorm1d(embedding_dim)
norm_b = batch_norm(embedding.permute(0,2,1))


print(norm_a)
print(norm_b.permute(0,2,1))
# print(torch.max(norm_a - norm_b.permute(0,2,1)))