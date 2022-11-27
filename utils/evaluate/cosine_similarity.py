import torch
from torch import nn
from torch.functional import Tensor


def compute_cosine_similarity(vectors: tuple[Tensor], dimension=1) -> Tensor:
    return float(nn.CosineSimilarity(dim=dimension, eps=1e-8)(*vectors))


if __name__ == "__main__":
    vector_1 = torch.randn(1, 128)
    vector_2 = torch.randn(1, 128)
    print(compute_cosine_similarity((vector_1, vector_2)))
