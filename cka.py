"""
Nguyen, Thao, Maithra Raghu, and Simon Kornblith. "Do Wide and Deep Networks Learn the Same Things? 
    Uncovering How Neural Network Representations Vary with Width and Depth." International 
    Conference on Learning Representations.

https://github.com/numpee/CKA.pytorch by Dongwan Kim (Github: Numpee)
adapted by Vita Shaw
"""

import torch
    
def cka(K, L, num_batch):
    hsic_KL = 0
    hsic_KK = 0
    hsic_LL = 0
    for idx in range(num_batch):
        gram_K = gram(K[idx::num_batch])[None, :, :].cuda()
        gram_L = gram(L[idx::num_batch])[None, :, :].cuda()
        hsic_KL += hsic1(gram_K, gram_L)
        hsic_KK += hsic1(gram_K, gram_K)
        hsic_LL += hsic1(gram_L, gram_L)
    return hsic_KL / torch.sqrt(hsic_KK * hsic_LL)

def hsic1(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    '''
    Batched version of HSIC.
    :param K: Size = (B, N, N) where N is the number of examples and B is the group/batch size
    :param L: Size = (B, N, N) where N is the number of examples and B is the group/batch size
    :return: HSIC tensor, Size = (B)
    '''
    assert K.size() == L.size()
    assert K.dim() == 3
    K = K.clone()
    L = L.clone()
    n = K.size(1)

    # K, L --> K~, L~ by setting diagonals to zero
    K.diagonal(dim1=-1, dim2=-2).fill_(0)
    L.diagonal(dim1=-1, dim2=-2).fill_(0)

    KL = torch.bmm(K, L)
    trace_KL = KL.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)
    middle_term = K.sum((-1, -2), keepdim=True) * L.sum((-1, -2), keepdim=True)
    middle_term /= (n - 1) * (n - 2)
    right_term = KL.sum((-1, -2), keepdim=True)
    right_term *= 2 / (n - 2)
    main_term = trace_KL + middle_term - right_term
    hsic = main_term / (n ** 2 - 3 * n)
    return hsic.squeeze(-1).squeeze(-1)

def gram(x: torch.Tensor) -> torch.Tensor:
    return x.matmul(x.t())