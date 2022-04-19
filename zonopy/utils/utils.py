import torch
def compare_permuted_gen(G1, G2,eps = 1e-6):
    assert G1.shape == G2.shape

    dim_G = len(G1.shape)
    permute_order = [dim_G-1] + list(range(dim_G-1))
    reverse_order = list(range(1,dim_G))+[0]
    G1,G2 = G1.permute(permute_order), G2.permute(permute_order)
    n_gens,dims = G1.shape[0], list(G1.shape[1:])
    
    dim_mul = 1
    for d in dims:
        dim_mul *= d
    
    for i in range(n_gens):
        distance_gen = G1[i].reshape([1]+dims)-G2
        for _ in range(dim_G-1):
            distance_gen = torch.norm(distance_gen,dim=1)
        
        closest_idx = torch.argmin(distance_gen)
        if distance_gen[closest_idx] > dim_mul**(0.5)*eps:
            return False
        G2 = G2[[j for j in range(n_gens-i) if j != closest_idx]].reshape([-1]+dims)
    return True


def compare_permuted_dep_gen(expMat1, expMat2, G1, G2,eps = 1e-6):
    assert G1.shape == G2.shape and expMat1.shape == expMat2.shape
    dim_G = len(G1.shape)
    permute_order = [dim_G-1] + list(range(dim_G-1))
    reverse_order = list(range(1,dim_G))+[0]
    G1,G2 = G1.permute(permute_order), G2.permute(permute_order)
    n_gens,dims = G1.shape[0], G1.shape[1:]
    for i in range(n_gens):
        diff_expMat = expMat1[:,i].reshape(-1,1)-expMat2
        j = torch.sum(abs(diff_expMat),dim=0) == 0
        assert sum(j) == 1
        if torch.linalg.norm(G1[i]-G2[j]) > eps:
            return False
    return True

def sign_cs(order):
    if order%4 == 2 or order%4 == 3:
        return 1
    else:
        return -1

def sign_sn(order):
    if order%4 == 0 or order%4 == 3:
        return 1
    else:
        return -1

