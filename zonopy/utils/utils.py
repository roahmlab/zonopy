import torch
def compare_permuted_gen(G1, G2,eps = 1e-6):
    assert G1.shape == G2.shape

    dim_G = len(G1.shape)
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
    n_gens = G1.shape[0]
    for i in range(n_gens):
        diff_expMat = expMat1[:,i].reshape(-1,1)-expMat2
        j = torch.sum(abs(diff_expMat),dim=1) == 0
        assert sum(j) == 1
        try:
            if torch.linalg.norm(G1[i]-G2[j]) > eps:
                return False
        except:
            import pdb;pdb.set_trace()
    return True


