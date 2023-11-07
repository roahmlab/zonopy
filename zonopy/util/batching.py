import torch
import numpy as np
import zonopy as zp

def stack(bpzlist, dim=0):
    """Stack a list of polyZonotopes or batchPolyZonotopes along a given dimension.
    
    Args:
        bpzlist (list): List of polyZonotopes or batchPolyZonotopes to stack
        dim (int, optional): Dimension to stack along. Defaults to 0.
        
    Returns:
        batchPolyZonotope: Stacked batchPolyZonotope
    """
    assert len(bpzlist) > 0, "Expected at least 1 element input!"

    # Dispatch to specialized version for polyZonotopes if needed (these functions can be merged later)
    promotion = np.array([isinstance(pz, zp.polyZonotope) for pz in bpzlist])
    if np.all(promotion):
        return zp.batchPolyZonotope.from_pzlist(bpzlist)

    # Promote any polynomial zonotopes to bpz's as needed
    for idx, tf in enumerate(promotion):
        if not tf:
            batch_shape = bpzlist[idx].batch_shape
            break
    bpzlist = [expand(pz, batch_shape) if promote else pz for pz, promote in zip(bpzlist, promotion)]
    
    # Check type
    assert np.all([isinstance(bpz, zp.batchPolyZonotope) for bpz in bpzlist]), "Expected all elements to be of type batchPolyZonotope"
    # Validate dimensions match
    n_bpz = len(bpzlist)
    dimension = bpzlist[0].dimension
    dtype = bpzlist[0].dtype
    device = bpzlist[0].device
    batch_shape = bpzlist[0].batch_shape
    assert dim <= len(batch_shape), "Expected dim to be less than or equal to the number of batch dimensions!"
    assert [bpz.dimension for bpz in bpzlist].count(dimension) == n_bpz, "Expected all elements to have the same dimensions!"
    assert [bpz.batch_shape for bpz in bpzlist].count(batch_shape) == n_bpz, "Expected all elements to have the same batch shape!"

    # First loop to extract key parts
    all_ids = [None]*n_bpz
    dep_gens = [None]*n_bpz
    all_c = [None]*n_bpz
    n_grest = [None]*n_bpz
    for i, bpz in enumerate(bpzlist):
        all_ids[i] = bpz.id
        dep_gens[i] = bpz.n_dep_gens
        all_c[i] = bpz.c.unsqueeze(-2)
        n_grest[i] = bpz.n_indep_gens
    
    # Combine
    all_ids = np.unique(np.concatenate(all_ids, axis=None))
    all_dep_gens = np.sum(dep_gens)
    dep_gens_idxs = np.cumsum([0]+dep_gens)
    n_grest = np.max(n_grest)
    all_c = torch.stack(all_c)

    # Preallocate
    all_G = torch.zeros((n_bpz,) + batch_shape + (all_dep_gens, dimension), dtype=dtype, device=device)
    all_grest = torch.zeros((n_bpz,) + batch_shape + (n_grest, dimension), dtype=dtype, device=device)
    all_expMat = torch.zeros((all_dep_gens, len(all_ids)), dtype=torch.int64, device=device)
    last_expMat_idx = 0

    # expand remaining values
    for bpzid in range(n_bpz):
        # Expand ExpMat (replace any with nonzero to fix order bug!)
        matches = np.nonzero(np.expand_dims(bpzlist[bpzid].id,1) == all_ids)[1]
        end_idx = last_expMat_idx + bpzlist[bpzid].expMat.shape[0]
        all_expMat[last_expMat_idx:end_idx,matches] = bpzlist[bpzid].expMat
        last_expMat_idx = end_idx
    
        # expand out all G matrices
        all_G[bpzid,...,dep_gens_idxs[bpzid]:dep_gens_idxs[bpzid+1],:] = bpzlist[bpzid].G

        # Expand out all grest
        grest = bpzlist[bpzid].Grest
        all_grest[bpzid,...,:grest.shape[0],:] = grest
    
    # Combine, reduce, output.
    Z = torch.concat((all_c, all_G, all_grest), dim=-2)
    if dim != 0:
        Z = Z.transpose(0, dim)
    out = zp.batchPolyZonotope(Z, all_dep_gens, all_expMat, all_ids, copy_Z=False).compress(2)
    return out

def expand(pz, shape):
    new_Z = pz.Z.expand(*shape, *pz.Z.shape)
    return zp.batchPolyZonotope(new_Z, pz.n_dep_gens, pz.expMat, pz.id, copy_Z=False)