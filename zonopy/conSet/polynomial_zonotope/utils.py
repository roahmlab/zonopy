"""
Utilities for polynomial zonotope and matrix polynomial zonotope
Author: Yongseok Kwon
Reference: CORA
"""
import torch
from zonopy.conSet import DEFAULT_OPTS

def removeRedundantExponents(ExpMat,G,eps=0):
    '''
    add up all generators that belong to terms with identical exponents
    
    ExpMat: <torch.tensor> matrix containing the exponent vectors
    G: <torch.tensor> generator matrix
    
    return,
    ExpMat: <torch.tensor> modified exponent matrix
    G: <torch.tensor> modified generator matrix
    '''
    # True for non-zero generators
    # NOTE: need to fix or maybe G should be zero tensor for empty

    dim_G = len(G.shape)    
    idxD = torch.any(abs(G)>eps,-1)
    for _ in range(dim_G-2):
        idxD = torch.any(idxD,-1)

    # skip if all non-zero OR G is non-empty 
    if not all(idxD) or G.numel() == 0:
        # if all generators are zero
        if all(~idxD):
            Gnew = torch.tensor([]).reshape((0,)+G.shape[1:])
            ExpMatNew = torch.tensor([],dtype=ExpMat.dtype).reshape((0,)+ExpMat.shape[1:])
            return ExpMatNew, Gnew
        else:
            # keep non-zero genertors
            G = G[idxD]
            ExpMat = ExpMat[:,idxD]

    # add hash value of the exponent vector to the exponent matrix
    temp = torch.arange(ExpMat.shape[1]).reshape(-1,1) + 1
    rankMat = torch.hstack((ExpMat.to(dtype=torch.float)@temp.to(dtype=torch.float),ExpMat))
    # sort the exponents vectors according to the hash value
    ind = torch.unique(rankMat,dim=0,sorted=True,return_inverse=True)[1].argsort()
    
    ExpMatTemp = ExpMat[ind]
    Gtemp = G[ind]
    
    # vectorized 
    ExpMatNew, ind_red = torch.unique_consecutive(ExpMatTemp,dim=0,return_inverse=True)
    if ind_red.max()+1 == ind_red.numel():
        return ExpMatTemp,Gtemp

    n_rem = ind_red.max()+1
    ind = torch.arange(n_rem).reshape(-1,1) == ind_red 
    num_rep = ind.sum(1)

    Gtemp2 = Gtemp.repeat((n_rem,)+(1,)*(dim_G-1))[ind.reshape(-1)].cumsum(0)
    Gtemp2 = torch.cat((torch.zeros((1,)+Gtemp2.shape[1:]),Gtemp2),0)
    
    num_rep2 = torch.hstack((torch.zeros(1,dtype=torch.long),num_rep.cumsum(0)))
    Gnew = (Gtemp2[num_rep2[1:]] - Gtemp2[num_rep2[:-1]])
    return ExpMatNew, Gnew

def mergeExpMatrix(id1, id2, expMat1, expMat2):
    '''
    Merge the ID-vectors of two polyZonotope and adapt the  matrice accordingly
    id1: <>,
    id2:
    expMat1: <>
    expMat1: <>
    
    return,
    id: <>, merged ID-vector
    expMat1: <>, adapted exponent matric of the first polynomial zonotope
    expMat2: <>, adapted exponent matric of the second polynomial zonotope
    '''
    L1 = len(id1)
    L2 = len(id2)

    # ID vectors are identical
    if L1 == L2 and all(id1==id2):
        id = id1

    # ID vectors not identical -> MERGE
    else:        
        id =id1
        ind2 =torch.zeros_like(id2)

        Ind_rep = id2.reshape(-1,1) == id1
        ind = torch.any(Ind_rep,axis=1)
        non_ind = ~ind
        ind2[ind] = Ind_rep.nonzero()[:,1]
        ind2[non_ind] = torch.arange(non_ind.sum()) + len(id)
        id = torch.hstack((id,id2[non_ind]))
        # construct the new exponent matrices
        L = len(id)
        expMat1 = torch.hstack((expMat1,torch.zeros(len(expMat1),L-L1,dtype=expMat1.dtype)))
        temp = torch.zeros(len(expMat2),L,dtype = expMat2.dtype)
        temp[:,ind2] = expMat2
        expMat2 = temp

    return id, expMat1, expMat2

def pz_repr(pz):
    # TODO: number of show 
    # TODO: precision

    pz_mat = pz.Z.T
    pz_repr = 'polyZonotope('
    for i in range(pz.dimension):
        if i == 0 and pz.dimension>1:
            pz_repr += '[['
        elif i ==0:
            pz_repr += '['
        else:
            pz_repr += '              ['
        for j in range(pz_mat.shape[1]):
            pz_repr +=  ('{{:.{}f}}').format(4).format(pz_mat[i,j].item())
            
            if j == 0 or (j != pz_mat.shape[1]-1 and j == pz.G.shape[0]):
                pz_repr += ' | '
            elif j !=pz_mat.shape[1]-1:
                pz_repr +=', '
        if i != pz_mat.shape[0]-1:
            pz_repr +='],\n'
        elif pz_mat.shape[0]>1:
            pz_repr += ']]'
        else:
            pz_repr += ']'

    if pz.dtype != DEFAULT_OPTS.DTYPE:
        pz_repr += f', dtype={pz.dtype}'
 
    pz_repr += '\n       '
    
    expMat_str = str(pz.expMat.to(dtype=int,device='cpu'))

    del_dict = {'tensor':'ExpMat=','    ':'       ','(':'',')':''}
    for del_el in del_dict.keys():
        expMat_str = expMat_str.replace(del_el,del_dict[del_el])

    pz_repr += expMat_str
    if pz.itype != DEFAULT_OPTS.ITYPE:
        pz_repr += f', itype={pz.itype}'
    if pz.device != DEFAULT_OPTS.DEVICE:
        pz_repr += f', device={pz.device}'
    pz_repr += ')'
    return pz_repr
'''
def check_decimals(tensor):
    max_digits, min_digits = 1,1
    max 
    if len(tensor.shape) == 0:
        value = tensor.item()
        n_full = str(len(value))
        n_digits = str(len(round(value)))
        if n_full != n_digits:
            n_decimals = n_full-n_digits-1
        else:
            n_decimals = 0
        return n_digits, n_digits, n_decimals
    else:
        for i in range(tensor.shape[0]):
            n_digits,_,n_decimals=check_decimals(tensor[i])
    return n_digits
'''
if __name__ == '__main__':
    
    expMat = torch.tensor([[1],[2],[1]])
    G = torch.tensor([[2],[3],[4]])

    expMat, G = removeRedundantExponents(expMat,G)
    print(G)
    print(expMat)

    expMat1 = torch.tensor([[1,2],[2,3],[1,4]])
    id1 = torch.tensor([1,2])
    expMat2 = torch.tensor([[1,2],[2,3],[1,4]])
    id2 = torch.tensor([2,3])
    id, expMat1,expMat2 = mergeExpMatrix(id1,id2,expMat1,expMat2)    

    print(id,expMat1,expMat2)