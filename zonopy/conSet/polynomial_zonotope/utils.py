"""
Utilities for polynomial zonotope and matrix polynomial zonotope
Author: Yongseok Kwon
Reference: CORA
"""
import torch
from queue import PriorityQueue
from zonopy.conSet import DEFAULT_OPTS

def argsortrows(Mat):
    '''
    argument of sorted rows of a matrix
    Mat: <torch.tensor, dtype=int> matrix
    
    return,
    ind: <list>, index
    '''
    
    num_row = Mat.shape[0]
    List = Mat.tolist()
    Q = PriorityQueue()
    for i in range(num_row):
        row_list = List[i]
        row_list.append(i)
        Q.put(row_list)

    ind = [] 
    for i in range(num_row):
        ind.append(Q.get()[-1])



    # np.lexsort([Mat[:,i] for i in reversed(range(Mat.shape[1]))])
    # np.lexsort(tuple(b.T.flip(0)))
    # torch.unique(b.T,dim=-1,sorted=True,return_inverse=True)[1].argsort()+1
    '''
    
    
    '''

    return ind
    



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
    itype = ExpMat.dtype
    dtype = G.dtype
    device = G.device

    dim_G = len(G.shape)
    permute_order = [dim_G-1] + list(range(dim_G-1))
    reverse_order = list(range(1,dim_G))+[0]
    G = G.permute(permute_order)
    
    idxD = torch.any(abs(G)>eps,-1)
    for _ in range(dim_G-2):
        idxD = torch.any(idxD,-1)

    # skip if all non-zero OR G is non-empty 
    if not all(idxD) or G.numel() == 0:
        # if all generators are zero
        if all(~idxD):
            # NOTE: might be better to assign None
            Gnew = torch.tensor([],dtype=dtype,device=device).reshape(G.shape[1:]+(0,))
            ExpMatNew = torch.tensor([],dtype=itype,device=device).reshape(ExpMat.shape[:1]+(0,))
            '''
            ExpMatNew = torch.zeros(ExpMat.shape[0],1)
            Gnew = torch.zeros(G.shape[0],1)
            '''
            return ExpMatNew, Gnew
        else:
            # keep non-zero genertors
            G = G[idxD]
            ExpMat = ExpMat[:,idxD]

    # add hash value of the exponent vector to the exponent matrix
    temp = torch.arange(ExpMat.shape[0],dtype=itype,device=device).reshape(1,-1) + 1
    
    
    '''  
    rankMat = torch.hstack(((temp@ExpMat).T,ExpMat.T))
    # sort the exponents vectors according to the hash value
    ind = argsortrows(rankMat)
    '''
    rankMat = torch.vstack((temp@ExpMat,ExpMat))
    # sort the exponents vectors according to the hash value
    ind = torch.unique(rankMat,dim=-1,sorted=True,return_inverse=True)[1].argsort()
    
    ExpMatTemp = ExpMat[:,ind]
    Gtemp = G[ind]
    
    # vectorized 
    ExpMatNew, ind_red = torch.unique_consecutive(ExpMatTemp,dim=-1,return_inverse=True)
    # NOTE
    if ind_red.max()+1 == ind_red.numel():
        return ExpMatTemp,Gtemp.permute(reverse_order)

    n_rem = ind_red.max()+1
    ind = torch.arange(n_rem).reshape(-1,1) == ind_red 
    num_rep = ind.sum(1)

    Gtemp2 = Gtemp.repeat((n_rem,)+(1,)*(dim_G-1))[ind.reshape(-1)].cumsum(0)
    Gtemp2 = torch.cat((torch.zeros((1,)+Gtemp2.shape[1:],device=device),Gtemp2),0)
    
    num_rep2 = torch.hstack((torch.zeros(1,dtype=int,device=device),num_rep.cumsum(0)))
    Gnew = (Gtemp2[num_rep2[1:]] - Gtemp2[num_rep2[:-1]]).permute(reverse_order)


    '''
    # initialization
    counterNew = 0
    ExpMatNew = torch.zeros_like(ExpMat)
    Gnew = torch.zeros_like(G)
    
    # first entry
    ExpMatNew[:,counterNew] = ExpMatTemp[:,0]
    Gnew[counterNew] = Gtemp[0]

    # loop over all exponent vectors
    
    for counter in range(1,ExpMatTemp.shape[1]):
        if all(ExpMatNew[:,counterNew] == ExpMatTemp[:,counter]):
            Gnew[counterNew] += Gtemp[counter]
        else:
            counterNew += 1
            Gnew[counterNew] = Gtemp[counter]
            ExpMatNew[:,counterNew] = ExpMatTemp[:,counter]

    ExpMatNew = ExpMatNew[:,:counterNew+1]
    Gnew = Gnew[:counterNew+1].permute(reverse_order)
    '''
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
        itype = expMat1.dtype
        device = expMat1.device

        
        id =id1
        ind2 =torch.zeros_like(id2)

        Ind_rep = id2.reshape(-1,1) == id1
        ind = torch.any(Ind_rep,axis=1)
        non_ind = ~ind
        ind2[ind] = Ind_rep.nonzero()[:,1]
        ind2[non_ind] = torch.arange(non_ind.sum()) + len(id)
        id = torch.hstack((id,id2[non_ind]))

        '''
        # merge the two sets
        id = id1
        ind2 =torch.zeros_like(id2)
        for i in range(L2):
            ind = [j for j in range(len(id)) if id[j] == id2[i]]
            # if ind is empty
            if not ind:
                id2_i = id2[i].clone().detach()
                id = torch.hstack((id,id2_i))
                ind2[i] = len(id)-1
            else:
                ind2[i] = ind[0]
        '''
        # construct the new exponent matrices
        L = len(id)
        expMat1 = torch.vstack((expMat1,torch.zeros(L-L1,expMat1.shape[1],dtype=itype,device=device)))
        temp = torch.zeros(L,expMat2.shape[1],dtype=itype,device=device)
        temp[ind2,:] = expMat2
        expMat2 = temp

    return id, expMat1, expMat2

def pz_repr(pz):
    # TODO: number of show 
    # TODO: precision

    pz_mat = torch.hstack((pz.c.reshape(-1,1),pz.G,pz.Grest))
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
            
            if j == 0 or (j != pz_mat.shape[1]-1 and j == pz.G.shape[1]):
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
    
    expMat = torch.tensor([[1]])
    G = torch.tensor([[0],[0],[0]])

    expMat, G = removeRedundantExponents(expMat,G)
    print(G)
    print(expMat)