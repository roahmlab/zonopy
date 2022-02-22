import torch
from queue import PriorityQueue

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

    return ind
    



def removeRedundantExponents(ExpMat,G):
    '''
    add up all generators that belong to terms with idential exponents
    
    ExpMat: <torch.tensor> matrix containing the exponent vectors
    G: <torch.tensor> generator matrix
    
    return,
    ExpMat: <torch.tensor> modified exponent matrix
    G: <torch.tensor> modified generator matrix
    '''
    # True for non-zero generators
    # NOTE: need to fix or maybe G should be zero tensor for empty
    dim_G = len(G.shape)
    permute_order = [dim_G-1] + list(range(dim_G-1))
    reverse_order = list(range(1,dim_G))+[0]
    G = G.permute(permute_order)
    
    idxD = G
    for _ in range(len(G.shape)-1):    
        idxD = torch.any(idxD,-1)
    
    ExpMat = ExpMat.to(dtype=int)

    # skip if all non-zero OR G is already empty 
    if not all(idxD) or G.numel() == 0:
        # if all generators are zero
        if all(~idxD):
            # NOTE: might be better to assign None
            
            ExpMatNew =  torch.tensor([],dtype=int).reshape(ExpMat.shape[0],0)
            Gnew = torch.tensor([]).reshape(list(G.shape[1:])+[0])
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
    temp = torch.arange(ExpMat.shape[0]).reshape(1,-1) + 1
    rankMat = torch.hstack(((temp@ExpMat).T,ExpMat.T))
    
    # sort the exponents vectors according to the hash value
    ind = argsortrows(rankMat)
    ExpMatTemp = ExpMat[:,ind]
    Gtemp = G[ind]
    
    # initialization
    counterNew = 0
    ExpMatNew = torch.zeros(ExpMat.shape,dtype=int)
    Gnew = torch.zeros(G.shape)
    
    # first entry
    ExpMatNew[:,counterNew] = ExpMat[:,0]
    Gnew[counterNew] = G[0]

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
    return ExpMatNew, Gnew

def mergeExpMatrix(id1, id2, expMat1, expMat2):
    '''
    Merge the ID-vectors of two polyZonotope and adapt the exponent matrice accordingly
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
        # merge the two sets
        id = id1
        ind2 =torch.zeros(id2.shape,dtype=int)
        for i in range(L2):
            ind = [j for j in range(len(id)) if id[j] == id2[i]]
            # if ind is empty
            if not ind:
                id = torch.hstack((id,torch.tensor(id2[i])))
                ind2[i] = len(id)
            else:
                ind2[i] = ind[0]

        # construct the new exponent matrices
        L = len(id)

        expMat1 = torch.vstack((expMat1,torch.zeros(L-L1,expMat1.shape[1],dtype=int)))
        temp = torch.zeros(L,expMat2.shape[1],dtype=int)
        temp[ind2,:] = expMat2
        expMat2 = temp

    return id, expMat1, expMat2

if __name__ == '__main__':
    ExpMat = torch.tensor([[1,2,1,1],[3,1,5,3],[6,7,1,6],[5,8,3,5],[2,4,5,2]])
    print(ExpMat)
    temp = torch.arange(ExpMat.shape[0]).reshape(1,-1) + 1
    rankMat = torch.hstack(((temp@ExpMat).T,ExpMat.T))


    print(temp)
    print(rankMat)