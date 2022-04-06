import zonopy as zp
from queue import PriorityQueue
import torch
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
    
    idxD = G
    for _ in range(len(G.shape)-1):    
        idxD = torch.any(idxD,-1)

    # skip if all non-zero OR G is already empty 
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
    rankMat = torch.hstack(((temp@ExpMat).T,ExpMat.T))
    
    # sort the exponents vectors according to the hash value
    ind = argsortrows(rankMat)
    ExpMatTemp = ExpMat[:,ind]
    Gtemp = G[ind]
    
    # initialization
    counterNew = 0
    ExpMatNew = torch.zeros_like(ExpMat)
    Gnew = torch.zeros_like(G)
    
    # first entry
    ExpMatNew[:,counterNew] = ExpMatTemp[:,0]
    Gnew[counterNew] = Gtemp[0]

    # loop over all exponent vectors
    for counter in range(1,ExpMatTemp.shape[1]):
        import pdb; pdb.set_trace()

        if all(ExpMatNew[:,counterNew] == ExpMatTemp[:,counter]):
            Gnew[counterNew] += Gtemp[counter]
        else:
            counterNew += 1
            Gnew[counterNew] = Gtemp[counter]
            ExpMatNew[:,counterNew] = ExpMatTemp[:,counter]

    ExpMatNew = ExpMatNew[:,:counterNew+1]
    Gnew = Gnew[:counterNew+1].permute(reverse_order)
    return ExpMatNew, Gnew
c = torch.Tensor([ 4.1853, -1.0609])
G = torch.Tensor([[ 5.1260e-04,  4.3752e+00, -2.6803e+00, -2.7513e+00,  1.1857e+00,
          4.2866e+00,  4.8214e+00, -2.7805e+00, -3.8583e+00,  2.4807e+00,
         -3.6996e-01,  1.0081e+00,  1.0337e+00,  2.6602e+00,  1.5330e+00,
          6.4309e-01,  2.0640e+00,  2.0446e+00, -7.3344e-01, -2.2034e+00,
          5.0377e-01,  2.9093e+00,  3.0601e-01, -1.9468e+00, -2.2155e+00,
          1.7562e+00,  8.9513e-01, -4.4200e+00, -2.2837e+00,  2.8134e+00,
         -1.1373e+00],
        [-4.7014e+00, -1.6406e+00, -4.4025e+00, -4.6174e+00, -4.6619e+00,
          3.3081e+00, -1.0602e+00, -3.9202e+00,  2.3568e+00,  4.5362e+00,
          1.8258e+00,  3.8170e-01,  2.9129e+00, -2.8616e+00, -2.3320e+00,
         -2.1643e+00, -2.9778e+00,  4.9793e+00,  1.6532e+00,  3.0852e+00,
         -1.5742e+00, -3.1645e-01,  3.3601e+00, -1.4663e+00,  2.3570e+00,
          4.5717e+00,  1.2407e-01,  2.7794e+00, -7.4073e-01, -3.9379e+00,
          4.6175e+00]])
Grest = torch.Tensor([[-1.9119,  2.3286, -3.0362, -1.4332, -2.1285, -1.6258,  1.6839, -1.2078,
         -4.8995,  4.2057],
        [-4.4166, -3.3944, -2.1451,  3.1436, -3.5001,  4.3883, -2.7569,  3.1477,
         -1.8862, -4.0404]])
expMat = torch.tensor([[ 3,  0, 11, 13, 14,  2,  1,  2,  4,  3,  6, 11,  1,  0, 11, 14,  8,  6,
         10,  2,  1, 13, 12,  4,  8,  6,  2,  9,  4,  7, 12],
        [ 5, 12,  4,  6, 13,  2, 11, 10, 10,  3,  0, 13, 14,  7, 14, 10, 10,  6,
          0, 13, 13, 13,  9, 12,  8,  9,  1,  5,  7,  6, 12],
        [ 4,  7,  1, 12,  4,  8,  0,  8,  8,  5,  4,  6, 13,  5,  8,  4, 11,  8,
         12,  5, 14,  1, 12,  9,  9, 12, 13, 10,  7, 13, 10],
        [ 4, 14,  5,  7, 10, 11,  1,  6, 11,  1, 12, 13,  1,  0,  8,  9,  0,  6,
         11,  1,  6,  1,  2,  6,  1,  2,  2,  4, 10,  5,  4],
        [ 0,  5,  3,  2,  0, 13, 11,  2,  1,  7,  0,  0,  3, 14,  9, 11,  5,  8,
          9, 13,  7,  1, 14, 12, 14,  2,  9,  8, 12,  7,  3],
        [ 0,  2, 12,  5,  7,  1, 14,  3,  2,  2, 13,  7,  9, 14,  1,  4,  5, 10,
          2, 14,  9, 10,  0,  2,  7, 13,  4, 11, 14,  8,  9],
        [ 9,  1,  3,  1,  4,  1,  6, 13,  6,  8,  0, 13, 10,  8,  2,  5, 13, 14,
          6, 12, 13,  4,  2, 14,  8,  4,  7, 14,  9, 13, 14],
        [ 0,  6,  0,  5,  0,  7,  3,  0,  4,  4,  9,  3,  3,  3, 10,  0, 11,  5,
          9,  0,  0, 12,  3,  7,  9, 13, 12, 13, 14, 10,  9],
        [ 4,  1,  7,  1,  5,  8,  6,  4,  4, 10,  7,  0,  9,  2,  7,  9,  1,  8,
         14, 10, 12, 11, 14,  6,  9, 11, 13,  6,  4, 13, 11],
        [ 2,  2,  4,  7,  5,  0,  1, 10, 11,  8,  7,  9,  4,  8,  7, 10,  8,  1,
          0,  2,  3, 10, 13, 10,  9,  8, 14, 10, 10, 10, 14]],
       dtype=torch.int32)


pz = zp.polyZonotope(c,G,Grest,expMat)


#[2., 9.], [-0.2319, -0.4073]
#[6., 9., 7.], [ 0.5291, -0.6531, -0.8033]
pz1 = pz.slice_dep([6, 9, 7],[ 0.5291, -0.6531, -0.8033])
pz2 = pz.slice_dep([6],[0.5291])
import pdb; pdb.set_trace()
pz22 = pz2.slice_dep([7],[-0.8033])
pz222 = pz22.slice_dep([9],[-0.6531])
import pdb; pdb.set_trace()





expMat, G= removeRedundantExponents(torch.hstack((pz1.expMat,pz2.expMat)),torch.hstack((pz1.G,-pz2.G)))









