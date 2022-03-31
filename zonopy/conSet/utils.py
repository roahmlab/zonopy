"""
Utilities of computation for continuous set
Author: Yongseok Kwon
Reference:
"""
import torch

def delete_column(Mat,idx_del):
    '''
    Mat: <torch.Tensor>
    ,shape [M,N]
    idx_del: <torch.Tensor>, <torch.int64> OR <int>
    , shape [n]
    
    return <torch.Tensor>
    , shape [M,N-n]
    # NOTE: may want to assert dtype latter
    '''
    N = Mat.shape[1]
    n= idx_del.numel()
    if type(idx_del) == int:
        idx_del = torch.tensor([idx_del],dtype=int)
    assert len(Mat.shape) == 2 and len(idx_del.shape) == 1
    assert n==0 or max(idx_del) < N
    
    idx_del,_ = torch.sort(idx_del)
    Mat_new = torch.zeros(Mat.shape[0],N-n)
    j = 0
    k = 0
    for i in range(N):
        if j < n and i == idx_del[j]:
            j +=1
        else:        
            Mat_new[:,i-j] = Mat[:,i]
    return Mat_new
    '''
    if idx_del.numel() >= 1:
        Mat_rec = Mat[:,[i for i in range(N) if i != idx_del[0]]]
        idx_del_rec = idx_del[1:]
        Mat = delete_column(Mat_rec,idx_del_rec)

    return Mat
    '''
    '''
    idx_remain = torch.zeros(N,dtype=bool)
    idx_full = torch.arange(N)

    for i in range(N):
        idx_remain[i] = all(idx_full[i] != idx_del)
    
    return Mat[:,idx_remain]
    '''
# mat mul tools
def G_mul_c(G,c):
    G_c = G.permute(2,0,1) @ c
    return G_c.permute(1,0)
    
def G_mul_g(G,g,dim=None):
    if dim is None:
        dim = G.shape[0]
    G_g = G.permute(2,0,1) @ g
    return G_g.permute(1,0,2).reshape(dim,-1)

def C_mul_G(C,G,dims=None):
    if dims is None:
        dims = [C.shape[0],C.shape[1]]
    C_G = C.reshape(1,dims[0],dims[1])@G.permute(2,0,1)
    return C_G.permute(1,2,0)

def G_mul_C(G,C):
    G_C = G.permute(2,0,1) @ C
    return G_C.permute(1,2,0)

def G_mul_G(G1,G2,dims=None):
    if dims is None:
        dims = [G1.shape[0],G1.shape[1],G2.shape[1]]
    G_G = G1.permute(2,0,1).reshape(-1,1,dims[0],dims[1]) @ G2.permute(2,0,1)
    return G_G.permute(2,3,0,1).reshape(dims[0],dims[2],-1)

if __name__ == '__main__':
    M = torch.tensor([[1,2,3,4],[5,1,6,1]])
    i = torch.tensor([0,1])
    M_new = delete_column(M,i)
    print(M_new)
    #import pdb; pdb.set_trace()
    
    
