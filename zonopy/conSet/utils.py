import torch

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
    G = (torch.arange(24,dtype=float)+1).reshape(2,3,4)
    c = torch.arange(3,dtype=float)+1
    g = (torch.arange(15,dtype=float)+1).reshape(3,5)
    Gc = G_mul_c(G,c)
    Gg = G_mul_g(G,g)
    #import pdb; pdb.set_trace()
    
    
