import torch
from zonopy import interval, zonotope, matZonotope, polyZonotope, matPolyZonotope
from zonopy.utils.utils import compare_permuted_gen, compare_permuted_dep_gen, sign_cs, sign_sn


def close(zono1,zono2,eps = 1e-6):
    assert isinstance(zono1, type(zono2)) 
    if isinstance(zono1, zonotope):
        assert zono1.dimension == zono2.dimension
        eps = zono1.dimension**(0.5)*eps
        zono1, zono2 = zono1.deleteZerosGenerators(eps), zono2.deleteZerosGenerators(eps)
        if zono1.n_generators != zono2.n_generators or torch.norm(zono1.center-zono2.center) > eps:
            return False
        return compare_permuted_gen(zono1.generators,zono2.generators,eps)
    elif isinstance(zono1, matZonotope):
        assert zono1.n_rows == zono2.n_rows and zono1.n_cols == zono2.n_cols
        eps = (zono1.n_rows*zono1.n_cols)**(0.5)*eps
        zono1, zono2 = zono1.deleteZerosGenerators(eps), zono2.deleteZerosGenerators(eps)
        if zono1.n_generators != zono2.n_generators or torch.norm(zono1.center-zono2.center) > eps:
            return False
        return compare_permuted_gen(zono1.generators,zono2.generators,eps)
    elif isinstance(zono1,polyZonotope):
        assert zono1.dimension == zono2.dimension
        eps = zono1.dimension**(0.5)*eps
        zono1, zono2 = zono1.deleteZerosGenerators(eps), zono2.deleteZerosGenerators(eps)
        if torch.any(torch.sort(zono1.id).values != torch.sort(zono2.id).values):
            return False
        if zono1.n_dep_gens != zono2.n_dep_gens or zono1.n_indep_gens != zono2.n_indep_gens or torch.norm(zono1.c-zono2.c) > eps:
            return False
        if not compare_permuted_gen(zono1.Grest,zono2.Grest,eps):
            return False
        return compare_permuted_dep_gen(zono1.expMat[torch.argsort(zono1.id)],zono2.expMat[torch.argsort(zono2.id)],zono1.G,zono2.G,eps)

    else:
        print('Other types are not implemented yet.')
    
def cross(zono1,zono2):
    '''
    
    '''
    if isinstance(zono2,torch.Tensor):
        assert len(zono2.shape) == 1 and zono2.shape[0] == 3
        if isinstance(zono1,torch.Tensor):
            assert len(zono1.shape) == 1 and zono1.shape[0] == 3
            return torch.cross(zono1,zono2)
        elif isinstance(zono1,polyZonotope):
            assert zono1.dimension ==3
            return cross(-zono2,zono1)

    elif type(zono2) == polyZonotope:
        assert zono2.dimension == 3
        if type(zono1) == torch.Tensor:
            assert len(zono1.shape) == 1 and zono1.shape[0] == 3
            zono1_skew_sym = torch.tensor([[0,-zono1[2],zono1[1]],[zono1[2],0,-zono1[1]],[-zono1[1],zono1[0],0]])
            dtype = zono2.dtype
            itype = zono2.itype
            device = zono2.device

        elif type(zono1) == polyZonotope:
            assert zono1.dimension ==3
            c = zono1.c
            G = zono1.G
            Grest = zono1.Grest

            dtype = zono1.dtype
            itype = zono1.itype
            device = zono1.device
            
            Z = torch.hstack((c.reshape(-1,1),G,Grest))
            Z_skew = torch.zeros(3,3,Z.shape[1])
            
            num_c_G = 1+zono1.G.shape[1]
            
            for j in range(Z.shape[1]):
                z = Z[:,j]
                Z_skew[:,:,j] = torch.tensor([[0,-z[2],z[1]],[z[2],0,-z[1]],[-z[1],z[0],0]])
            
            zono1_skew_sym = matPolyZonotope(Z_skew[:,:,0],Z_skew[:,:,1:num_c_G],Z_skew[:,:,num_c_G:],zono1.expMat,zono1.id)

        return zono1_skew_sym@zono2    
    

def dot(zono1,zono2):
    if isinstance(zono1,torch.Tensor):
        if isinstance(zono2,polyZonotope):
            assert len(zono1.shape) == 1 and zono1.shape[0] == zono2.dimension
            zono1 = zono1.to(dtype=zono2.dtype)

            c = (zono1@zono2.c).reshape(1)
            G = (zono1@zono2.G).reshape(1,-1)
            Grest = (zono1@zono2.Grest).reshape(1,-1)
            return polyZonotope(c,G,Grest,zono2.expMat,zono2.id,zono2.dtype,zono2.itype,zono2.device)



def sin(Set,order=6):
    if isinstance(Set,interval):
        if Set.numel() == 1:
            if Set.sup-Set.inf >= 2*torch.pi:
                res_inf, res_sup = [-1], [1]
            else:
                inf = (Set.inf% (2*torch.pi))[0]
                sup = (Set.sup% (2*torch.pi))[0]
                if inf <= torch.pi/2:
                    if sup < inf:
                        res_inf, res_sup = [-1], [1]
                    elif sup <= torch.pi/2:
                        res_inf, res_sup = [torch.sin(inf)], [torch.sin(sup)]
                    elif sup < 3/2*torch.pi:
                        res_inf, res_sup = [torch.min(torch.sin(inf),torch.sin(sup))], [1]
                    else:
                        res_inf, res_sup = [-1], [1]
                elif inf <= 3/2*torch.pi:
                    if sup <= torch.pi/2:
                        res_inf, res_sup = [-1], [torch.max(torch.sin(inf),torch.sin(sup))]
                    elif sup < inf:
                        res_inf, res_sup = [-1], [1]
                    elif sup <= 3/2*torch.pi:
                        res_inf, res_sup = [torch.sin(sup)], [torch.sin(inf)]
                    else:
                        res_inf, res_sup =  [-1], [torch.max(torch.sin(inf),torch.sin(sup))]
                else: # inf in [pi, 2*pi]
                    if sup <= torch.pi/2:
                        res_inf, res_sup = [torch.sin(inf)], [torch.sin(sup)]
                    elif sup <= 3*torch.pi/2:
                        res_inf, res_sup = [torch.min(torch.sin(inf),torch.sin(sup))], [1]
                    elif sup < inf:
                        res_inf, res_sup = [-1], [1]
                    else:
                        res_inf, res_sup = [torch.sin(inf)], [torch.sin(sup)]
        else:
            res_inf, res_sup = torch.zeros_like(Set.inf), torch.zeros_like(Set.inf)

            inf = Set.inf%(2*torch.pi)
            sup = Set.sup%(2*torch.pi)
            
            a1 = Set.sup-Set.inf>= 2*torch.pi
            a2 = ~a1
            b1 = inf<=torch.pi/2
            b2 = ~b1
            B = inf<=3/2*torch.pi
            b3 = b2*B
            b4 = (~B)*(inf<=2*torch.pi)
            c1 = sup<=torch.pi/2
            c4 = ~c1
            c3 = sup>3/2*torch.pi
            c2 = c4*(~c3)            
            d1 = sup<inf 
            d2 = ~d1

            ind1 = tuple(torch.nonzero(a1).T)
            res_inf[ind1] = -1
            res_sup[ind1] = 1

            ind2 = tuple(torch.nonzero(a2*b1*d1).T)
            res_inf[ind2] = -1
            res_sup[ind2] = 1

            ind3 = tuple(torch.nonzero(a2*b1*c1*d2).T)
            res_inf[ind3] = torch.sin(inf[ind3])
            res_sup[ind3] = torch.sin(sup[ind3])
            
            ind4 = tuple(torch.nonzero(a2*b1*c2).T)
            res_inf[ind4] = torch.min(torch.sin(inf[ind4]),torch.sin(sup[ind4]))
            res_sup[ind4] = 1

            ind5 = tuple(torch.nonzero(a2*b1*c3).T)
            res_inf[ind5] = -1
            res_sup[ind5] = 1

            ind6 = tuple(torch.nonzero(a2*b3*c4*d1).T)
            res_inf[ind6] = -1
            res_sup[ind6] = 1

            ind7 = tuple(torch.nonzero(a2*b3*c1).T)
            res_inf[ind7] = -1
            res_sup[ind7] = torch.max(torch.sin(inf[ind7]),torch.sin(sup[ind7]))

            ind8 = tuple(torch.nonzero(a2*b3*c2*d2).T)
            res_inf[ind8] = torch.sin(sup[ind8])
            res_sup[ind8] = torch.sin(inf[ind8])

            ind9 = tuple(torch.nonzero(a2*b3*c3*d2).T)
            res_inf[ind9] = -1
            res_sup[ind9] = torch.max(torch.sin(inf[ind9]),torch.sin(sup[ind9]))

            ind10 = tuple(torch.nonzero(a2*b4*c3*d1).T)
            res_inf[ind10] = -1
            res_sup[ind10] = 1

            ind11 = tuple(torch.nonzero(a2*b4*c1).T)
            res_inf[ind11] = torch.sin(inf[ind11])
            res_sup[ind11] = torch.sin(sup[ind11])

            ind12 = tuple(torch.nonzero(a2*b4*c2).T)
            res_inf[ind12] = torch.min(torch.sin(inf[ind12]),torch.sin(sup[ind12]))
            res_sup[ind12] = 1

            ind13 = tuple(torch.nonzero(a2*b4*c3*d2).T)
            res_inf[ind13] = torch.sin(inf[ind13])
            res_sup[ind13] = torch.sin(sup[ind13])
        return interval(res_inf,res_sup,Set.dtype,Set.device)
    elif isinstance(Set,polyZonotope):
        assert Set.dimension == 1
        pz = torch.sin(Set.c)
        cos_c = torch.cos(Set.c)
        sin_c = torch.sin(Set.c)

        factor = 1
        T_factor = 1
        pz_neighbor = Set - Set.c

        for i in range(order):
            factor = factor * (i+1) 
            T_factor = T_factor * pz_neighbor
            if i%2 == 1:
                pz += sign_sn(i)*sin_c/factor*T_factor
            else:
                pz += sign_sn(i)*cos_c/factor*T_factor

        rem = pz_neighbor.to_interval()
        remPow = (T_factor*pz_neighbor).to_interval()

        #import pdb; pdb.set_trace()
        if order%2 == 1:
            J = sin(Set.c + interval([0],[1],Set.dtype,Set.device)*rem)
        else:
            J = cos(Set.c + interval([0],[1],Set.dtype,Set.device)*rem)
        if order%4==1 or order%4==2:
            J = -J


        remainder = 1/(factor*(order+1))*remPow*J
        pz.c += remainder.center()
        pz.Grest = torch.hstack((pz.Grest,remainder.rad().reshape(1,1)))
        pz.Grest = torch.sum(abs(pz.Grest),dim=1).reshape(1,1)
        return pz

def cos(Set,order = 6):
    if isinstance(Set,interval):
        if Set.numel() == 1:
            if Set.sup-Set.inf >= 2*torch.pi:
                res_inf, res_sup = [-1], [1]
            else:
                inf = (Set.inf% (2*torch.pi))[0]
                sup = (Set.sup% (2*torch.pi))[0]

                if inf <= torch.pi:
                    if sup < inf:
                        res_inf, res_sup = [-1], [1]
                    elif sup <= torch.pi:
                        res_inf, res_sup = [torch.cos(sup)], [torch.cos(inf)]
                    else:
                        res_inf, res_sup = [-1], [torch.max(torch.cos(inf),torch.cos(sup))]
                else: # inf in [pi, 2*pi]
                    if sup <= torch.pi:
                        res_inf, res_sup = [torch.min(torch.cos(inf),torch.cos(sup))], [1]
                    elif sup < inf:
                        res_inf, res_sup = [-1], [1]
                    else:
                        res_inf, res_sup = [torch.cos(inf)], [torch.cos(sup)]
        else:
            res_inf, res_sup = torch.zeros_like(Set.inf), torch.zeros_like(Set.inf)
            
            inf = Set.inf%(2*torch.pi)
            sup = Set.sup%(2*torch.pi)

            a1 = Set.sup-Set.inf>= 2*torch.pi
            a2 = ~a1
            b1 = inf<=torch.pi
            b2 = ~b1
            c1 = sup<=torch.pi
            c2 = ~c1
            d1 = sup<inf
            d2 = ~d1

            ind1 = tuple(torch.nonzero(a1).T)
            res_inf[ind1] = -1
            res_sup[ind1] = 1

            ind2 = tuple(torch.nonzero(a2*b1*d1).T)
            res_inf[ind2] = -1
            res_sup[ind2] = 1
            
            ind3 = tuple(torch.nonzero(a2*b1*c1*d2).T)
            res_inf[ind3] = torch.cos(sup[ind3])
            res_sup[ind3] = torch.cos(inf[ind3])

            ind4 = tuple(torch.nonzero(a2*b1*c2).T)
            res_inf[ind4] = -1    
            res_sup[ind4] = torch.max(torch.cos(inf[ind4]),torch.cos(sup[ind4]))

            ind5 = tuple(torch.nonzero(a2*b2*c2*d1).T)
            res_inf[ind5] = -1    
            res_sup[ind5] = 1

            ind6 = tuple(torch.nonzero(a2*b2*d2).T)
            res_inf[ind6] = torch.min(torch.cos(inf[ind6]),torch.cos(sup[ind6])) 
            res_sup[ind6] = 1

            ind7 = tuple(torch.nonzero(a2*b2*c2*d2).T)
            res_inf[ind7] = torch.cos(inf[ind7])
            res_sup[ind7] = torch.cos(sup[ind7])
        return interval(res_inf,res_sup,Set.dtype,Set.device)

    elif isinstance(Set,polyZonotope):
        assert Set.dimension == 1
        pz = torch.cos(Set.c)
        cos_c = torch.cos(Set.c)
        sin_c = torch.sin(Set.c)

        factor = 1
        T_factor = 1
        pz_neighbor = Set - Set.c

        for i in range(order):
            factor = factor * (i+1) 
            T_factor = T_factor * pz_neighbor
            if i%2 == 1:
                pz += sign_cs(i)*cos_c/factor*T_factor
            else:
                pz += sign_cs(i)*sin_c/factor*T_factor

        rem = pz_neighbor.to_interval()
        remPow = (T_factor*pz_neighbor).to_interval()

        #import pdb; pdb.set_trace()
        if order%2 == 0:
            J = sin(Set.c + interval([0],[1],Set.dtype,Set.device)*rem)
        else:
            J = cos(Set.c + interval([0],[1],Set.dtype,Set.device)*rem)
        if order%4==0 or order%4==1:
            J = -J
        remainder = 1/(factor*(order+1))*remPow*J
        pz.c += remainder.center()
        pz.Grest = torch.hstack((pz.Grest,remainder.rad().reshape(1,1)))
        pz.Grest = torch.sum(abs(pz.Grest),dim=1).reshape(1,1)
        return pz


