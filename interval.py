import torch
import math
import numbers
from torch import Tensor

class interval:
    def __init__(self, inf, sup):
        assert isinstance(inf, Tensor) and isinstance(sup, Tensor), "input is expected to be a pytorch tensor"
        assert inf.shape == sup.shape, "inf and sup is expected to be of the same shape"
        assert torch.all(inf <= sup), "inf should be <= sup entry-wise"
        assert inf.dtype == sup.dtype, "inf and sup should have the same dtype"
        assert inf.device == sup.device, "inf and sup should be on the same device"
        self.inf = inf.clone()
        self.sup = sup.clone()
        self.dtype = inf.dtype
        self.device = inf.device
        self.shape  = self.inf.shape

    def __add__(self, other):
        if isinstance(other, interval):
            return interval(self.inf + other.inf, self.sup + other.sup)
        elif isinstance(other, Tensor) and other.numel() == 1:
            return interval(self.sup + other, other.interval + other)
        else:
            assert False, "such addition is not implemented yet"

    def __sub__(self, other):
        if isinstance(other, interval):
            return interval(self.inf - other.sup, self.sup - other.inf)
        elif isinstance(other, Tensor) and other.numel() == 1:
            return interval(self.sup - other, other.interval - other)
        else:
            assert False, "such substraction is not implemented yet"

    def __mul__(self, other):
        if isinstance(other,interval):
            results = [self.inf * other.inf, self.inf * other.sup, self.sup * other.inf, self.sup * other.sup]
            new_inf = Tensor([min(results)]).to(self.dtype).to(self.device)
            new_sup = Tensor([max(results)]).to(self.dtype).to(self.device)

            return interval(new_inf, new_sup)
        elif (isinstance(other, Tensor) and other.numel() == 1) or isinstance(other,numbers.Number):
            if other >= 0:
                return interval(other * self.inf, other * self.sup)
            else:
                return interval(other * self.sup, other * self.inf)
        else:
            assert False, "such multiplication is not implemented yet"

    __rmul__ = __mul__

    def __getitem__(self, pos):
        inf = self.inf[pos]
        sup = self.sup[pos]
        return interval(inf, sup)

    def __setitem__(self, pos, value):
        # set one interval
        if isinstance(value, interval):
            self.inf[pos] = value.inf
            self.sup[pos] = value.sup
        else:
            self.inf[pos] = value
            self.sup[pos] = value

    def __len__(self):
        return len(self.inf)

    def __str__(self):
        return f"Interval of shape {self.inf.shape}.\n Inf: {self.inf}.\n Sup: {self.sup}.\n"

    def dim(self):
        return self.inf.dim()

    '''


    def __truediv__(self, other):
        inverse = None
        if other.inf == other.sup == 0:
            return interval(Tensor([]).to(self.dtype).to(self.device))
        elif other.inf < 0 < other.sup:
            inverse = interval(Tensor([math.inf, math.inf]).to(self.dtype).to(self.device))
        elif other.inf < 0 and other.sup == 0:
            inverse = interval(Tensor([-math.inf, 1 / other.inf]).to(self.dtype).to(self.device))
        elif other.inf == 0 and other.sup > 0:
            inverse = interval(Tensor([1 / other.sup, math.inf]).to(self.dtype).to(self.device))
        else:
            inverse = interval(Tensor([1 / other.sup, 1 / other.inf]).to(self.dtype).to(self.device))

        return self.__mul__(inverse)

    def __and__(self, other):
        if self.is_empty or other.is_empty:
            return interval(Tensor([]).to(self.dtype).to(self.device))

        new_inf = max(self.inf, other.inf)
        new_sup = min(self.sup, other.sup)
        if new_inf <= new_sup:
            return interval(Tensor([new_inf, new_sup]).to(self.dtype).to(self.device))
        else:
            return interval(Tensor([]).to(self.dtype).to(self.device))

    def __or__(self, other):
        if self.is_empty:
            return interval(other.interval_tensor)
        elif other.is_empty:
            return interval(self.interval_tensor)

        if max(self.inf, other.inf) <= min(self.sup, other.sup):
            new_inf = min(self.inf, other.inf)
            new_sup = max(self.sup, other.sup)
            return interval(Tensor([new_inf, new_sup]).to(self.dtype).to(self.device))
        else:
            return interval(Tensor([math.nan, math.nan]).to(self.dtype).to(self.device))

    def exp(self):
        return interval(torch.exp(self.interval_tensor))

    def log(self):
        if self.inf >=0:
            return interval(torch.log(self.interval_tensor))
        else:
            print("error in log")

    def abs(self):
        if self.sup < 0:
            return interval(torch.abs(Tensor([self.sup, self.inf]).to(self.dtype).to(self.device)))
        elif x.inf > 0:
            return interval(self.interval_tensor)
        else:
            return interval(Tensor([0, torch.max(torch.abs(self.interval_tensor))]))

    def sin(self):
        y_inf = self.inf % (2 * math.pi) / (math.pi / 2)
        y_sup = se;f.sup % (2 * math.pi) / (math.pi / 2)

        if (self.sup - self.inf >= 2 * math.pi) or (0 <= y_inf < 1 and 0 <= y_sup < 1 and y_inf > y_sup) or 
            (0 <= y_inf < 1 and 3 <= y_sup < 4) or (1 <= y_inf < 3 and 1<= y_sup < 3 and y_inf > y_sup):
            return interval(Tensor([-1, 1]).to(self.dtype).to(self.device))
        elif (0 <= y_inf < 1 and 0 <= y_sup < 1 and y_inf <= y_sup) or (3 <= y_inf < 4 and 0 <= y_sup < 1) or 
            (3 <= y_inf < 4 and 3 <= y_sup < 4 and y_inf <= y_sup):
            return interval(torch.sin(self.interval_tensor))
        elif (0 <= y_inf < 1 and 1 <= y_sup < 3) or (3 <= y_inf < 4 and 1 <= y_sup < 3):
            return interval(Tensor([torch.min(torch.sin(self.interval_tensor)),1]).to(self.dtype).to(self.device))
        elif (1 <= y_inf < 3 and 0 <= y_sup < 1) or (1 <= y_inf < 3 and 3 <= y_sup < 4):
            return interval(Tensor([-1,torch.max(torch.sin(self.interval_tensor))]).to(self.dtype).to(self.device))
        elif 1 <= y_inf < 3 and 1 <= y_sup < 3 and y_inf <= y_sup:
            return interval(torch.sin(Tensor([self.y_sup, self.y_inf]).to(self.dtype).to(self.device)))

    def cos(self):
        y_inf = self.inf % (2 * math.pi) / (math.pi / 2)
        y_sup = se;f.sup % (2 * math.pi) / (math.pi / 2)

        if (self.sup - self.inf >= 2 * math.pi) or (0 <= y_inf < 2 and 0 <= y_sup < 2 and y_inf > y_sup) or 
            (2 <= y_inf < 4 and 2 <= y_sup < 4 and y_inf > y_sup):
            return interval(Tensor([-1, 1]).to(self.dtype).to(self.device))
        elif (2 <= y_inf < 4 and 2 <= y_sup < 4 and y_inf <= y_sup):
            return interval(torch.cos(self.interval_tensor))
        elif (2 <= y_inf < 4 and 0 <= y_sup < 2):
            return interval(Tensor([torch.min(torch.cos(self.interval_tensor)),1]).to(self.dtype).to(self.device))
        elif (0 <= y_inf < 2 and 2 <= y_sup < 4):
            return interval(Tensor([-1,torch.max(torch.sin(self.interval_tensor))]).to(self.dtype).to(self.device))
        elif 1 <= y_inf <= 3 and 1 <= y_sup <= 3 and y_inf <= y_sup:
            return interval(torch.cos(Tensor([self.y_sup, self.y_inf]).to(self.dtype).to(self.device)))
    '''

def matmul_interval(mat, intv):
    assert isinstance(mat, Tensor) or isinstance(mat, interval), "the matrix should be in the type of a torch tensor"
    assert isinstance(intv, interval), "the intv should be in the type of an interval"
    assert mat.dim() == 2, "the dimension of the matrix should be 2"
    assert intv.dim() == 2, "the dimenstion of the interval matrix should be 2"
    assert mat.shape[1] == intv.shape[0], "the dimension of mat and interval should match"

    I, K = mat.shape
    K, J = intv.shape

    new_inf = torch.zeros(I,J).to(intv.dtype).to(intv.device)
    new_sup = torch.zeros_like(new_inf)

    for i in range(I):
        for k in range(K):
            for j in range(J):
                new_intv = mat[i,k] * intv[k,j]
                new_inf[i,j] += new_intv.inf.item()
                new_sup[i,j] += new_intv.sup.item()

    '''
    # to acclerate the matrix case (perhaps)
    for i in range(I):
        negative_indices = mat[i] > 0
        inf_mat = intv.inf.clone()
        sup_mat = intv.sup.clone()
        inf_mat[:,negative_indices] = intv.sup[:,negative_indices]
        sup_mat[:,negative_indices] = intv.inf[:,negative_indices]

        for j in range(J):
            new_inf[i,j] = torch.dot(mat[i], inf_mat[j,:])
            new_sup[i,j] = torch.dot(mat[i], sup_mat[j,:])
    '''

    return interval(new_inf, new_sup)


def cross_interval(vec, intv):
    assert len(vec) == 3, "we are considering only 3d vec"
    new_inf = torch.zeros(3).to(intv.dtype).to(intv.device)
    new_sup = torch.zeros_like(new_inf)

    intvs = []
    intvs.append(vec[1] * intv[2] - vec[2] * intv[1])
    intvs.append(vec[2] * intv[0] - vec[0] * intv[2])
    intvs.append(vec[0] * intv[1] - vec[1] * intv[0])

    for i in range(3):
        new_inf[i] = intvs[i].inf
        new_sup[i] = intvs[i].sup

    return interval(new_inf, new_sup)


if __name__ == '__main__':
    print("testing the functionality...")
    print("--- class construction test... ---")
    inf = torch.Tensor([1])
    sup = torch.Tensor([2])
    intv = interval(inf, sup)
    print(f"interval is: {intv}")

    print("--- testing index... ---")
    inf1 = torch.Tensor([[0.9,1.9],[2.9,3.9]])
    sup1 = torch.Tensor([[1.1,2.1],[3.1,4.1]])
    intv1 = interval(inf1, sup1)
    print(f"intv[0,0] = {intv1[0,0]}")
    print(f"intv[0] = {intv1[0]}")


    print("--- basic operation test... ---")
    intv1 = interval(torch.Tensor([-4]), torch.Tensor([2]))
    intv2 = interval(torch.Tensor([-12]), torch.Tensor([0]))
    print(f"intv1 + intv2 = {intv1 + intv2}")
    print(f"intv1 - intv2 = {intv1 - intv2}")
    print(f"intv1 * intv2 = {intv1 * intv2}")

    print("--- cross test... ---")
    intv1 = interval(torch.Tensor([0.9,1.9,2.9]), torch.Tensor([1.1,2.1,3.1]))
    intv2 = interval(torch.Tensor([0.9,4.9,-0.1]), torch.Tensor([1.1,5.1,0.1]))
    print(f"intv1 x intv2 = {cross_interval(intv1, intv2)}")
    

    print("--- matrix interval test... ---")
    inf1 = torch.Tensor([[0.9,1.9],[2.9,3.9]])
    sup1 = torch.Tensor([[1.1,2.1],[3.1,4.1]])
    intv1 = interval(inf1, sup1)

    inf2 = torch.Tensor([[9.9,9.9],[1.9,1.9]])
    sup2 = torch.Tensor([[10.1,10.1],[2.1,2.1]])
    intv2 = interval(inf2, sup2)
    print(f"intv1 matmul intev2 = {matmul_interval(intv1, intv2)}")
























