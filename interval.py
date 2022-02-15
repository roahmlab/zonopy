import torch
import math
from torch import Tensor

class interval:
    def __init__(self, input):
        assert(isinstance(input, Tensor), "input is expected to be a pytorch tensor")
        assert(input.shape == (2,) or input.shape == (0,), "input is expected to in shape (2,) or (0,)")
        if input.shape == (2,):
            assert(input[0] <= input[1], "lower limit should be smaller than upper limit")
            self.inf = input[0].item()
            self.sup = input[1].item()
            self.is_empty = False
        else:
            self.inf = None
            self.sup = None
            self.is_empty = True
        self.interval_tensor = input.clone().to(input.dtype).to(input.device)
        self.dtype = input.dtype
        self.device = input.device

    def __add__(self, other):
        if self.is_empty and other.is_empty:
            return interval(Tensor([]).to(self.dtype).to(self.device))
        elif self.is_empty:
            return interval(other.interval_tensor)
        elif other.is_empty:
            return interval(self.interval_tensor)
        else:
            return interval(self.interval + other.interval)

    def __sub__(self, other):
        if self.is_empty:
            return interval(Tensor([]).to(self.dtype).to(self.device))
        elif other.is_empty:
            return interval(self.interval_tensor)
        else:
            return interval(Tensor([self.inf - other.sup, self.sup - other.inf]).to(self.dtype).to(self.device))

    def __mul__(self, other):
        if isinstance(other,interval):
            if self.is_empty or other.is_empty:
                return interval(Tensor([]).to(self.dtype).to(self.device))

            results = [self.inf * other.inf, self.inf * other.sup, self.sup * other.inf, self.sup * other.sup]
            return interval(Tensor([min(results), max(results)]).to(self.dtype).to(self.device))
        else:
            return interval(other * self.interval_tensor)

    __rmul__ = __mul__

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






















