import torch
def dict_torch2np(dic):
    for key in dic.keys():
        if isinstance(dic[key],torch.Tensor):
            dic[key] = dic[key].numpy()
