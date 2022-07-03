import torch
def dict_torch2np(dic):
    for key in dic.keys():
        if isinstance(dic[key],torch.Tensor):
            dic[key] = dic[key].numpy().astype(float)
        elif isinstance(dic[key],torch.Tensor) and isinstance(dic[key][0],torch.Tensor):
            dic[key] = [el.numpy().astype(float) for el in dic[key]]

    return dic
