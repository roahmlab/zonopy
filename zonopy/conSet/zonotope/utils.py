import torch

def delete_column(Mat,idx_del):
    '''
    Mat: <torch.Tensor>
    ,shape [M,N]
    idx: <torch.Tensor>, <torch.int64> OR <int>
    -> <torch.Tensor>, <torch.int64>
    , shape [n]
    
    return <torch.Tensor>
    , shape [M,N-n]
    # NOTE: may want to assert dtype latter
    '''
    N = Mat.shape[1]
    if type(idx_del) == int:
        idx_del = torch.tensor([idx_del])

    assert len(Mat.shape) == 2
    assert len(idx_del.shape) == 1
    assert idx_del.numel()==0 or max(idx_del) < N

    idx_remain = torch.zeros(N,dtype=bool)
    idx_full = torch.arange(N)

    for i in range(N):
        idx_remain[i] = all(idx_full[i] != idx_del)

    return Mat[:,idx_remain]
