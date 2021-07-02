import torch
from torch import nn

def soft_argmax(voxels,device='cpu'):
    """
    Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
    Return: 3D coordinates in shape (batch_size, channel, 3)
    """
    voxels = voxels.unsqueeze(4)
    assert voxels.dim()==5
    # alpha is here to make the largest element really big, so it
    # would become very close to 1 after softmax
    alpha = 1000.0 
    N,C,H,W,D = voxels.shape
    soft_max = nn.functional.softmax(voxels.view(N,C,-1)*alpha,dim=2)
    soft_max = soft_max.view(voxels.shape)
    indices_kernel = torch.arange(start=0,end=H*W*D).unsqueeze(0).to(device)
    indices_kernel = indices_kernel.view((H,W,D))
    conv = soft_max*indices_kernel
    indices = conv.sum(2).sum(2).sum(2)
    z = indices%D
    y = (indices/D).floor()%W
    x = (((indices/D).floor())/W).floor()%H
    x = x / H
    y = y / W
    coords = torch.stack([x,y],dim=2)
    return coords


if __name__=='__main__':
    A = torch.randn(1,1,2,4)
    print(A)
    print(soft_argmax(A))
    print(torch.argmax(A))
