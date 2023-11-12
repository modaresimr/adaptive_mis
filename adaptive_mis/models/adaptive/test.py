import torch
def optimize_einsum2(A,B):
    N, M, L, H, W = A.shape
    _,C,_,_,_=B.shape
    A=A.permute(1,0,3,4,2).reshape(M,-1,L)
    B=B.permute(1,0,3,4,2).reshape(C,-1,L)
    print("A=",A.shape," B=",B.shape)
    bases_out = torch.einsum('mol, col-> cmo', A, B)
    bases_out=bases_out.reshape(C, M, N, H, W).permute(2,0,1,3,4)
    return bases_out

def optimize_einsum5(A,B):
    N, M, L, H, W = A.shape
    _,C,_,_,_=B.shape
    A=A.permute(1,0,3,4,2).reshape(M,-1,L)
    B=B.permute(1,0,3,4,2).reshape(C,-1,L)
    print("A=",A.shape," B=",B.shape)
    B_transposed = B.transpose(1, 2)  # B.shape becomes [3, 81, 501760]

    # Perform matmul, which includes sum over the last dimension
    bases_out = torch.matmul(A, B_transposed)
    # bases_out = torch.einsum('mol, col-> cmo', A, B)
    bases_out=bases_out.reshape(C, M, N, H, W).permute(2,0,1,3,4)
    return bases_out

def optimize_einsum4(A,B):
    N, M, L, H, W = A.shape
    _,C,_,_,_=B.shape
    A=A.permute(1,0,3,4,2)
    B=B.permute(1,0,3,4,2)

    bases_out = torch.einsum('mbhwl, cbhwl-> bcmhw', A, B)
    return bases_out



def optimize_einsum3(A, B):
    N, M, L, H, W = A.shape
    _, C, _, _, _ = B.shape

    # Reshape and permute A and B to align for multiplication across lhw and summation over l
    A_reshaped = A.reshape(N, M, L, -1)  # Shape: [N, M, L, H*W]
    B_reshaped = B.reshape(N, C, L, -1)  # Shape: [N, C, L, H*W]

    # Perform einsum with adjusted dimensions
    bases_out = torch.einsum('nmld, ncld -> ncmd', A_reshaped, B_reshaped)
    return bases_out.reshape(N, C, M, H, W)

def orig(A,B):
    return torch.einsum('bmlhw, bclhw-> bcmhw', G, H)
from auto_profiler import Profiler
G = torch.randn(10, 6,81, 224, 224)
H = torch.randn(10, 3,81, 224, 224)
with Profiler():
    orig_out=orig(G,H)
    for m in [optimize_einsum2, optimize_einsum3, optimize_einsum4]:
        # print(m.__name__)
        bases_out = m(G, H)
        print(m.__name__,bases_out.shape,torch.allclose(bases_out,orig_out,atol=.00001))
