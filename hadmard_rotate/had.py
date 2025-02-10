import torch

class had_util:
    @staticmethod
    def random_orthogonal_matrix(size, device):
        """
        Generate a random orthogonal matrix of the specified size.
        First, we generate a random matrix with entries from a standard distribution.
        Then, we use QR decomposition to obtain an orthogonal matrix.
        Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
        
        Args:
        size (int): The size of the matrix (size x size).
        
        Returns:
        torch.Tensor: An orthogonal matrix of the specified size.
        """
        torch.cuda.empty_cache()
        random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
        q, r = torch.linalg.qr(random_matrix)
        q *= torch.sign(torch.diag(r)).unsqueeze(0)
        return q

    @staticmethod
    def is_pow2(n):
        return (n & (n - 1) == 0) and (n > 0)

    @staticmethod
    def get_hadK(n, transpose=False):
        hadK, K = None, None
        if False:
            pass
        else:
            assert (had_util.is_pow2(n))
            K = 1
        return hadK, K

    @staticmethod
    def matmul_hadU(X, transpose=False):
        n = X.shape[-1]
        hadK, K = had_util.get_hadK(n, transpose)
        input = X.clone().view(-1, n, 1)
        output = input.clone()
        while input.shape[1] > K:
            input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
            output = output.view(input.shape)
            output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
            output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
            output = output.view(input.shape[0], input.shape[1], -1)
            (input, output) = (output, input)
        del output

        if K > 1:
            # Do not explicitly repeat - OOM
            # input = torch.bmm(
            #     hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
            # Use bcast instead
            input = hadK.view(1, K, K).to(input) @ input

        # return input.view(X.shape) / torch.tensor(n).sqrt()
        return input.view(X.shape)

    @staticmethod
    def random_hadamard_matrix(size, device):
        # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
        Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
        Q = Q * 2 - 1
        Q = torch.diag(Q)
        return had_util.matmul_hadU(Q).to(device)

    @staticmethod
    def get_orthogonal_matrix(size, mode, device='cuda'):
        if mode == 'random':
            return had_util.random_orthogonal_matrix(size, device)
        elif mode == 'hadamard':
            return had_util.random_hadamard_matrix(size, device)
        else:
            raise ValueError(f'Unknown mode {mode}')


GEMM_B=3
GEMM_M=2
GEMM_N=8
GEMM_K=4

As = torch.randn(GEMM_B, GEMM_M, GEMM_K, device='cuda')
Bs = torch.randn(GEMM_B, GEMM_K, GEMM_N, device='cuda')

# [k, r] (r = k)
Had = had_util.get_orthogonal_matrix(GEMM_K, 'hadamard').to(dtype=torch.float32)
print(Had)
print(Had @ Had.T)

# standartd batch gemm
O = torch.einsum('bmk,bkn->bmn', As, Bs)

# let's multiple this orthogonal matrix to both A/B
# we call it hadamard rotate
# note for Had matrix we use 'rk' or 'kr' to compute, the result is the same
As_H = torch.einsum('bmk,rk->bmr', As, Had)
Bs_H = torch.einsum('bkn,rk->brn', Bs, Had)

# then to matmul the rotated A/B
O_H = torch.einsum('bmk,bkn->bmn', As_H, Bs_H)

# below O/O_H should be the same
print(O)
print(O_H)
