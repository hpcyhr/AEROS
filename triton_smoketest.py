"""Verify Triton works on this V100 + CUDA + Triton install."""
import torch
import triton
import triton.language as tl

print(f'PyTorch:  {torch.__version__}')
print(f'Triton:   {triton.__version__}')
print(f'CUDA:     {torch.version.cuda}')
print(f'Device:   {torch.cuda.get_device_name(0)}')
print(f'Capability: {torch.cuda.get_device_capability(0)}')

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 1 << 20
x = torch.randn(n, device='cuda', dtype=torch.float32)
y = torch.randn(n, device='cuda', dtype=torch.float32)
out = torch.empty_like(x)

grid = (triton.cdiv(n, 1024),)
add_kernel[grid](x, y, out, n, BLOCK=1024)
torch.cuda.synchronize()

ref = x + y
err = (out - ref).abs().max().item()
print(f'\nKernel ran. max err = {err:.2e}')
print('PASS' if err < 1e-5 else 'FAIL')