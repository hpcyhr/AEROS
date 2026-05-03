"""Phase 1.1e — Verify M1 vs SJ cupy backend (the fair, fast baseline)."""
import time, gc
import torch
from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18


@torch.no_grad()
def baseline_forward(net, x):
    functional.reset_net(net)
    return net(x)


@torch.no_grad()
def chunked_forward(net, x, K):
    T = x.shape[0]
    functional.reset_net(net)
    chunks = []
    for i in range(T // K):
        chunks.append(net(x[i*K:(i+1)*K]))
    return torch.cat(chunks, dim=0)


def measure(fn, *args, n_warmup=2, n_iters=5):
    torch.cuda.synchronize()
    torch.cuda.empty_cache(); gc.collect()
    try:
        for _ in range(n_warmup): _ = fn(*args)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(n_iters): _ = fn(*args)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()/1e9, (time.perf_counter()-t0)/n_iters*1000, 'ok'
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); gc.collect()
        return None, None, 'OOM'


# 5 representative configs
configs = [
    (32, 32, 128),    # small
    (64, 32, 128),    # medium
    (128, 32, 128),   # medium-large (this was our headline config 23)
    (128, 32, 224),   # config 24 (the OOM-saver)
    (256, 32, 128),   # large (config 32)
]

device = torch.device('cuda:0')

for backend in ['torch', 'cupy']:
    print(f'\n=== Backend: {backend} ===')
    for T, B, H in configs:
        net = spiking_resnet18(spiking_neuron=neuron.LIFNode,
                               surrogate_function=surrogate.ATan(),
                               detach_reset=True).to(device)
        net.eval()
        functional.set_step_mode(net, step_mode='m')
        try:
            functional.set_backend(net, backend=backend)
        except Exception as e:
            print(f'  set_backend({backend}) failed: {e}')
            continue

        x = (torch.rand(T, B, 3, H, H, device=device) > 0.7).float()

        mem_b, wall_b, st_b = measure(baseline_forward, net, x)
        # Try a few K values
        Ks = [T, T//2, T//4, T//8, T//16, T//32, 1]
        Ks = sorted(set(K for K in Ks if K >= 1 and T % K == 0), reverse=True)

        if st_b == 'ok':
            print(f'  T={T} B={B} H={H}: baseline {wall_b:.0f}ms {mem_b:.2f}GB')
        else:
            print(f'  T={T} B={B} H={H}: baseline OOM')

        for K in Ks:
            mem_K, wall_K, st_K = measure(chunked_forward, net, x, K)
            if st_K == 'ok':
                if st_b == 'ok':
                    sd = wall_K / wall_b
                    sv = mem_b / mem_K
                    print(f'    K={K:>3d}: {wall_K:>6.0f}ms {mem_K:>5.2f}GB '
                          f'slowdown={sd:.2f}x savings={sv:.2f}x')
                else:
                    print(f'    K={K:>3d}: {wall_K:>6.0f}ms {mem_K:>5.2f}GB (base OOM)')
            else:
                print(f'    K={K:>3d}: {st_K}')
        del net, x
        torch.cuda.empty_cache(); gc.collect()
