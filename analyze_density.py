"""Print kill-switch verdict from aeros_density.npz."""
import argparse, numpy as np
p = argparse.ArgumentParser()
p.add_argument('--inp', default='aeros_density.npz')
args = p.parse_args()

d = np.load(args.inp)
all_d = np.concatenate([d[k].ravel() for k in d.files])
f05 = (all_d < 0.05).mean()
f10 = (all_d < 0.10).mean()
f30 = (all_d < 0.30).mean()
print(f'\nGlobal density distribution over all (layer, t, b) triples:')
print(f'  < 5%:  {f05*100:5.1f}%')
print(f'  < 10%: {f10*100:5.1f}%')
print(f'  < 30%: {f30*100:5.1f}%')
print(f'  mean:  {all_d.mean():.4f}')
print(f'  median:{np.median(all_d):.4f}')
print(f'\n=== Verdict ===')
if f10 >= 0.60:
    print('GREEN: ≥60% of layer-timesteps below 10% density. AEROS has clear headroom.')
    print('Next: Phase 1 — write toy event-driven Conv kernel, compare vs cuDNN at measured densities.')
elif f10 >= 0.30:
    print('YELLOW: 30-60% below 10% density. Density-adaptive design is mandatory; AEROS must')
    print('have a fast switch between event-list / bitset / dense modes. Proceed with caution.')
else:
    print('RED: <30% below 10% density. Most layer-timesteps are too dense for event-driven to win.')
    print('Either switch to a sparser dataset (try N-Caltech101) or kill AEROS direction.')