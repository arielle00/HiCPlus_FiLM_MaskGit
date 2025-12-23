import cooler, numpy as np, os, argparse, tqdm
from pathlib import Path

def dump_patches(cool_uri, out_dir, res_hr=10000, res_lr=40000,
                 patch=40, step=40, log=True):
    c_hr = cooler.Cooler(cool_uri.replace(str(res_lr), str(res_hr)))
    c_lr = cooler.Cooler(cool_uri)
    scale = res_lr // res_hr           # 4

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for chrom in c_hr.chromnames:
        Y = np.log1p(c_hr.matrix(balance=True).fetch(chrom))
        X = np.log1p(c_lr.matrix(balance=True).fetch(chrom))

        for i in range(0, Y.shape[0] - patch + 1, step):
            for j in range(0, Y.shape[1] - patch + 1, step):
                y = Y[i:i+patch, j:j+patch]
                x = X[i//scale:(i//scale)+patch, j//scale:(j//scale)+patch]
                fname = f"{chrom}_{i}_{j}.npz"
                np.savez_compressed(Path(out_dir)/fname,
                                    x=x.astype('float32'),
                                    y=y.astype('float32'))
        if log:
            print(f"{chrom}: {Y.shape} → {len(list(Path(out_dir).glob(f'{chrom}_*.npz')))} patches")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--uri", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hr", type=int, default=10000)
    p.add_argument("--lr", type=int, default=40000)
    args = p.parse_args()
    dump_patches(args.uri, args.out, args.hr, args.lr)
