import numpy as np

tiles = []
for c in ["chr1","chr2","chr3","chr4", "chr5","chr6","chr7","chr8","chr9","chr10","chr11", "chr12", "chr13","chr14","chr15","chr16","chr18","chr19","chr20","chr21"]:
    tiles.append(np.load(f"{c}_10kb_tiles.npy"))

tiles = np.concatenate(tiles, axis=0)
np.save("hic_vqgan_train.npy", tiles)