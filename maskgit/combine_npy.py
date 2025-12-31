import numpy as np
import argparse
from pathlib import Path


def combine_npy_files(data_dir, output_file=None):
    """
    Combine all .npy files in a directory into a single .npy file.
    
    Args:
        data_dir: Path to directory containing .npy files
        output_file: Output filename (if None, auto-generates based on directory name)
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise ValueError(f"Directory {data_dir} does not exist")
    
    # Find all .npy files in the directory
    npy_files = sorted(data_dir.glob("*.npy"))
    
    if len(npy_files) == 0:
        raise ValueError(f"No .npy files found in {data_dir}")
    
    print(f"[INFO] Found {len(npy_files)} .npy files in {data_dir}")
    print(f"[INFO] Files to combine:")
    for f in npy_files:
        print(f"  - {f.name}")
    
    # Load and combine all tiles
    tiles = []
    for npy_file in npy_files:
        print(f"[INFO] Loading {npy_file.name}...")
        data = np.load(npy_file)
        print(f"  Shape: {data.shape}")
        tiles.append(data)
    
    # Concatenate along the first axis (samples)
    print(f"\n[INFO] Concatenating {len(tiles)} arrays...")
    combined_tiles = np.concatenate(tiles, axis=0)
    print(f"[INFO] Combined shape: {combined_tiles.shape}")
    print(f"[INFO] Combined min/max: {combined_tiles.min():.4f} / {combined_tiles.max():.4f}")
    
    # Auto-generate output filename if not provided
    if output_file is None:
        if "lowres" in str(data_dir).lower():
            output_file = "hic_vqgan_train_lowres.npy"
        else:
            output_file = "hic_vqgan_train.npy"
    
    # Save combined file
    np.save(output_file, combined_tiles)
    print(f"\n[DONE] Saved combined tiles to {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine all .npy files in a directory into a single .npy file"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="chr_data_lowres",
        help="Directory containing .npy files (default: chr_data_lowres)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: auto-generated based on directory name)"
    )
    
    args = parser.parse_args()
    
    combine_npy_files(args.dir, args.output)