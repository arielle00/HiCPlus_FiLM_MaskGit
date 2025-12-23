#!/usr/bin/env python3
"""
check_hic.py — quick sanity checker for Juicer .hic files (for HiCPlus)

Usage:
  python check_hic.py -i 4DNFI35AE3O3-6h.hic -c 19 -r 10000
  python check_hic.py -i my.hic -c chr19               # auto-detect 10kb if present
  python check_hic.py -i my.hic -c 19 -s 0 -e 2000000  # sample first 2 Mb

What it does:
  1) Prints all available resolutions in the .hic
  2) Prints all chromosome names (so you see if they’re '19' vs 'chr19')
  3) Samples a small window on your chromosome at the requested resolution
     and prints a few nonzero pixels to confirm data is present.

Exit codes:
  0 = OK
  2 = chromosome not found in file
"""

from __future__ import print_function
import sys
import argparse
import hicstraw

def choose_chrom(user_chrom, chrom_names):
    """
    Map user input ('19' or 'chr19') to actual name in the file.
    """
    if user_chrom in chrom_names:
        return user_chrom
    # try adding/removing 'chr' prefix
    if user_chrom.startswith("chr"):
        alt = user_chrom[3:]
    else:
        alt = "chr" + user_chrom
    if alt in chrom_names:
        return alt
    return None

def choose_resolution(target, available):
    """
    If target is in available, return it; else return the smallest available.
    """
    av = sorted(available)
    if target in av:
        return target, None
    return av[0], target

def main():
    p = argparse.ArgumentParser(description="Sanity-check a .hic for HiCPlus")
    p.add_argument("-i", "--input", required=True, help="Path to .hic file")
    p.add_argument("-c", "--chrom", required=True, help="Chromosome (e.g., 19 or chr19)")
    p.add_argument("-r", "--resolution", type=int, default=10000, help="Target bin size (default: 10000)")
    p.add_argument("-s", "--start", type=int, default=0, help="Start (bp) of sample window (default: 0)")
    p.add_argument("-e", "--end",   type=int, default=1_000_000, help="End (bp) of sample window (default: 1,000,000)")
    p.add_argument("-n", "--nshow", type=int, default=10, help="How many nonzero pixels to print (default: 10)")
    args = p.parse_args()

    # Open file
    f = hicstraw.HiCFile(args.input)

    # Resolutions
    resolutions = sorted(map(int, f.getResolutions()))
    print("Resolutions:", resolutions)

    # Chromosomes
    chrom_objs = f.getChromosomes()
    chrom_names = [c.name for c in chrom_objs]
    print("Chromosomes:", " ".join(chrom_names))

    chrom = choose_chrom(args.chrom, chrom_names)
    if chrom is None:
        print("[!] Chromosome '{}' not found (tried both with/without 'chr').".format(args.chrom))
        sys.exit(2)

    # Resolution choice
    res, fallback = choose_resolution(args.resolution, resolutions)
    if fallback is not None:
        print("[!] Requested {} bp not present; using smallest available: {} bp".format(fallback, res))

    # Choose normalization: prefer KR, fall back to VC, then NONE
    print("Sampling {}:{}-{} at {} bp (KR if available)".format(chrom, args.start, args.end, res))
    mzd = None
    for norm in ("KR", "VC", "NONE"):
        try:
            mzd = f.getMatrixZoomData(chrom, chrom, "observed", norm, "BP", res)
            # Touch once to validate
            _ = mzd.getNorm() if hasattr(mzd, "getNorm") else None
            print("Using normalization:", norm)
            break
        except Exception:
            mzd = None
    if mzd is None:
        print("[!] Could not open matrix at {} bp.".format(res))
        sys.exit(1)

    # Print a few nonzero pixels using the object API
    shown = 0
    for rec in mzd.getRecords(args.start, args.end, args.start, args.end):
        if rec.counts != 0:
            print("bin1={:>7} bin2={:>7} count={}".format(rec.binX, rec.binY, rec.counts))
            shown += 1
            if shown >= args.nshow:
                break

    if shown == 0:
        print("[!] No nonzero pixels found in this window (could be sparse). Try a different region/chrom or wider range.")
    else:
        print("[OK] Printed {} sample nonzero pixels.".format(shown))

if __name__ == "__main__":
    main()
