import straw

# Open the .hic file
hic = straw.HiCFile("4DNFI1UEG1HD.hic")

# Chromosome sizes are in the 'chromosomes' dict
for chrom, length in hic.chromosomes.items():
    print(chrom, length)