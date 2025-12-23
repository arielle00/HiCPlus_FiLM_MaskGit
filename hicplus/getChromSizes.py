import hicstraw

f = hicstraw.HiCFile("../4DNFIXB4O92R-0h.hic")
chromosomes = f.getChromosomes()

with open("chrom.sizes", "w") as out:
    for chrom in chromosomes:
        if chrom.name != "All":  # skip pseudo "All"
            out.write(f"{chrom.name}\t{chrom.length}\n")
