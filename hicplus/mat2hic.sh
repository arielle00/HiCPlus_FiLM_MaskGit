dat=$1
chrom=../chrom_hg38.sizes

# Convert HicPlus output into Juicer-compatible 9-column pairs
awk '{for(i=0;i<$7;i++) print NR, $1, $2, $3, $4, "+", "+"}' "$dat" > "${dat}_tmp"


# Build .hic
java -Xmx40g -jar ../juicer_tools.jar pre -q 0 -d \
    -r 5000,10000,20000,25000,40000,50000,100000 \
    "${dat}_tmp" "${dat}.hic" "$chrom"
