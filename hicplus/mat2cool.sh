dat=$1 ##output from hicplus prediction
chrom=../chrom_hg38.sizes ##chrom size file, change to your own species. 

# cat $dat | tr ':' '\t'|tr '-' '\t' > ${dat}_tmp

###transfrom the matrix file to .cool file
cooler load -f bg2 ${chrom}:10000 ${dat} ${dat}.cool --input-copy-status duplex
# cooler load -f bg2 chrom_hg38.sizes:10000 test_19_20_0hmodel_4DNFI1UEG1HD_pred.out.txt test_19_20_0hmodel_4DNFI1UEG1HD_pred.10k.cool
## remove the intermediate tmp file. 
# rm ${dat}_tmp
