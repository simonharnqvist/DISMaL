VCF=$1
BED_CALLABLE=$2
GFF_ANNOTATION=$3
BLOCKSIZE=$4
LD_DECAY_KB=$5

# Make temp target dir
mkdir -p "temp"

# Subset GFF3 for introns
grep "intron" ${GFF_ANNOTATION} > temp/introns.gff3 &&

# Subset introns for correct length
awk -v '{ if ($4-$3 > ${BLOCKSIZE}) { print } }' introns.gff3 > introns_min${BLOCKSIZE}.gff3

# Remove introns that are too close (within LD)


# Intersect VCF with callable regions and introns
bedtools intersect -loj -a ${VCF} -b ${VCF_CALLABLE} temp/introns.gff3 > temp/callable_introns.vcf

