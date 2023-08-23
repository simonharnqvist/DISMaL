# Blocking workflow for DISMaL
Keep separate from DISMaL to allow users to update the more quickly changeable bioinformatics preprocessing.

## Steps:
0. Map reads to common reference; produce VCF (not part of workflow) and filter for quality
1. Intersect BED file of callable regions with VCF
2. Make blocks of length k from VCF
3. Select two samples from each block
4. Phase samples with binomial approach if not already phased
5. Make FASTA of samples
6. Concatenate

## 1. Intersect VCF with BED (callable regions) and GFF (coding regions)
```
grep "intron" [GFF] # see SOFA ontology; subset for non-coding
bcftools intersect [BED] [VCF] [GFF]
```

## 2. Make blocks of length k
1. Make start indices using k and distance between blocks
2. For each block, if more than n missing, discard block
3. Else, if n > 0, add positions before and after to block
4. Output temporary VCF of block

## 3. Randomly select two individuals from VCF
Subset VCF for two randomly selected individuals for each block VCF

## 4. Perform binomial phasing on each VCF (If unphased)

## 5. Make FASTA of samples
1. Make each FASTA using a random haplotype:
```
cat reference.fa | consensus --haplotype [RANDOM] block[N].vcf.gz
```
2. Edit header to compliant format

## 6. Concatenate all FASTAs to single file

