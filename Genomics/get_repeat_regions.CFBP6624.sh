#!/bin/bash
run2Dir=/Users/stc/genomics_project/run2
strainID=CFBP6624
workingDir=/Users/stc/genomics_project/mapping
fastaGzDir=/Users/stc/genomics_project/fasta.gz/
fastaFile=GCF_005221425.1_ASM522142v1_genomic.fna

echo "${strainID} Start!!"

# remove kmers with low frequency, keep >=5
cd ${run2Dir}/${strainID}
mkdir tmp
cp kmers_*.fasta tmp/; cd tmp
rm kmers_2.fasta kmers_3.fasta
cat kmers_*.fasta > ${strainID}.ge5.kmers.fasta
grep -c '>' ${strainID}.ge5.kmers.fasta


# prepare working dir
mkdir -p ${workingDir}/${strainID}
cp ${fastaGzDir}/${fastaFile}.gz ${workingDir}/${strainID}/
cd ${workingDir}/${strainID}/
gzip -d ${fastaFile}.gz

cd ${workingDir}/${strainID}

# Index the reference sequences
## bwa index
bwa index -a is ${workingDir}/${strainID}/${fastaFile};

## index reference sequence in the FASTA format
samtools faidx ${workingDir}/${strainID}/${fastaFile};

# Map the kmers (reads) to the reference sequences
echo ${workingDir}/${strainID}/${fastaFile}
echo ${workingDir}/${strainID}/log

bwa mem -p -k 11 -T 13 ${workingDir}/${strainID}/${fastaFile} ${run2Dir}/${strainID}/tmp/${strainID}.ge5.kmers.fasta 2> ${workingDir}/${strainID}/log | samtools view -Su - | samtools sort -m 30000000000 -o ${workingDir}/${strainID}/kmers.bam ;
samtools index ${workingDir}/${strainID}/kmers.bam ;

# regions with reads covered
# get reads coverage for each base
samtools depth ${workingDir}/${strainID}/kmers.bam > ${workingDir}/${strainID}/${strainID}.depth ;

# get region list
awk '{FS=OFS="\t"} NR==1 {a=$2;b=$2;ID=$1;next} ($2 != b+1){print ID, a, b; a=$2;ID=$1} {b=$2} END{print ID, a, b}' ${workingDir}/${strainID}/${strainID}.depth > ${workingDir}/${strainID}/${strainID}_repeat_regions.list

awk '{FS=OFS="\t"} {$4=$3-$2+1; print $0}' ${workingDir}/${strainID}/${strainID}_repeat_regions.list | awk '$4>=80' > ${workingDir}/${strainID}/${strainID}_repeat_regions.f.list

python ${workingDir}/extract_region.1.py ${workingDir}/${strainID}/${fastaFile} ${workingDir}/${strainID}/${strainID}_repeat_regions.f.list ${workingDir}/${strainID}/${strainID}_repeat_regions.f.fasta

grep -c '>' ${workingDir}/${strainID}/${strainID}_repeat_regions.f.fasta
echo "out fasta: ${workingDir}/${strainID}/${strainID}_repeat_regions.f.fasta"
echo "out list: ${workingDir}/${strainID}/${strainID}_repeat_regions.f.list"
echo "${strainID} Done!!"