#!/bin/bash
################################################################################
# ANALYSIS FOR SEQUENCE CONSERVATION
################################################################################
# 1. Find homologs with PSI-BLAST on the SwissProt database
# 2. Build multiple sequence alignment with MAFFT
################################################################################

FASTA_DIR='./fasta/'
ALIGN_DIR='./alignment/'
BLAST_DB='/home/jordi/work/db/Blast/swissprot'
NUM_THREADS=7

for FILE in $( ls $FASTA_DIR/*.fasta )
do
	echo $FILE
	PDB_ID=$(basename $FILE .fasta)

	# find homologs
	psiblast -query $FASTA_DIR$PDB_ID.fasta -db $BLAST_DB -out $ALIGN_DIR$PDB_ID.psiblast.out -num_threads $NUM_THREADS -outfmt 7 -num_iterations 5
	grep -i ^$PDB_ID $ALIGN_DIR$PDB_ID.psiblast.out | cut -f2 > $ALIGN_DIR$PDB_ID.homologs.txt
	cp $FILE $ALIGN_DIR$PDB_ID.homologs.fasta #ensure query sequence is first
	blastdbcmd -db $BLAST_DB -entry_batch $ALIGN_DIR$PDB_ID.homologs.txt >> $ALIGN_DIR$PDB_ID.homologs.fasta

	# align sequences
	mafft --quiet --thread $NUM_THREADS $ALIGN_DIR$PDB_ID.homologs.fasta > $ALIGN_DIR$PDB_ID.alignment.fasta

done
