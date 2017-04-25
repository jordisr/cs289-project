# -*- coding: utf-8 -*-
"""
Outputs three-letter amino acid identity based on FASTA file input.

"""

from Bio import SeqIO
from Bio.SeqUtils import seq3
import sys, re

def amino_acid_id(fasta):
    
    #read in FASTA sequence
    for record in SeqIO.parse(fasta, "fasta"):
        sequence=record.seq 
    
    #initialize dict of dict
    samples=dict()
    
    #create aa_id feature for each residue
    for i in range(len(sequence)):
        samples[i]={}
        samples[i]["aa_id"]=seq3(sequence[i])
    
    return samples
        

    