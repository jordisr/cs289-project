
"""
Identifies whether each residue is charged, polar, or hydrophobic.

"""
from Bio import SeqIO
from Bio.SeqUtils import seq1
import sys, re


def feature(chain):
    
    #initialize dict of dict
    samples=dict()
    
    #create aa_id feature for each residue
    for residue in chain:
        
        #get residue index
        index=residue.get_id()[1]
        
        #initialize feature dict
        samples[index]={"isCharged":0, "isPolar":0, "isHydrophobic":0}
        
        amino_acid=seq1(residue.get_resname())
        
        if amino_acid in ["D","E","H","K","R"]:
            samples[index]["isCharged"]=1
        elif amino_acid in ["Q","T","S","N","C","Y"]:
            samples[index]["isPolar"]=1    
        elif amino_acid in ["A","F","G","I","L","M","P","V","W"]:
            samples[index]["isHydrophobic"]=1
    
    return samples

def feature_names():
    return ["isCharged", "isPolar", "isHydrophobic"]
