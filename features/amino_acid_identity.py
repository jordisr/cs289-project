
"""
Outputs three-letter amino acid identity based on FASTA file input.

"""

from Bio import SeqIO
import sys, re

def feature(chain):
    
    #initialize dict of dict
    samples=dict()
    
    #create aa_id feature for each residue
    for residue in chain:
        
        index=residue.get_id()[1]
        
        #print(index)
        samples[index]={"isALA":0, "isARG":0, "isASN":0, "isASP":0, 
                    "isCYS":0, "isGLN":0, "isGLU":0, "isGLY":0,
                    "isHIS":0, "isILE":0, "isLEU":0, "isLYS":0, 
                    "isMET":0, "isPHE":0, "isPRO":0, "isSER":0, 
                    "isTHR":0, "isTRP":0, "isTYR":0, "isVAL":0}
    
        amino_acid=residue.get_resname()
        samples[index]["is"+str(amino_acid)]=1
    
    return samples

def feature_names():
    return ["isALA", "isARG", "isASN", "isASP", "isCYS", 
            "isGLN", "isGLU", "isGLY", "isHIS", "isILE", 
            "isLEU", "isLYS", "isMET", "isPHE", "isPRO", 
            "isSER", "isTHR", "isTRP", "isTYR", "isVAL"]
    
