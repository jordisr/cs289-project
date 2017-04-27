
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
        samples[index]={"isAla":0, "isArg":0, "isAsn":0, "isAsp":0, 
                    "isCys":0, "isGln":0, "isGlu":0, "isGly":0,
                    "isHis":0, "isIle":0, "isLeu":0, "isLys":0, 
                    "isMet":0, "isPhe":0, "isPro":0, "isSer":0, 
                    "isThr":0, "isTrp":0, "isTyr":0, "isVal":0}
    
        amino_acid=residue.get_resname()
        samples[index]["is"+str(amino_acid)]=1
    
    return samples

def feature_names():
    return ["isAla", "isArg", "isAsn", "isAsp", "isCys", 
            "isGln", "isGlu", "isGly", "isHis", "isIle", 
            "isLeu", "isLys", "isMet", "isPhe", "isPro", 
            "isSer", "isThr", "isTrp", "isTyr", "isVal"]
    
