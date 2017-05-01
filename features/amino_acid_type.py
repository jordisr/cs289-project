
"""
Identifies whether each residue is charged, polar, or hydrophobic.

"""
from Bio import SeqIO
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

        amino_acid=residue.get_resname().title()

        if amino_acid in ["Asp","Glu","His","Lys","Arg", "Pyl"]:
            samples[index]["isCharged"]=1
        elif amino_acid in ["Gln","Thr","Ser","Asn","Cys","Tyr"]:
            samples[index]["isPolar"]=1
        elif amino_acid in ["Ala","Phe","Gly","Ile","Leu","Met","Pro","Val","Trp"]:
            samples[index]["isHydrophobic"]=1

    return samples

def feature_names():
    return ["isCharged", "isPolar", "isHydrophobic"]
