"""
Returns exposure value for each residue.  Exposure is defined as the number of C-alpha (main chain) 
atoms that surround the central C-alpha atom. 
"""

import sys, re
import Bio.PDB
from Bio.PDB import ExposureCN


def feature(chain):
    
    #initialize dict of dict
    samples=dict()
    
    #calculate exposure for entire chain
    hse_cn=ExposureCN(chain)

    #assign exposure for each residue to dictionary
    for residue in hse_cn:
        exposure=residue[1]   
        res_index=residue[0].get_id()[1]
        samples[res_index]={"res_access":exposure}

    return samples

def feature_names():
    return ["res_access"]
