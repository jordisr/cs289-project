
"""
Returns molecular weight for each residue in protein.

"""

import sys, re
import Bio.PDB


def feature(chain):
    
    molec_weight={"ALA": 89, "ARG": 174, "ASN": 132, "ASP": 133, "CYS": 121,
                  "GLN": 146, "GLU": 147, "GLY": 75, "HIS": 155, "ILE": 131, 
                  "LEU": 131, "LYS": 146, "MET": 149, "PHE": 165, "PRO": 115, 
                  "SER": 105, "THR": 119, "TRP": 204, "TYR": 181, "VAL": 117}    
    
    #initialize dict of dict
    samples=dict()

    for residue in chain:
        
        res_index=residue.get_id()[1]
        name=residue.get_resname()
        
        weight=molec_weight[name]
                
        samples[res_index]={"molec_weight":weight}
        print(samples)
    return samples

def feature_names():
    return ["molec_weight"]
