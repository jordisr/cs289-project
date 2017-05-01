
"""
Returns molecular weight for each residue in protein.

"""

import sys, re
import Bio.PDB


def feature(chain):
    
    molec_weight={"Ala": 89, "Arg": 174, "Asn": 132, "Asp": 133, "Cys": 121,
                  "Gln": 146, "Glu": 147, "Gly": 75, "His": 155, "Ile": 131, 
                  "Leu": 131, "Lys": 146, "Met": 149, "Phe": 165, "Pro": 115, 
                  "Ser": 105, "Thr": 119, "Trp": 204, "Tyr": 181, "Val": 117}    
    
    #initialize dict of dict
    samples=dict()

    for residue in chain:
        
        res_index=residue.get_id()[1]
        name=residue.get_resname().title()
        
        weight=molec_weight[name]
                
        samples[res_index]={"molec_weight":weight}
        print(samples)
    return samples

def feature_names():
    return ["molec_weight"]
