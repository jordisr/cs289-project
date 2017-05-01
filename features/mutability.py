
"""
Returns the relative propensity of each residue to mutate.

"""

import sys, re
import Bio.PDB


def feature(chain):
    
    mutability={"Ala": 100, "Arg":  65, "Asn": 134, "Asp": 106, "Cys": 20, 
                "Gln":  93, "Glu": 102, "Gly":  49, "His":  66, "Ile": 96, 
                "Leu":  40, "Lys":  56, "Met":  94, "Phe":  41, "Pro": 56, 
                "Ser": 120, "Thr":  97, "Trp":  18, "Tyr":  41, "Val": 74,
                "Sec":  20, "Pyl": 56}    
    
    #initialize dict of dict
    samples=dict()

    for residue in chain:
        
        res_index=residue.get_id()[1]
        name=residue.get_resname().title()
        
        mut_res=mutability[name]
                
        samples[res_index]={"mutability":mut_res}
      
    return samples

def feature_names():
    return ["mutability"]
