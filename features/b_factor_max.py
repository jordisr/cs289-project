
"""
Returns maximum b-factor value for each residue.

"""

import sys, re
import Bio.PDB

def feature(chain):
    
    #initialize dict of dict
    samples=dict()

    for residue in chain:
        
        #find max b-factor for each residue
        b_factor_list=[]
        for atom in residue:
            b_factor_list.append(atom.get_bfactor())
        b_factor_max=max(b_factor_list)
        
        #append b-factor feature to dictionary    
        res_index=residue.get_id()[1]
        samples[res_index]={"b_factor_max":b_factor_max}

    return samples

def feature_names():
    return ["b_factor_max"]
