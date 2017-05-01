
"""

"""

import sys, re
import Bio.PDB
import pickle

  
def feature(chain):
    
    # read data
    data_path=sys.path.append("./DSSP/dssp_output.txt")
    output = open(data_path, 'rb')
    sec_struct = pickle.load(output)  
    
    full_id=next(chain.get_residues()).get_full_id()
    
    protein_SS=sec_struct[full_id[0]] 
    
    for key in protein_SS:
        key=int(key)
        
    return protein_SS

def feature_names():
    return ["SS_alphahelix","SS_betabridge", "SS_strand", "SS_3-10helix",
            "SS_pihelix", "SS_turn","SS_bend"]
