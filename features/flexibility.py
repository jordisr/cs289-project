
"""
Returns flexibility of the residue.

"""

import sys, re
import Bio.PDB


def feature(chain):
    
    flexibility={"Ala": 0.360, "Arg": 0.530, "Asn":  0.460, "Asp":  0.510, "Cys":  0.350,
                 "Gln": 0.490, "Glu": 0.500, "Gly":  0.540, "His":  0.320, "Ile":  0.460,
                 "Leu": 0.370, "Lys": 0.470, "Met":  0.300, "Phe":  0.310, "Pro":  0.510,
                 "Ser": 0.510, "Thr": 0.440, "Trp":  0.310, "Tyr":  0.420, "Val":  0.390,
                 "Sec": 0.350, "Pyl": 0.470}    
    
    #initialize dict of dict
    samples=dict()

    for residue in chain:
        
        res_index=residue.get_id()[1]
        name=residue.get_resname().title()
        
        flex_res=flexibility[name]
                
        samples[res_index]={"flexibility":flex_res}

    return samples

def feature_names():
    return ["flexibility"]
