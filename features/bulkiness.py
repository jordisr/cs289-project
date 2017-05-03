
"""
Returns value for how bulky each residue is.

"""

import sys, re
import Bio.PDB


def feature(chain):
    
    bulkiness={"Ala": 11.500, "Arg": 14.280, "Asn": 12.820, "Asp": 11.680, "Cys": 13.460,
               "Gln": 14.450, "Glu": 13.570, "Gly":  3.400, "His": 13.690, "Ile": 21.400,
               "Leu": 21.400, "Lys": 15.710, "Met": 16.250, "Phe": 19.800, "Pro": 17.430,
               "Ser":  9.470, "Thr": 15.770, "Trp": 21.670, "Tyr": 18.030, "Val": 21.570,
               "Sec": 13.460, "Pyl": 15.710}     
    
    #initialize dict of dict
    samples=dict()

    for residue in chain:
        
        res_index=residue.get_id()[1]
        name=residue.get_resname().title()
        
        bulk_res=bulkiness[name]
                
        samples[res_index]={"bulkiness":bulk_res}

    return samples

def feature_names():
    return ["bulkiness"]
