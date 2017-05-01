
"""
Returns polarity value for each residue in protein. Polarity values based on Grantham, Science 185:862-864(1974).

"""

import sys, re
import Bio.PDB

def feature(chain):
    
    polarity={"Ala":  8.1, "Arg": 10.5, "Asn": 11.0, "Asp": 13.0, "Cys":  5.5, 
              "Gln": 10.5, "Glu": 12.3, "Gly":  9.0, "His": 10.4, "Ile":  5.2,
              "Leu":  4.9, "Lys": 11.3, "Met":  5.7, "Phe":  5.2, "Pro":  8.0,
              "Ser":  9.2, "Thr":  8.6, "Trp":  5.4, "Tyr":  6.2, "Val":  5.9  }    
    
    #initialize dict of dict
    samples=dict()

    for residue in chain:
        
        res_index=residue.get_id()[1]
        name=residue.get_resname().title()
        
        polar_res=polarity[name]
                
        samples[res_index]={"polarity":polar_res}

    return samples

def feature_names():
    return ["molec_weight"]
