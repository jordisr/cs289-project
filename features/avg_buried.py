
"""
Returns the area of the residue that is buried in a folded protein structure on average.

"""

import sys, re
import Bio.PDB


def feature(chain):
    
    avg_buried={"Ala":  86.600, "Arg": 162.200, "Asn": 103.300, "Asp":  97.800, "Cys": 132.300,
              "Gln": 119.200, "Glu": 113.900, "Gly":  62.900, "His": 155.800, "Ile": 158.000,
              "Leu": 164.100, "Lys": 115.500, "Met": 172.900, "Phe": 194.100, "Pro": 92.900,
              "Ser":  85.600, "Thr": 106.500, "Trp": 224.600, "Tyr": 177.700, "Val": 141.000,
              "Sec": 132.300, "Pyl": 115.500}    
    
    #initialize dict of dict
    samples=dict()

    for residue in chain:
        
        res_index=residue.get_id()[1]
        name=residue.get_resname().title()
        
        avg_buried_res=avg_buried[name]
                
        samples[res_index]={"avg_buried":avg_buried_res}
       
    return samples

def feature_names():
    return ["avg_buried"]
