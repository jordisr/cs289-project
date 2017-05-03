"""
Returns value of refractive index of each residue.

"""

import sys, re
import Bio.PDB

def feature(chain):
    
    refractivity={"Ala":  4.340, "Arg": 26.660, "Asn": 13.280, "Asp": 12.000, "Cys": 35.770,
                  "Gln": 17.560, "Glu": 17.260, "Gly":  0.000, "His": 21.810, "Ile": 19.060, 
                  "Leu": 18.780, "Lys": 21.290, "Met": 21.640, "Phe": 29.400, "Pro": 10.930,
                  "Ser":  6.350, "Thr": 11.010, "Trp": 42.530, "Tyr": 31.530, "Val": 13.920,
                  "Sec": 35.770, "Pyl": 21.290} 
    
    #initialize dict of dict
    samples=dict()

    for residue in chain:
        
        #extract residue index and identity
        res_index=residue.get_id()[1]
        name=residue.get_resname().title()
        
        #assign refractivity value to residue dictionary
        refrac_res=refractivity[name]
        samples[res_index]={"refractivity":refrac_res}
        
    return samples

def feature_names():
    return ["refractivity"]
