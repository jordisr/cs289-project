
"""
Returns vector directions and absolute distance between central carbon on 
C-alpha chain and the first carbon on the C-beta chain (side chain). 

"""

import sys, re
import Bio.PDB
from Bio.PDB import HSExposureCB
import numpy as np


def feature(chain):
    
    #initialize dict of dict
    samples=dict()
    
    #calculate exposure for entire chain
    hse=HSExposureCB(chain)

    #assign exposure for each residue to dictionary
    for residue in chain:
        
        #calculate vector to first side chain carbon from central carbon
        vector_cb=hse._get_cb(residue, residue, residue)        
        
        #extract vector directions
        vector_cb_x=vector_cb[0][0]
        vector_cb_y=vector_cb[0][1]
        vector_cb_z=vector_cb[0][2]
        vector_len=np.sqrt(vector_cb_x**2+vector_cb_y**2+vector_cb_z**2)
       
        res_index=residue.get_id()[1]
        samples[res_index]={"dist_cb_x":vector_cb_x, "dist_cb_y":vector_cb_y,
                            "dist_cb_z":vector_cb_z, "dist_cb":vector_len}

    return samples

def feature_names():
    return ["dist_cb_x", "dist_cb_y", "dist_cb_z", "dist_cb"]
