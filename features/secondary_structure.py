
"""
Takes pre-processed DSSP dictionaries and translates them into residue features. 
"""

import sys, re
import Bio.PDB
import pickle

def feature(chain):
    
    # read in pre-processed DSSP data as nested dictionaries
    output = open("DSSP/dssp_output.txt", 'rb')
    sec_struct = pickle.load(output)  
    
    #identify PDB_id
    full_id=next(chain.get_residues()).get_full_id()
    
    #extract secondary structure from data file
    protein_SS=sec_struct[full_id[0]] 
    
    #convert residue indexes to integers
    for key in protein_SS:
        key=int(key)
        
    return protein_SS

def feature_names():
    return ["SS_alphahelix","SS_betabridge", "SS_strand", "SS_3-10helix",
            "SS_pihelix", "SS_turn","SS_bend"]
