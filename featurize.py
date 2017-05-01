'''
DESCRIPTION
Make a big ol' table of protein structure and sequence features from PDB files.

USAGE
python featurize.py [FILE]|[DIRECTORY]
'''

import sys, os, glob, re
from Bio.PDB import *
from multiprocessing import Pool
import pandas as pd

from neighbors import get_neighbors
sys.path.append("./features")
###### IMPORT FEATURES HERE
import amino_acid_identity
import amino_acid_type
import avg_buried
import b_factor
import bulkiness
import centrality
import conservation
import flexibility
import hydrophobicity
import polarity
import refractivity
import res_access
import res_exposure
import res_weight
###### END OF FEATURE IMPORT

# just a placeholder to get us started
class Protein:
    def __init__(self, pdb_id):
        self.residues = {}
        self.id = pdb_id
        self.residue_neighbors = {}
    def add(self, feature_output):
        # does not check for key overlap!
        for res in feature_output:
            if res in self.feature_table:
                self.feature_table[res] = {**self.feature_table[res], **feature_output[res]}
            else:
                self.feature_table[res] = feature_output[res]

def featurize(pdb_file):

    # extract pdb id from file name
    path_re = re.match(r'(.+/)?([0-9a-zA-Z]+)_?(\w+)?\.pdb',pdb_file)
    pdb_id = path_re.group(2)
    chain_id = 'A'

    # load Structure object from PDB file
    parser = PDBParser()
    pdb_object = parser.get_structure(pdb_id,pdb_file)

    # take first model and chain A (should be modified for generalizability)
    chain = pdb_object[0][chain_id]

    #print("-- STAND BACK -- FEATURIZING, FOOL!")

    # list of feature script names
    module_list = [amino_acid_identity, amino_acid_type, avg_buried, b_factor,
                   bulkiness, centrality, conservation, flexibility, hydrophobicity,
                   polarity, refractivity, res_access, res_exposure, res_weight]
        
    # data structure to abstract details of feature scripts

    protein = protein_features()

    # merge output onto larger data structure
    for module in module_list:
        protein.add(module.feature(chain))

    # get list of neighbors from neighbors.py
    neighbors = get_neighbors(chain)

    # neighborize features here
    neighborized = dict()
    for residue in protein.feature_table:
        neighbor_list = neighbors['within10'][residue]

        #### placeholder for fancier stuff
        top_neighbor = neighbor_list[0]

        neighbor_features = protein.feature_table[top_neighbor]
        for_res = dict()
        for key,val in neighbor_features.items():
            for_res['nn1_'+key] = val
        neighborized[residue] = {**protein.feature_table[residue], **for_res}
        #####


    # add label information (not used in classification)
    #for residue in protein.feature_table:
    #    protein.feature_table[residue] = {'chain':chain_id, 'pdb':pdb_id, 'res_id':residue, **protein.feature_table[residue]}

    # add label information (not used in classification)
    for residue in neighborized:
        neighborized[residue] = {'chain':chain_id, 'pdb':pdb_id, 'res_id':residue, **neighborized[residue]}

    #print(protein.feature_table) # DEBUG
    return neighborized

'''
def neighborize_features(protein, metric):

    for res in protein.residues:
        neighbors = protein.neighbors[metric][res]
        for feature in res.get
          

    return
'''


if __name__ == '__main__':

    path = sys.argv[1]
    NUM_THREADS = 8

    df = pd.DataFrame()

    if os.path.isdir(path):
        pdb_files = glob.glob(path+'/*.pdb')
        pool = Pool(processes=NUM_THREADS)

        for output in pool.map(featurize, pdb_files):
            # concatenate rows from each pdb into one large table
            df = df.append(list(output.values()),ignore_index=True)
            print(df)
    else:
        output = featurize(path)
        df = df.append(list(output.values()),ignore_index=True)
        print(df)

    # save data structure to pickle/compressed file
    # also save it as csv
    # feature subsets in this script or another?
