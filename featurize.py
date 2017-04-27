'''
DESCRIPTION
Make a big ol' table of protein structure and sequence features from PDB files.

USAGE
python featurize.py [FILE]|[DIRECTORY]
'''

import sys, os, glob, re
from Bio.PDB import *
from multiprocessing import Pool

from neighbors import get_neighbors
sys.path.append("./features")
###### IMPORT FEATURES HERE
import centrality
import hydrophobicity
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
            if res in self.residues:
                self.residues[res] = {**self.residues[res], **feature_output[res]}
            else:
                self.residues[res] = feature_output[res]

def featurize(pdb_file):

    # extract pdb id from file name
    path_re = re.match(r'(.+/)?([0-9a-zA-Z]+)_?(\w+)?\.pdb',pdb_file)
    pdb_id = path_re.group(2)

    parser = PDBParser()
    biopdb = parser.get_structure(pdb_id,pdb_file)
    
    print("-- STAND BACK -- FEATURIZING, FOOL!")

    # list of feature script names
    module_list = [centrality, hydrophobicity]

    # data structure to abstract details of feature scripts
    protein = Protein(pdb_id)
    print(protein.residues)

    # merge output onto larger data structure
    for module in module_list:
        protein.add(module.feature(biopdb))

    # get list of neighbors from neighbors.py
    protein.neighbors = get_neighbors(pdb_id,pdb_file)
    
    # metric chooses the
    metrics = ['sequence10', 'within5', 'within10']
    neighborize_features(protein, metrics[0])  # kludge?

    print(protein.residues) # DEBUG

    # OUTLINE
    # - Append/average features from neighbors to make large table
    # - return data structure (pandas object?)

    return protein


def neighborize_features(protein, metric):

    for res in protein.residues:
        neighbors = protein.neighbors[metric][res]
        for feature in res.get


    return


if __name__ == '__main__':

    path = sys.argv[1]
    NUM_THREADS = 8

    if os.path.isdir(path):
        pdb_files = glob.glob(path+'/*.pdb')
        pool = Pool(processes=NUM_THREADS)
        featurized_proteins = pool.map(featurize, pdb_files)
        # concatenate rows from each pdb into one large table
        # data_frame = ...
    else:
        data_frame = featurize(path)

    # save data structure to pickle/compressed file
    # also save it as csv
    # feature subsets in this script or another?
