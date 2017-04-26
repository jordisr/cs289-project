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

def featurize(pdb_file):

    # extract pdb id from file name
    path_re = re.match(r'(.+/)?([0-9a-zA-Z]+)_?(\w+)?\.pdb',pdb_file)
    pdb_id = path_re.group(2)

    # Could add code here to read PDB into Bio.PDB

    print("-- STAND BACK -- FEATURIZING, FOOL!")

    ##### CALL INDIVIDUAL FEATURE FUNCTIONS HERE
    print(centrality.feature(pdb_id,pdb_file))
    print(hydrophobicity.feature(pdb_id,pdb_file))
    ##### END OF FEATURE CALLING

    # OUTLINE
    # - Get list of neighbors from neighbors.py
    neighbors = get_neighbors(pdb_id,pdb_file)
    print(neighbors)
    # - Append/average features from neighbors to make large table
    # - return data structure (pandas object?)
    return None

if __name__ == '__main__':

    path = sys.argv[1]
    NUM_THREADS = 8

    if os.path.isdir(path):
        pdb_files = glob.glob(path+'/*.pdb')
        pool = Pool(processes=NUM_THREADS)
        data_frame_list = pool.map(featurize, pdb_files)
        # concatenate rows from each pdb into one large table
        # data_frame = ...
    else:
        data_frame = featurize(path)

    # save data structure to pickle/compressed file
    # also save it as csv
    # feature subsets in this script or another?
