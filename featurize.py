'''
DESCRIPTION
Make a big ol' table of protein structure and sequence features from PDB files.

USAGE
python featurize.py [FILE]|[DIRECTORY]
'''

import sys, os, glob, re, csv
from multiprocessing import Pool
from Bio.PDB import *
import pandas as pd

# get structural neighbors for each residue
from nearest_neighbors import nearest_neighbors

# path for custom features
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
import mutability
import polarity
import refractivity
import res_access
import res_exposure
import res_weight
import secondary_structure
###### END OF FEATURE IMPORT

# data structure to store features associated with each residue in a protein
class ProteinFeatures:
    def __init__(self, pdb_id, chain_id):
        self.feature_table = {}
        self.residue_neighbors = {}
        self.id = pdb_id
        self.chain = chain_id
    def add(self, feature_output):
        # does not check for key overlap!
        for res in feature_output:
            if res in self.feature_table:
                self.feature_table[res] = {**self.feature_table[res], **feature_output[res]}
            else:
                self.feature_table[res] = feature_output[res]
    def neighborize_features(self, feature_list):
        neighborized = {}
        for res_i in self.feature_table:
            for nn_id,res_j in enumerate(self.residue_neighbors[res_i]):
                for feature in feature_list:
                    self.feature_table[res_i]['nn'+str(nn_id+1)+'_'+feature] = self.feature_table[res_j][feature]
        return True

def featurize(pdb_file):

    # extract pdb id from file name
    path_re = re.match(r'(.+/)?([0-9a-zA-Z]+)_?(\w+)?\.pdb',pdb_file)
    pdb_id = path_re.group(2)
    #chain_id = 'A'
    chain_id = path_re.group(3).upper() # read chain from filename

    # load Structure object from PDB file
    parser = PDBParser()
    pdb_object = parser.get_structure(pdb_id,pdb_file)

    print(pdb_file, flush=True) #OUTPUT FOR TRACKING PROGRESS

    # take first model and specified chain
    chain = pdb_object[0][chain_id]

    # list of feature script names
    module_list = [amino_acid_identity, amino_acid_type, avg_buried, b_factor,
                   bulkiness, centrality, conservation, flexibility, hydrophobicity,
                   mutability, polarity, refractivity, res_access, res_exposure, 
                   secondary_structure, res_weight]

    # data structure to abstract details of feature scripts
    protein = ProteinFeatures(pdb_id,chain_id)

    # iteratively add features to each residue in the protein
    for module in module_list:
        protein.add(module.feature(chain))

    # list of all features used
    all_features = []
    for module in module_list:
        all_features.extend(module.feature_names())

    # get list of structural neighbors from neighbors.py
    protein.residue_neighbors = nearest_neighbors(chain)

    # create new features from neighbors
    protein.neighborize_features(all_features)

    # add PDB/chain label information (not used for classification)
    for residue in protein.feature_table:
        y_label = int((pdb_id, chain_id, residue) in csa_labels)
        protein.feature_table[residue] = {'y_label':y_label, 'chain':chain_id, 'pdb':pdb_id, 'res_id':residue, **protein.feature_table[residue]}
    return protein.feature_table

if __name__ == '__main__':

    path = sys.argv[1]
    NUM_THREADS = 7

    # read labels into dictionary for label lookup
    csa_labels = dict()
    with open('./data/csa/csa_from_lit.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            csa_labels[(row[0],row[3], int(row[4]))] = 1

    df = pd.DataFrame()

    if os.path.isdir(path):
        pdb_files = glob.glob(path+'/*.pdb')
        pool = Pool(processes=NUM_THREADS)

        for output in pool.map(featurize, pdb_files):
            # concatenate rows from each pdb into one large table
            df = df.append(list(output.values()),ignore_index=True)
    else:
        output = featurize(path)
        df = df.append(list(output.values()),ignore_index=True)

    # output table to csv for machine learning
    df.to_csv('features.csv')
