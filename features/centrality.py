'''
Calculate residue centrality, defined as frac{n-1}{\sum_{i \neq j} d(i,j)},
where n is the total number of residues in the PDB, and d(i,j) is the Euclidean
distance between residue i and residue j.
'''

from Bio.PDB import *
import sys, re

def feature(structure):

    # name of feature
    feature = 'centrality'

    # initialize dict of dicts
    output = dict()

    # total number of residues
    num_residues = len(list(structure.get_residues()))

    for res_i in structure:
        res_i_id = int(res_i.id[1])
        output[res_i_id] = {feature: 0}
        for res_j in structure:
            if res_i is not res_j:
                ca_distance = res_i['CA']-res_j['CA']
                output[res_i_id][feature] += ca_distance
        # invert total of pairwise distances to get centrality
        output[res_i_id][feature] = (num_residues-1)/output[res_i_id][feature]

    return output

if __name__ == '__main__':

    pdb_file = sys.argv[1]
    path_re = re.match(r'(.+/)?([0-9a-zA-Z]+)_?(\w+)?\.pdb',pdb_file)
    pdb_id = path_re.group(2)

    print(feature(pdb_id, pdb_file))
