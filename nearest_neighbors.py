'''
Use priority queue to find 10 nearest structural neighbors.
Added some error checking to make sure residue has CA atom.
'''

import sys, re
from heapq import heappush, heappop, nsmallest
from Bio.PDB import *
import numpy as np

def nearest_neighbors(structure, nn=10):

    # initialize dict of dicts
    output = dict()

    for res_i in structure:
        neighbors = []
        res_i_id = int(res_i.id[1])
        output[res_i_id] = []
        for res_j in structure:
            if 'CA' in res_i:
                if res_i is not res_j:
                    res_j_id = int(res_j.id[1])
                    if 'CA' in res_j:
                        ca_distance = res_i['CA']-res_j['CA']
                        heappush(neighbors, (ca_distance, res_j_id))
                        neighbors = nsmallest(nn, neighbors)
                output[res_i_id] = [tup[1] for tup in neighbors]
            else:
                output[res_i_id] = []
    return output

if __name__ == '__main__':

    # parse PDB from command line for testing
    pdb_file = sys.argv[1]
    path_re = re.match(r'(.+/)?([0-9a-zA-Z]+)_?(\w+)?\.pdb',pdb_file)
    pdb_id = path_re.group(2)
    chain_id = 'A'
    parser = PDBParser()
    pdb_object = parser.get_structure(pdb_id,pdb_file)
    chain = pdb_object[0][chain_id]

    print(nearest_neighbors(chain))
