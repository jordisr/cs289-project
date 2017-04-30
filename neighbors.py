'''
Calculate sequence and structural neighbors for each residue.
- Sequence (+/- 5 residues)
    If residue is within first or last 5, will take 10 nearest in sequence
- Structure (within 5 Å, within 10Å, within 15Å)
'''

import sys, re
from Bio.PDB import *
import numpy as np

def get_neighbors(structure):

    # neighborhood metrics
    metrics = ['sequence10', 'within5', 'within10']

    # initialize dict of dicts
    output = dict()
    for metric in metrics:
        output[metric] = dict()

    for res_i in structure:
        res_i_id = int(res_i.id[1])
        for metric in metrics:
            # initialize empty list of neighbors
            output[metric][res_i_id] = []
        for res_j in structure:
            if res_i is not res_j:
                res_j_id = int(res_j.id[1])
                # sequence neighbors (take 20 and choose closest 10)
                if np.abs(res_i_id - res_j_id) <= 10:
                    output['sequence10'][res_i_id].append(res_j_id)
                # distance neighbors
                ca_distance = res_i['CA']-res_j['CA']
                if ca_distance < 5:
                    output['within5'][res_i_id].append(res_j_id)
                    output['within10'][res_i_id].append(res_j_id)
                elif ca_distance < 10:
                    output['within10'][res_i_id].append(res_j_id)

    # For residues in the first or last 5, there are not five residues on either
    #  side. Workaround is to take at most 10 residues on either side and then
    #  take the closest 10 of those.
    for res_id in output['sequence10']:
        #print('orig',output['sequence10'][res_id]) #uncomment for testing
        output['sequence10'][res_id] = sorted(sorted(output['sequence10'][res_id],key=lambda x: np.abs(res_id-x))[:10])
        #print('sort',output['sequence10'][res_id])) #uncomment for testing

    return output

if __name__ == '__main__':

    pdb_file = sys.argv[1]
    path_re = re.match(r'(.+/)?([0-9a-zA-Z]+)_?(\w+)?\.pdb',pdb_file)
    pdb_id = path_re.group(2)

    print(get_neighbors(pdb_id, pdb_file))
