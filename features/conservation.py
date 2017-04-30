'''
Sequence conservation
'''
import re, sys
from collections import defaultdict
import numpy as np
from Bio import AlignIO
from Bio.PDB import *

def feature_names():
    return ['entropy']

def shannon_entropy(dist):
    entropy = 0
    for key,value in dist.items():
        if key is not '-':
            entropy -= value*np.log2(value)
    return entropy

def conservation(align_filename):
    # load alignment with biopython
    align_format = "fasta"
    alignment = AlignIO.read(align_filename, align_format)

    # find positions that are in query
    query_seq = 0
    not_gap = []
    for i,residue in enumerate(alignment[query_seq]):
        if residue is not '-':
            not_gap.append(i)

    # calculate entropy at each position
    conservation_list = []
    for column_id in not_gap:
        column = alignment[:,column_id]
        counts = defaultdict(int)
        column_length = len(column)
        for res in column:
            counts[res] += 1./column_length
        conservation_list.append(shannon_entropy(counts))

    # list of conservation scores
    return conservation_list

def feature(structure):
    feature = feature_names()[0]

    # initialize dict of dicts
    output = dict()

    # calculate conservation from alignment
    align_filename = 'alignment/12as_a.mafft'
    conservation_list = conservation(align_filename)

    for index,res in enumerate(structure):
        res_id = int(res.id[1])
        output[res_id] = {feature: conservation_list[index]}

    return output

if __name__ == '__main__':

    pdb_file = sys.argv[1]
    path_re = re.match(r'(.+/)?([0-9a-zA-Z]+)_?(\w+)?\.pdb',pdb_file)
    pdb_id = path_re.group(2)

    parser = PDBParser()
    biopdb = parser.get_structure(pdb_id,pdb_file)

    structure = biopdb[0]['A']

    print(feature(structure))
