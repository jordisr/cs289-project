'''
Kyte-Doolittle hydrophobicity scale
'''

from Bio.PDB import *
import sys, re

def feature(structure):

    # name of feature
    feature = 'hydrophobicity'

    # initialize dict of dicts
    output = dict()

    kyte_doolittle = {
        'Ala':  1.800,
        'Arg': -4.500,
        'Asn': -3.500,
        'Asp': -3.500,
        'Cys':  2.500,
        'Gln': -3.500,
        'Glu': -3.500,
        'Gly': -0.400,
        'His': -3.200,
        'Ile':  4.500,
        'Leu':  3.800,
        'Lys': -3.900,
        'Met':  1.900,
        'Phe':  2.800,
        'Pro': -1.600,
        'Ser': -0.800,
        'Thr': -0.700,
        'Trp': -0.900,
        'Tyr': -1.300,
        'Val':  4.200,
    }

    for res in structure:
        res_id = int(res.id[1])
        res_name = res.get_resname().title()
        output[res_id] = {feature: kyte_doolittle[res_name]}

    return output

def feature_names():
    return ['hydrophobicity']

'''
if __name__ == '__main__':

    pdb_file = sys.argv[1]
    path_re = re.match(r'(.+/)?([0-9a-zA-Z]+)_?(\w+)?\.pdb',pdb_file)
    pdb_id = path_re.group(2)

    print(feature(pdb_id, pdb_file))
'''
