import sys, re, csv
from Bio.PDB import *
from Bio.SeqUtils import seq1
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
from Bio import SeqIO

pdb_file_path = './data/pdb/'
fasta_out_path = './data/fasta/'

with open('./data/csa/csa_from_lit.csv','r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        pdb_id = row[0]
        chain_id = row[3]
        try:
            # output atoms only from chain of interest
            with open(pdb_file_path+'download/'+pdb_id+'.pdb', 'r') as in_file:

                with open(pdb_file_path+pdb_id+'_'+chain_id+'.pdb', 'w') as out_file:
                    for line in in_file:
                        if line[0:4] == 'ATOM' and line[21] == chain_id:
                            out_file.write(line)
                    out_file.write('END\n')

            # This part is probably slower than it should be but since it is run
            # once, is not a huge deal. Bio.SeqIO can supposedly read
            # pdb-atom to look at just atoms and not seqres fields in compiling
            # the sequence.
            parser = PDBParser()
            structure = parser.get_structure(pdb_id,pdb_file_path+'download/'+pdb_id+'.pdb')
            out_path = fasta_out_path+pdb_id+'_'+chain_id+'.fasta'

            residues = []
            for res in structure[0][chain_id]:
                if is_aa(res):
                    res_name = seq1(res.get_resname().title())
                    residues.append(res_name)
            seq = SeqRecord(Seq(''.join(residues), IUPAC.protein), id=pdb_id+'_'+chain_id.lower(),description='')
            SeqIO.write(seq, out_path, "fasta")
        except:
            print(row) # print CSA input if something goes wrong
