
"""
Returns dict of dict of dict of local secondary structure for each residue 
in protein for each PDB file.

Requires pointing to folder of DSSP files and PDB files.

"""

# DSSPData class and methods modified from http://openwetware.org/wiki/Wilke:ParseDSSP

import sys, re
import Bio.PDB
import os
import pickle

class DSSPData:
  def __init__(self):
      self.resnum = []
      self.struct = []


  def parseDSSP(self, file):
    input_handle = open(file, 'r')

    line_num = 0
    start=False
    for line in input_handle:
  
      if( re.search('#', line) ):
        start=True
        continue

      if( start ):
        
        self.struct.append( line[16:17] )
        x=line.split()
        self.resnum.append(x[1])

  def getResnums(self):
    return self.resnum
  def getSecStruc(self):
    return self.struct


sec_struct={}

pdb_path=sys.argv[1]
dssp_path=sys.argv[2]
 
pdb_id=[] 
for root, dirs, filenames in os.walk(pdb_path):
    for f in filenames:
        split=f.split(".")
        pdb_id.append(split[0])


for pdb in pdb_id:
    protein_struct={}
    
    full_path=dssp_path+pdb+".dssp"

    parser=DSSPData() 
    protein_dssp=parser.parseDSSP(full_path)
    
    res_indexes=parser.getResnums()    
    struc_codes=parser.getSecStruc()

    for i in range(len(res_indexes)):
        res_struct={"SS_alphahelix":0,"SS_betabridge":0, "SS_strand":0, "SS_3-10helix":0,
                "SS_pihelix":0, "SS_turn":0,"SS_bend":0}
        if struc_codes[i]=="H":
            res_struct["SS_alphahelix"]=1
        elif struc_codes[i]=="B":
            res_struct["SS_betabridge"]=1
        elif struc_codes[i]=="E":
            res_struct["SS_strand"]=1
        elif struc_codes[i]=="G":
            res_struct["SS_3-10helix"]=1
        elif struc_codes[i]=="I":
            res_struct["SS_pihelix"]=1
        elif struc_codes[i]=="T":
            res_struct["SS_turn"]=1
        elif struc_codes[i]=="S":
            res_struct["SS_bend"]=1
        try:
            protein_struct[int(res_indexes[i])]=res_struct
        except:
            break
            
    sec_struct[pdb]=protein_struct

output = open('dssp_output.txt', 'ab+')

pickle.dump(sec_struct, output)
output.close()
   

 
