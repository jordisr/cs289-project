
"""
Returns dict of dict of local secondary structure for each residue 
in protein.

"""
import sys, re
import Bio.PDB

def parseDSSP(file):
    input_handle = open(file, 'r')

    res_SS=[]
    res_indexes=[]

    start=False
    for line in input_handle:
  
      if( re.search('#', line) ):
        start=True
        continue

      if( start ):
        
        res_SS.append(line[16:17])
        res_indexes.append(line[6:10])
    
    return(res_SS, res_indexes)

def feature(chain):

    full_id=next(chain.get_residues()).get_full_id()
    
    pdb_id=full_id[0]
    chain_id=full_id[2]

    dssp_path="data/dssp/"
    full_path=dssp_path+pdb_id+'_'+chain_id+".dssp"    
    
    struc_codes, res_indexes=parseDSSP(full_path)
    
    pdb_res_indexes=[]
    for residue in chain:
        pdb_res_indexes.append(residue.get_id()[1])
    
    protein_struct={}
    
    for i in range(len(pdb_res_indexes)):
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
            protein_struct[int(pdb_res_indexes[i])]=res_struct
        except:
            continue
         
    return protein_struct
    

def feature_names():
    return ["SS_alphahelix","SS_betabridge", "SS_strand", "SS_3-10helix",
            "SS_pihelix", "SS_turn","SS_bend"]

