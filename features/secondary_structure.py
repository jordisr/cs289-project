
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
    
    #iterate through PDB file 
    start=False
    for line in input_handle:
  
      if( re.search('#', line) ):
        start=True
        continue

      if( start ):
        
        #append secondary structure and residue indices to list
        res_SS.append(line[16:17])
        res_indexes.append(line[6:10])
    
    return(res_SS, res_indexes)

def feature(chain):

    #extract pdb_id from chain
    full_id=next(chain.get_residues()).get_full_id()  
    pdb_id=full_id[0]

    dssp_path="data/dssp/"
    full_path=dssp_path+pdb_id+".dssp"    
    
    #extract secondary structure from input chain
    struc_codes, res_indexes=parseDSSP(full_path)
    
    protein_struct={}
    
    #update residue feature dictionary 
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
         
    return protein_struct
    

def feature_names():
    return ["SS_alphahelix","SS_betabridge", "SS_strand", "SS_3-10helix",
            "SS_pihelix", "SS_turn","SS_bend"]

