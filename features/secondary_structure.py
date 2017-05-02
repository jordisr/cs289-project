
"""
Returns dict of dict of local secondary structure for each residue 
in protein.

"""
import sys, re
import Bio.PDB

def parseDSSP(file):
    input_handle = open(file, 'r')

    #res_SS=[]
    #res_indexes=[]
    res_dict={}

    start=False
    for line in input_handle:
  
      if( re.search('#', line) ):
        start=True
        continue

      if( start ):
        
        res_dict[line[6:10]]=line[16:17]


    return(res_dict)

def feature(chain):

    full_id=next(chain.get_residues()).get_full_id()
    
    pdb_id=full_id[0]
    chain_id=full_id[2]

    dssp_path="data/dssp/"
    full_path=dssp_path+pdb_id+'_'+chain_id+".dssp"     
    
    dssp_dict=parseDSSP(full_path)
    
    protein_struct={}    
    
    for residue in chain:
        res_id=residue.get_id()[1]

        res_struct={"SS_alphahelix":0,"SS_betabridge":0, "SS_strand":0, "SS_3-10helix":0,
                "SS_pihelix":0, "SS_turn":0,"SS_bend":0}
        
        if res_id in dssp_dict:
    
            if dssp_dict[res_id]=="H":
                res_struct["SS_alphahelix"]=1
            elif dssp_dict[res_id]=="B":
                res_struct["SS_betabridge"]=1
            elif dssp_dict[res_id]=="E":
                res_struct["SS_strand"]=1
            elif dssp_dict[res_id]=="G":
                res_struct["SS_3-10helix"]=1
            elif dssp_dict[res_id]=="I":
                res_struct["SS_pihelix"]=1
            elif dssp_dict[res_id]=="T":
                res_struct["SS_turn"]=1
            elif dssp_dict[res_id]=="S":
                res_struct["SS_bend"]=1
        
        try:
            protein_struct[res_id]=res_struct
        except:
            continue
         
    return protein_struct
    

def feature_names():
    return ["SS_alphahelix","SS_betabridge", "SS_strand", "SS_3-10helix",
            "SS_pihelix", "SS_turn","SS_bend"]
