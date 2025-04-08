import os
import sys
file_path = sys.argv[0]
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path,"module"))

from module.Hamiltonian.Hamiltonian4TB.tight_binding_hamiltonian import TBHamiltonian    

import pickle


file_path = "/home/hp/users/kfh/DFTBAI1/example/test_TB/Fe_fm/Fe_fm_pd_4neighbour.pkl"
with open(file_path,"rb") as f:
    # pickletools.dis(f)
    model_dict = pickle.load(f)
    matrix = model_dict['model']
    model_info = model_dict['info']
    num_symbols = model_dict['num_symbols']
    name = model_dict['name']
    
tb_model = 0    
for i in matrix.keys():
    tb_model += matrix[i]

up = [0,2,4,6,8,10,12,14]
dn = [1,3,5,7,9,11,13,15]
print(tb_model[up,up])