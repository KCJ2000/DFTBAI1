import os
import sys
file_path = sys.argv[0]
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path,"module"))

import time
import torch

from module.Hamiltonian.Hamiltonian4TB.tight_binding_hamiltonian import TBHamiltonian    
from module.physics_property.band.band import Band
from module.parameter.para4band.para4band import Para4Band_train

model_input = {"sysinit":{
                            "sys_name":"VX2_d_p_3n",
                            "group_type":"Magnetic Group",
                            "group_name":"123.342",
                            "lattice_type":"TetrPrim",
                            "lattice_parameter":{"a":1,"c":10},
                            "atompos":[[0,0.5,0.5],[0,0,0.411776]],
                            "magdirect":[[0,0,1],[0,0,0]],
                            "neighbour_list":[3,3]
                            },
            "orbit_init":[{"orbit_list":["dx2-y2","dxy","dz2","dxz"],"spin_dict":{"dx2-y2":1,"dxy":1,"dz2":1,"dxz":1}},
                          {"orbit_list":["px","py"],"spin_dict":{"px":0,"py":0}}]}
model = TBHamiltonian(**model_input)
model.save_model("/Volumes/KINGSTON/DFTBAI/DFTBAI_code/dftbai/example/test_TB/VX2")
print(model.sym_hamiltonian_dict)