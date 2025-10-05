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
                            "lattice_parameter":{"a":1,"c":5},
                            "atompos":[[0,0.5,0.5],[0,0,0.411776]],
                            "magdirect":[[0,0,1],[0,0,0]],
                            "neighbour_list":[3,3]
                            },
            "orbit_init":[{"orbit_list":["dx2-y2","dxy","dz2","dxz"],"spin_dict":{"dx2-y2":1,"dxy":1,"dz2":1,"dxz":1}},
                          {"orbit_list":["px","py","pz"],"spin_dict":{"px":1,"py":1,"pz":1}}]}
# model = TBHamiltonian(**model_input)
# model.save_model("/Volumes/KINGSTON/DFTBAI/DFTBAI_code/dftbai/example/test_TB/VX2")
# print(model.sym_hamiltonian_dict)


mask = []
device = "cuda:0"
device = None
para_train = Para4Band_train("/Volumes/KINGSTON/DFTBAI/DFTBAI_code/dftbai/example/test_TB/VX2/VX2_d_p_3n.pkl",
                              zero_index=mask,
                              mask_index=mask,
                              device=device)
band = Band()
band.get_data("/Volumes/KINGSTON/DFTBAI/DFTBAI_code/dftbai/example/BAND-total/VX2/V2Br2/BAND.dat")
k_points = torch.tensor(band.content["k_vector"]).transpose(dim0=0,dim1=1)*2*torch.pi
band_index = [10,11,12,13,14,15]
energy = torch.tensor(band.content["energy"][:,band_index])
energy = energy.reshape(energy.shape[0],-1)
model_index = [i for i in range(10,22)]
# para = torch.tensor([[1,0,0,1,1,1,0,0,1,0,0,1,0,1,1]],dtype=torch.float32)
# para = torch.randn(1,15)
# for i in mask:
#     para[0,i] = 0

start_time = time.time()
para_train.train(epoch = int(1e6),
                k_points = k_points,
                energy = energy,
                model_index=model_index)
end_time = time.time()
print(end_time-start_time)