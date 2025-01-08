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

# model_input = {"sysinit":{
#                             "sys_name":"Si_sps'",
#                             "group_type":"Space Group",
#                             "group_name":"227",
#                             "lattice_type":"CubiFace",
#                             "lattice_parameter":{"a":1},
#                             "atompos":[[1/8,1/8,1/8]],
#                             # "magdirect":[[0,0,0]],
#                             "neighbour_list":[2]
#                             },
#             "orbit_init":[{"orbit_list":["s","s","px","py","pz"]}]}

# tight_binding_model = TBHamiltonian(**model_input)
# print(tight_binding_model.sym_hamiltonian_dict)

mask = [1,2,6,7,9,10,12]
mask = []
# device = "cuda:1"
device = None
para_train = Para4Band_train("/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/Si_sps'.pkl",
                              zero_index=mask,
                              mask_index=mask,
                              device=device)
band = Band()
band.get_data("/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/BAND.dat")
k_points = torch.tensor(band.content["k_vector"]).transpose(dim0=0,dim1=1)*2*torch.pi
band_index = [1,2,3,4]
energy = torch.tensor(band.content["energy"][:,band_index])
model_index = [1,2,3,4]
# para = torch.tensor([[1,0,0,1,1,1,0,0,1,0,0,1,0,1,1]],dtype=torch.float32)
para = torch.randn(1,15)


start_time = time.time()
para_train.train(epoch = int(1e7),
                k_points = k_points,
                energy = energy,
                model_index=model_index,
                para=para)
end_time = time.time()
print(end_time-start_time)

