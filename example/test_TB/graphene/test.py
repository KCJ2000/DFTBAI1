import os
import sys
file_path = sys.argv[0]
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path,"module"))
import time
import torch


from module.Hamiltonian.Hamiltonian4TB.tight_binding_hamiltonian import TBHamiltonian    
from module.parameter.para4band.para4band import Para4Band_train 
from module.physics_property.band.band import Band

# model_input = {"sysinit":{
#                             "sys_name":"Graphene_pz_withspin",
#                             "group_type":"Magnetic Group",
#                             "group_name":"191.234",
#                             "lattice_type":"HexaPrim",
#                             "lattice_parameter":{"a":1,"c":3},
#                             "atompos":[[1/3,2/3,0]],
#                             "magdirect":[[0,0,0]],
#                             "n_neighbour":3
#                             },
#             "orbit_init":[{"orbit_list":["pz"],"spin_dict":{"pz":1}}]}

model_input = {"sysinit":{
                            "sys_name":"Graphene_pz",
                            "group_type":"Space Group",
                            "group_name":"P6/mmm",
                            "lattice_type":"HexaPrim",
                            "lattice_parameter":{"a":1,"c":3},
                            "atompos":[[1/3,2/3,0]],
                            # "magdirect":[[0,0,0]],
                            "neighbour_list":[3]
                            },
            "orbit_init":[{"orbit_list":["pz"]}]}

tight_binding_model = TBHamiltonian(**model_input)
tight_binding_model.save_model("/home/hp/users/kfh/DFTBAI1/example/test_TB/graphene")
print(tight_binding_model.sym_hamiltonian_dict)

### 训练
mask = []
device = "cuda:1"
para_train = Para4Band_train("/home/hp/users/kfh/DFTBAI1/example/test_TB/graphene/Graphene_pz.pkl",
                              mask,
                              device=device)
band = Band()
band.get_data("/home/hp/users/kfh/DFTBAI1/example/test_TB/graphene/band.npz")
k_points = torch.tensor(band.content["k_vector"]).transpose(dim0=0,dim1=1)*2*torch.pi
band_index = [0,1]
energy = torch.tensor(band.content["energy"].reshape(k_points.shape[1],-1))
model_index = [0,1]

para = torch.randn(1,3)
for i in mask:
    para[0,i] = 0
start_time = time.time()
para_train.train(epoch = int(1e7),
                k_points = k_points,
                energy = energy,
                model_index=model_index,
                para=para)
end_time = time.time()
print(end_time-start_time)


