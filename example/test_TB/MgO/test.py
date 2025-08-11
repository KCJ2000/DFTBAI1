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



model_input = {"sysinit":{
                            "sys_name":"MgO_spd_sp2_7n",
                            "group_type":"Space Group",
                            "group_name":"225",
                            "lattice_type":"CubiFace",
                            "lattice_parameter":{"a":1},
                            "atompos":[[0,0,0],[0.5,0.5,0.5]],
                            # "magdirect":[[1,1,0]],
                            "neighbour_list":[7,7]
                            },
            "orbit_init":[{"orbit_list":["s","px","py","pz","dxy","dyz","dxz","dz2","dx2-y2"]},
                          {"orbit_list":["s","px","py","pz","px","py","pz"]}]
            }
# model = TBHamiltonian(**model_input)
# model.save_model("/data/home/kongfh/DFTBAI1/example/test_TB/MgO")
# print(model.sym_hamiltonian_dict)


mask = []
device = "cuda:0"
# device = None
# device = "cpu"

model_path = "/data/home/kongfh/DFTBAI1/example/test_TB/MgO"
model_path = os.path.join(model_path,model_input["sysinit"]["sys_name"]+".pkl")

para_train = Para4Band_train(model_path,
                              zero_index=mask,
                              mask_index=mask,
                              device=device)
band = Band()
band.get_data("/data/home/kongfh/DFTBAI1/example/BAND-total/MgO/BAND.dat")
k_points = torch.tensor(band.content["k_vector"]).transpose(dim0=0,dim1=1)*2*torch.pi
band_index = [i for i in range(4,20)]
energy = torch.tensor(band.content["energy"][:,band_index,0].reshape(k_points.shape[1],-1))
model_index = [i for i in range(16)]

print("band_index:",band_index)
print("model_index:",model_index)
print("mask:",mask)
print("neighbour_list",model_input["sysinit"]["neighbour_list"])

# para = torch.randn(1,42)
para = None

start_time = time.time()
para_train.train(epoch = int(1e6),
                k_points = k_points,
                energy = energy,
                model_index=model_index,
                max_iteration = 5e4,
                para=para)
end_time = time.time()
print(end_time-start_time)