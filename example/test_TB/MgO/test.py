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
                            "sys_name":"MgO_s2p2_s2p_6n",
                            "group_type":"Space Group",
                            "group_name":"225",
                            "lattice_type":"CubiFace",
                            "lattice_parameter":{"a":1},
                            "atompos":[[0,0,0],[0.5,0.5,0.5]],
                            # "magdirect":[[1,1,0]],
                            "neighbour_list":[6,6]
                            },
            "orbit_init":[{"orbit_list":["s","s","px","py","pz","px","py","pz"]},
                          {"orbit_list":["s","s","px","py","pz"]}]
            }
model = TBHamiltonian(**model_input)
model.save_model("/home/hp/users/kfh/DFTBAI1/example/test_TB/MgO")
print(model.sym_hamiltonian_dict)


# mask = [1,3,7,9]
# mask = [1,3,7,9,14,16,18,20,23,25,27,29]
mask = []
device = "cuda:1"
# device = None
# device = "cpu"

model_path = "/home/hp/users/kfh/DFTBAI1/example/test_TB/MgO"
model_path = os.path.join(model_path,model_input["sysinit"]["sys_name"]+".pkl")

para_train = Para4Band_train(model_path,
                              zero_index=mask,
                              mask_index=mask,
                              device=device)
band = Band()
band.get_data("/home/hp/users/kfh/DFTBAI1/example/BAND-total/MgO/BAND.dat")
k_points = torch.tensor(band.content["k_vector"]).transpose(dim0=0,dim1=1)*2*torch.pi
band_index = [7,8,9,10,11,12,13,14,15,16,17,18,19]
energy = torch.tensor(band.content["energy"][:,band_index,0].reshape(k_points.shape[1],-1))
model_index = [i for i in range(13)]

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
                para=para)
end_time = time.time()
print(end_time-start_time)