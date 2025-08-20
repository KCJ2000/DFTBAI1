import os
import sys
file_path = sys.argv[0]
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path,"module"))
import time
import torch
import numpy as np


from module.Hamiltonian.Hamiltonian4TB.tight_binding_hamiltonian import TBHamiltonian    
from module.parameter.para4band.para4band import Para4Band_train 
from module.physics_property.band.band import Band

model_input={
    "sysinit":{
        "sys_name":"MX2_d(no_z2)_p_3n",
        "group_type":"Magnetic Group",
        "group_name":"14.75",
        "lattice_type":"MonoPrim",
        "lattice_parameter":{"a":5,"Gamma":90/180*np.pi,"b":1,"c":1},
        "atompos":[[0.5,0,0],[0.54,0.85,0.35]],
        "magdirect":[[-1,0,0],[0,0,0]],
        "neighbour_list":[4,4]
    },
    "orbit_init":[{"orbit_list":["dyz","dxz","dx2-y2","dxy"],"spin_dict":{"dyz":1,"dxz":1,"dx2-y2":1,"dxy":1}},
                  {"orbit_list":["pz","py","px"],"spin_dict":{"px":1,"pz":1,"py":1}}]
    }
# model = TBHamiltonian(**model_input)
# model.save_model("/data/home/kongfh/DFTBAI1/example/test_TB/MX2")
# print(model.sym_hamiltonian_dict)


mask = []
device = "cuda:0"

model_path = "/data/home/kongfh/DFTBAI1/example/test_TB/MX2"
model_path = os.path.join(model_path,model_input["sysinit"]["sys_name"]+".pkl")

para_train = Para4Band_train(model_path,
                              zero_index=mask,
                              mask_index=mask,
                              device=device)
band = Band()
band.get_data("/data/home/kongfh/DFTBAI1/example/BAND-total/MnTe2-metal/BAND.dat")
k_points = torch.tensor(band.content["k_vector"]).transpose(dim0=0,dim1=1)*2*torch.pi
band_index = [i for i in range(72,82)]
energy = torch.tensor(band.content["energy"][:,band_index,:].reshape(k_points.shape[1],-1))
model_index = [i for i in range(14,34)]

print("energy.shape",energy.shape)
print("band_index:",band_index)
print("model_index:",model_index)
print("mask:",mask)
print("neighbour_list",model_input["sysinit"]["neighbour_list"])

# para = torch.randn(1,42)
para = None

start_time = time.time()
# para_train.train_emphasis_fermi(epoch = int(1e6),
#                 k_points = k_points,
#                 energy = energy,
#                 model_index=model_index,
#                 para=para,
#                 emphasis_range=2.0,conv_limit=torch.tensor(0.5),fermi_energy=0.0)
para_train.train(epoch = int(1e6),
                k_points = k_points,
                energy = energy,
                model_index=model_index,
                para=para)
end_time = time.time()
print(end_time-start_time)