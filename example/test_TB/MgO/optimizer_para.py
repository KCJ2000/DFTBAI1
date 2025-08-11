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
#                             "sys_name":"MgO_spd_sp2_6n",
#                             "group_type":"Space Group",
#                             "group_name":"225",
#                             "lattice_type":"CubiFace",
#                             "lattice_parameter":{"a":1},
#                             "atompos":[[0,0,0],[0.5,0.5,0.5]],
#                             # "magdirect":[[1,1,0]],
#                             "neighbour_list":[7,7]
#                             },
#             "orbit_init":[{"orbit_list":["s","px","py","pz","dxy","dyz","dxz","dz2","dx2-y2"]},
#                           {"orbit_list":["s","px","py","pz","px","py","pz"]}]
#             }
# model = TBHamiltonian(**model_input)
# model.save_model("/data/home/kongfh/DFTBAI1/example/test_TB/MgO")
# print(model.sym_hamiltonian_dict)


mask = []
device = "cuda:0"
# device = None
# device = "cpu"

model_path = "/data/home/kongfh/DFTBAI1/example/test_TB/MgO/MgO_spd5_sp_6n.pkl"
# model_path = os.path.join(model_path,model_input["sysinit"]["sys_name"]+".pkl")

para_train = Para4Band_train(model_path,
                              zero_index=mask,
                              mask_index=mask,
                              device=device)
band = Band()
band.get_data("/data/home/kongfh/DFTBAI1/example/BAND-total/MgO/BAND.dat")
k_points = torch.tensor(band.content["k_vector"]).transpose(dim0=0,dim1=1)*2*torch.pi
band_index = [i for i in range(7,20)]
energy = torch.tensor(band.content["energy"][:,band_index,0].reshape(k_points.shape[1],-1))
model_index = [i for i in range(13)]

print("band_index:",band_index)
print("model_index:",model_index)
# print("mask:",mask)
# print("neighbour_list",model_input["sysinit"]["neighbour_list"])

para_input = torch.tensor([[-7.5009e-01,  4.1407e-01,  4.0449e-01, -9.8113e-01,  4.0533e+00,
          6.0720e+00,  8.8398e-02,  2.6849e-01, -2.4602e-02, -2.5670e-01,
          9.1768e-01, -1.0209e-01, -4.2827e-01,  6.5260e-01,  4.7592e-01,
          9.8886e-01,  2.5987e-01,  7.8599e-01, -3.3849e-03, -7.7059e-02,
         -2.3286e-02,  1.3711e-01,  1.4224e-01,  1.5171e-01,  2.4333e-01,
          1.6373e-01, -1.0962e-01,  1.8486e-01, -2.2847e-01,  2.8279e-01,
          4.8754e-02, -1.2920e-01,  4.2982e-01, -1.8972e-01,  7.2059e-02,
         -1.3356e-01, -4.7222e-02, -1.0511e-01, -4.4688e-01, -4.2745e-01,
         -1.4338e-01, -1.1487e-01, -5.6324e-02, -1.4407e-01, -2.9561e-02,
         -2.8715e-01, -1.0424e-02, -1.1978e-01, -1.8715e-01,  3.8145e-01,
         -5.5468e-01,  2.7207e-01, -4.3586e-02, -3.3556e-01,  4.7591e-02,
         -6.9830e-02, -1.0361e-01, -5.9329e-02,  6.3153e-02, -3.6341e-03,
         -1.4325e-01,  1.9788e-01,  2.2819e-02, -4.0429e-01,  3.9375e-01,
          4.9663e-01,  1.6903e-01,  4.8653e-01, -4.7426e-01,  1.0715e-01,
          2.1461e-02,  3.0985e-03,  6.6908e-02, -2.7942e-02,  1.3719e-02,
         -1.0764e-01,  1.5268e-01, -2.1112e-01, -4.1998e-01, -8.4206e-02,
         -5.4250e-02, -1.5155e-02,  9.8277e-02,  9.9726e-03, -3.7895e-01,
          1.3293e-01,  1.2000e-01,  3.3747e-01, -1.3864e-01,  1.7551e-01,
          9.6780e-02, -6.9354e-02,  3.0895e-01, -2.8854e-01, -3.3366e-02,
         -1.6744e-02, -3.1075e-01,  9.2583e-02, -4.0624e-02]])

start_time = time.time()
para_train.train(epoch = int(1e4),
                k_points = k_points,
                energy = energy,
                model_index=model_index,
                max_iteration = 5e4,
                para=para_input)
end_time = time.time()
print(end_time-start_time)