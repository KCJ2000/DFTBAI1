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
                            "sys_name":"Fe_fm_d_5n",
                            "group_type":"Magnetic Group",
                            "group_name":"139.537",
                            "lattice_type":"TetrBody",
                            "lattice_parameter":{"a":1,"c":1},
                            "atompos":[[0,0,0]],
                            "magdirect":[[1,1,0]],
                            "neighbour_list":[5]
                            },
            "orbit_init":[{"orbit_list":["dxz","dz2","dx2-y2","dxy"],"spin_dict":{"dz2":1,"dxz":1,"dxy":1,"dx2-y2":1}}]}
# model = TBHamiltonian(**model_input)
# model.save_model("/data/home/kongfh/DFTBAI1/example/test_TB/Fe_fm")
# print(model.sym_hamiltonian_dict)


# mask = [1,3,7,9]
# mask = [1,3,7,9,14,16,18,20,23,25,27,29]
mask = []
device = "cuda:0"
# device = None
# device = "cpu"
# model_path = "/home/hp/users/kfh/DFTBAI1/example/test_TB/Fe_fm/Fe_fm.pkl"
# model_path = "/home/hp/users/kfh/DFTBAI1/example/test_TB/Fe_fm/Fe_fm_6neighbour.pkl"
model_path = "/data/home/kongfh/DFTBAI1/example/test_TB/Fe_fm/Fe_fm_d_5n.pkl"
para_train = Para4Band_train(model_path,
                              zero_index=mask,
                              mask_index=mask,
                              device=device)
band = Band()
band.get_data("/data/home/kongfh/DFTBAI1/example/BAND-total/Fe-fm/BAND.dat")
k_points = torch.tensor(band.content["k_vector"]).transpose(dim0=0,dim1=1)*2*torch.pi
band_index = [5,6,7,8,9]
# energy_up = torch.tensor(band.content["energy"][:,band_index,0])
# energy_dn = torch.tensor(band.content["energy"][:,band_index,1])
# model_index_up = [i for i in range(0,8,2)]
# model_index_dn = [i for i in range(0,8,2)]
energy = torch.tensor(band.content["energy"][:,band_index])
energy = energy.reshape(energy.shape[0],-1)
model_index = [i for i in range(0,10)]

print("band_index:",band_index)
print("model_index",model_index)
# print("model_index_up:",model_index_up)
# print("model_index_dn:",model_index_dn)
print("mask:",mask)
print("neighbour_list",model_input["sysinit"]["neighbour_list"])

# para = torch.randn(1,42)
para = torch.tensor(
    [[ 8.1256e-02,  3.7057e-01,  2.2687e-01,  5.1696e-01,  7.9303e-02,
         -7.7855e-01, -1.7011e-01,  3.2638e-01, -4.5020e-02, -2.7956e-01,
         -2.8774e-01,  4.9632e-01, -1.2401e-02,  1.9734e-01,  8.1611e-01,
          6.6164e-01,  3.7732e-01,  3.7052e-01, -3.9248e-03, -2.1479e-01,
          2.0670e-01, -3.1641e-02,  8.6599e-01,  8.8306e-01, -1.5197e+00,
         -1.7180e+00, -1.6516e+00,  5.3264e-01, -2.4255e-02, -1.5661e-01,
         -1.7330e-01, -3.2331e-02,  5.8041e-02,  3.0386e-01, -5.6806e-02,
         -4.4084e-02,  1.5697e-01,  7.7773e-02, -1.8502e-01,  4.1794e-02,
          3.3377e-02,  1.4405e-01, -1.3740e-01,  6.2312e-01, -4.5875e-01,
         -2.1227e-01,  3.0886e-01, -1.0603e-01,  7.3478e-01, -5.5677e-01,
         -1.0088e-02, -1.4967e-01, -1.6839e-01,  2.4405e-01,  8.4805e-01,
         -2.3499e+00, -2.3478e-02,  2.7390e-01, -1.6009e+00, -2.8153e-01,
          2.0681e-01, -1.0430e-01, -1.2178e-01,  2.3680e-01, -1.8216e-01,
          1.7472e-01,  5.6276e-01, -4.1845e-01, -9.8973e-02, -7.1913e-02,
         -3.4491e-01, -5.7619e-04,  3.9386e-02, -1.7418e-02, -9.9290e-02,
          6.5773e-02,  3.5979e-02,  6.9112e-02,  7.9551e-02,  3.6154e-03,
          7.8891e-02, -1.5444e-01,  2.1124e-02,  3.9193e-02,  3.8462e-01,
         -9.3764e-02, -2.5575e-01,  1.1463e-01, -7.8484e-02, -3.3419e-02,
         -4.9202e-03,  5.3191e-02,  3.9864e-02,  3.5827e-02, -3.3461e-02,
         -9.8405e-02,  6.2879e-02, -3.4244e-02, -5.5348e-01, -2.4808e-02,
          1.1683e+00, -6.6542e-02, -1.1253e-01, -6.7826e-02,  1.0352e+00,
          6.2760e-02,  1.1950e-01,  5.3548e-02,  8.6141e-02, -5.0882e-02,
         -9.7631e-02,  6.0814e-03, -7.1269e-03,  1.0699e-01,  1.2575e-01,
         -1.1029e-01, -4.4404e-02,  1.0459e-01, -1.8350e-02,  4.1688e-02,
          1.4266e-02, -1.8772e-02,  2.9107e-02,  1.7630e-01,  2.6331e-01,
         -1.6737e-01,  9.0365e-02, -2.2995e-01,  3.2143e-01, -4.5699e-01,
         -9.2233e-01, -2.9851e-01,  1.0164e+00,  1.5666e-01,  1.7371e-01,
          5.3352e-02, -6.1346e-01,  6.9369e-02,  4.9707e-02,  2.0553e-01,
          6.3998e-01,  8.0468e-03,  3.3798e-02, -9.9636e-02,  1.4645e-02,
          2.3186e-02,  8.9500e-03,  7.3572e-03, -6.3165e-01,  7.6900e-01,
         -3.6758e-01, -2.6402e-01,  2.4786e-01,  4.4872e-01,  5.5180e-01,
         -2.1601e-01,  7.1056e-01, -6.0306e-01,  1.3734e-01, -9.1109e-01,
          8.7525e-01, -4.4752e-01, -1.6807e+00, -3.2131e-01, -8.7709e-01,
          2.0451e-01, -6.2118e-02,  3.3276e-02, -3.4559e-02, -2.2932e-02,
         -2.1546e-02,  6.7143e-02,  7.7217e-03,  9.3798e-03, -1.8422e-02,
         -1.3481e-02, -1.3683e-02, -6.2009e-03,  3.9470e-02, -1.9269e-02,
          3.4869e-02,  4.2589e-03,  1.9704e-02, -6.2389e-02, -5.0497e-02,
          4.2362e-02, -5.2626e-03, -1.7981e-02, -5.3786e-02, -1.8977e+00,
          5.1574e-03,  6.8011e-02, -1.6439e-02, -1.6125e-01,  3.0555e-01,
          3.8112e-02, -5.9959e-02,  9.2086e-03, -1.3660e-01,  1.7072e-02]]
)

start_time = time.time()
# para_train.train_Magnetic(epoch = int(1e6),
#                 k_points = k_points,
#                 energy_up = energy_up,energy_dn = energy_dn,                
#                 model_index_up = model_index_up,model_index_dn = model_index_dn,
#                 max_iteration=5e4,
#                 para=para)
para_train.train(epoch = int(1e4),
                k_points = k_points,
                energy = energy,              
                model_index = model_index,
                max_iteration=5e4,
                para=para)
end_time = time.time()
print(end_time-start_time)