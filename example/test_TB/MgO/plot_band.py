import os
import sys
file_path = sys.argv[0]
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path,"module"))
import torch
import numpy as np


from module.physics_property.band.band import Band

band = Band()

band.get_data("/home/hp/users/kfh/DFTBAI1/example/BAND-total/MgO/BAND.dat")
#band.get_data("/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/BAND.dat")
print(band.content['k_vector'].shape)
print(band.content["energy"].shape)

band_index = [12,13,14,15,16,17,18]
model_index = [0,1,2,3,4,5,6]

# band.plot_data("/home/hp/users/kfh/DFTBAI1/example/test_TB/Fe_fm/DFT_Fe_fm.png",band_index)

### MgO_sp_sp
para_input = torch.tensor([[ 5.7728e-01,  2.5660e+00,  4.7586e+00,  6.2387e+00, -2.6187e-01,
          2.7352e-01, -3.8152e-01,  7.6249e-01, -5.1464e-01,  1.4123e+00,
         -2.5034e-02, -7.0609e-02, -1.0507e+00,  2.1496e+00,  2.0091e-01,
         -3.1234e-01,  3.7666e-03,  1.8798e-01, -1.4462e-01,  6.1336e-01,
         -5.2591e-01, -3.3116e-01, -1.4414e-01, -1.1983e-01,  2.1991e-01,
         -1.2758e-01,  1.9792e-01,  9.7995e-02,  1.3952e-01,  2.0120e-01,
          4.4860e-02,  6.2433e-02,  1.7590e-02, -8.7602e-03, -3.5072e-02,
          1.7091e-01,  1.1811e-02,  5.3018e-02,  1.2089e-01, -4.3278e-02,
          5.5369e-01,  2.5370e-01,  3.7853e-01, -5.2807e-02, -2.0152e-02,
         -5.6801e-01]])
### MgO_spd_sp_5n
para_input = torch.tensor([[ 6.1841e+00,  5.9635e+00,  7.8594e+00,  3.7963e+00,  4.5764e+00,
          5.8960e-04,  8.9209e-01, -8.1877e-01, -4.7850e-02,  7.1311e-01,
          9.6426e-01,  5.0398e-01,  8.1556e-01, -1.4169e+00,  1.0631e+00,
         -1.4817e+00,  2.9926e-01, -4.0030e-01,  5.2254e-01, -1.5818e-01,
          5.7110e-01,  1.0692e-01, -2.6191e-01,  2.6564e-01, -3.6237e-01,
         -1.1480e-01, -2.5826e-01,  5.6934e-01, -9.2133e-02,  1.7669e-01,
          1.4353e-01, -1.0823e-01, -1.2368e-01, -1.2452e-01,  2.1343e-01,
          1.8870e-01, -2.2124e-01,  6.1457e-01, -2.8098e-01, -4.8066e-01,
          3.8961e-01,  9.8559e-02, -2.4697e-01, -3.6975e-01,  1.2231e-01,
         -1.6610e-01,  3.1768e-01,  1.1331e-01, -2.8806e-01,  1.0521e-01,
         -5.8017e-02, -1.8455e-01, -1.9344e-01,  2.7073e-01, -1.1501e-01,
          7.0095e-02,  3.0766e-01, -1.1584e-01,  5.9402e-02, -7.8530e-02,
         -2.7015e-01,  3.3403e-03,  5.1395e-01, -1.7189e-02, -4.8393e-01,
         -3.4937e-01, -3.9461e-01,  2.3544e-01]])

threshold = 1e-2
para_input = torch.where(abs(para_input) < threshold, torch.tensor(0.0), para_input)

band.init_calculate_model("/home/hp/users/kfh/DFTBAI1/example/test_TB/MgO/MgO_spd_sp.pkl",
                              para = para_input
                              )
# band.plot_model(band.content['k_vector'],save_path="/home/hp/users/kfh/DFTBAI1/example/test_TB/Fe_fm/calculate_band.png",
#                     select_band=model_index
#                     )

print(para_input.shape)
band.plot_compare(band.content['k_vector'],save_path="/home/hp/users/kfh/DFTBAI1/example/test_TB/MgO/compare.png",
                  model_index=model_index,band_index=band_index)