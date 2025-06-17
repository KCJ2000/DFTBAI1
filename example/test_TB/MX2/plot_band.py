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

band.get_data("/home/hp/users/kfh/DFTBAI1/example/BAND-total/MnTe2-metal/BAND.dat")
#band.get_data("/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/BAND.dat")
print(band.content['k_vector'].shape)
print(band.content["energy"].shape)

band_index = [i for i in range(70,84)]
model_index = [i for i in range(2,30)]

# band.plot_data("/home/hp/users/kfh/DFTBAI1/example/test_TB/MgO/DFT_MgO.png",band_index)

para_input = torch.tensor([[-1.2843e-01, -1.1571e-01,  9.3130e-02, -2.6837e-01,  2.4044e-01,
         -1.2707e-01,  1.4156e-01,  4.5186e-02,  1.1723e-01,  3.3617e-02,
          9.1755e-02,  5.3241e-02,  4.1893e-02,  2.5911e-02,  5.9065e-02,
          2.6330e-01,  2.0798e-02,  1.5248e-01,  1.1457e-02,  3.8161e-02,
         -1.3172e-02,  1.4679e-01,  7.8807e-03, -2.6471e-02,  2.0921e-01,
          1.2058e-01,  5.9236e-02, -5.0353e-02,  7.5247e-03, -1.3003e-04,
          7.4823e-02,  3.2312e-01, -4.7006e-02,  2.6125e-01, -1.3079e-02,
          1.6049e-01,  2.2978e-01, -1.1145e-01, -1.0346e-03,  2.3511e-01,
         -1.8769e-02, -4.1516e-02, -6.8642e-02, -3.3122e-02, -1.2684e-01,
          1.8288e-01, -1.2646e-01,  1.0782e-01,  1.9212e-01,  2.8314e-01,
         -2.7613e-02,  2.5902e-02,  1.3589e-01,  1.5978e-01,  2.1693e-01,
          1.4788e-01,  1.0001e-01,  1.1862e-01,  7.2531e-03,  1.7646e-01,
          7.6143e-02,  5.0246e-02, -2.4461e-02, -5.7035e-02,  1.8375e-01,
          8.9583e-02,  1.1979e-01,  5.2960e-02,  2.3637e-01,  6.1515e-02,
         -2.1511e-02,  2.7029e-02, -1.0597e-01,  1.9653e-01,  1.3326e-01,
         -2.1923e-01, -3.9193e-02, -1.4015e-01,  7.9307e-02,  1.2832e-01,
         -7.9383e-03,  1.6882e-01,  1.4775e-01, -8.0516e-02,  4.4787e-02,
          1.0594e-02,  2.2573e-01,  1.8769e-01, -3.1210e-02, -3.0371e-02,
         -3.7884e-02,  1.4041e-02,  8.8379e-02,  1.2043e-01,  6.4006e-03,
         -1.7882e-02,  2.2371e-01,  7.8838e-02, -9.1546e-02,  4.8672e-02,
         -1.1713e-02, -4.4197e-02, -8.0368e-02,  6.3208e-02,  4.5590e-02,
         -6.7215e-02, -1.5391e-02,  7.2848e-02, -5.1940e-02, -2.4485e-02,
          1.4468e-01, -2.6128e-02, -3.9308e-02, -1.5585e-02,  1.5868e-02,
          4.1414e-02,  1.1812e-01, -2.0978e-01,  2.2289e-01,  1.2574e-02,
          6.2849e-02,  2.1419e-01,  1.7968e-01, -6.5612e-02, -4.6017e-02,
          5.9836e-02, -1.2046e-02,  1.8426e-01,  9.9172e-02,  5.0872e-02,
          1.9751e-01,  3.1780e-02,  1.2328e-01,  1.1030e-02,  6.2422e-02,
         -2.4947e-03,  1.1762e-01, -1.9723e-02,  1.9250e-01, -7.3016e-02,
          1.3432e-01, -1.6320e-01,  6.8306e-02,  1.4188e-01, -2.2367e-02,
         -1.0193e-02,  1.2897e-01,  1.2321e-01,  3.9826e-02, -2.7425e-02,
          2.9092e-01,  5.1033e-02,  1.0621e-01, -1.7964e-02,  7.2795e-02,
          6.4422e-03,  9.8267e-02, -1.2842e-01,  1.7577e-01,  4.4457e-02,
          7.3429e-02,  1.0169e-01,  6.0330e-02,  1.4311e-01, -4.3793e-02,
          6.0120e-02, -5.4539e-02, -3.9022e-02,  1.4295e-01, -1.1230e-01,
          2.1828e-01, -9.7270e-02,  7.3755e-02,  1.5010e-01,  8.8285e-02,
          3.0122e-02, -2.2391e-02,  3.6572e-02, -1.8123e-02, -9.9533e-02,
          4.4489e-02,  1.6591e-01, -1.1414e-02,  5.1538e-03,  3.8799e-02,
          5.1065e-02,  5.3482e-02, -1.3049e-01, -9.2542e-02,  1.6615e-01,
          1.2818e-01, -3.2846e-02,  3.4938e-02,  5.8843e-02,  2.7278e-01,
         -1.5650e-02,  9.3489e-02,  1.6899e-01,  6.9447e-02,  7.5425e-02,
          9.0498e-03, -1.5803e-01,  7.2303e-02,  1.6090e-01]])



threshold = 1e-2
para_input = torch.where(abs(para_input) < threshold, torch.tensor(0.0), para_input)

band.init_calculate_model("/home/hp/users/kfh/DFTBAI1/example/test_TB/MX2/MX2.pkl",
                              para = para_input
                              )
band.plot_model(band.content['k_vector'],save_path="/home/hp/users/kfh/DFTBAI1/example/test_TB/MX2/calculate_band.png",
                    select_band=model_index
                    )

print(para_input.shape)
band.plot_compare(band.content['k_vector'],save_path="/home/hp/users/kfh/DFTBAI1/example/test_TB/MX2/compare.png",
                  model_index=model_index,band_index=band_index,title="MnTe2")


