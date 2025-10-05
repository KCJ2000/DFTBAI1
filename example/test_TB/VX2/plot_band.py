import os
import sys
file_path = sys.argv[0]
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path,"module"))
import torch


from module.physics_property.band.band import Band

band = Band()
# para_input = torch.tensor([[-4.6034,  0.0000,  0.0000,  5.1515,  1.3112,  1.9006,  0.0000,  0.0000,  1.2579,  0.0000,  0.0000,  0.9963,  0.0000, -0.2933, -1.1782]])

# band.init_calculate_model("/Volumes/KINGSTON/DFTBAI/DFTBAI_code/dftbai/example/test_TB/Si_like/Si_PC/Si_sps'.pkl",
#                               para = para_input
#                               )
band.get_data("/Volumes/KINGSTON/DFTBAI/DFTBAI_code/dftbai/example/test_TB/Si_like/Si_PC/BAND.dat")
print(band.content['k_vector'].shape)
print(band.content["energy"].shape)
select_band = [1,2,3,4]
# band.plot_model(band.content['k_vector'],save_path="/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/calculate_band.png",
#                     select_band=select_band
#                     )

band.plot_compare(input_data=band.content['k_vector'],
                  save_path="/Volumes/KINGSTON/DFTBAI/DFTBAI_code/dftbai/example/test_TB/Si_like/Si_PC/Si_s2p_2n(without mask).eps",
                  model_index=select_band,
                  band_index=select_band,title="Si_s2p_2n(without mask)")