import os
import sys
file_path = sys.argv[0]
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path,"DFTBAI"))


from work.construct_model.construct_model import WorkFlow4construct_model
from DFTBAI.Hamiltonian.Hamiltonian4TB.tight_binding_hamiltonian import TBHamiltonian    
from DFTBAI.parameter.gradient_para.grad_parameter import Grad_Parameter
from test.test_band.band import Band

# model_input = {"sysinit":{
#                             "sys_name":"Si_sps'",
#                             "group_type":"Space Group",
#                             "group_name":"227",
#                             "lattice_type":"CubiFace",
#                             "lattice_parameter":{"a":1},
#                             "atompos":[[1/8,1/8,1/8]],
#                             # "magdirect":[[0,0,0]],
#                             "n_neighbour":2
#                             },
#             "orbit_init":[{"orbit_list":["s","s","px","py","pz"]}]}



# work = WorkFlow4construct_model(file_path = "G:\\DFTBAI\\DFAITB1\\example\\test_TB\\",
#                                 model_class=TBHamiltonian,
#                                 model_input = model_input)


your_path = "/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like"
file_path = os.path.join(your_path,"Si_sps'.pkl")
data_file = os.path.join(your_path,"Si_PC/BAND.dat")
init_para = [-4.2,0,0,6.6850,1.715,-8.3/4,0,0,5.7292/4,0,0,5.3749/4,0,1.715/4,4.575/4]
init_para = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
# init_para = [[13.93145976  0.          0.         26.02521817 58.14195824  1.57600395
#    0.          0.          1.62123411  0.          0.          1.17487604
#    0.          3.55260717  3.56631498]]
mask = [1,2,6,7,9,10,12]
workflow = WorkFlow4construct_model(TBHamiltonian,file_path = file_path,read_file=True,
                                        optimizer=Grad_Parameter,physics_property=Band,data_file = data_file,
                                        init_para=init_para,choose_band=[0,1,2,3,4],mask=mask,out_period=200)


