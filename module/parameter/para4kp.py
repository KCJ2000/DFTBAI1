import pickle 
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# print(sys.path)
import torch
from torch import nn
import numpy as np


class ParaKP(nn.Module):
    def __init__(self,model_path:str,zero_index=None,device:str=None) -> None:
        super(ParaKP,self).__init__()
        self.model,self.rotation,self.num_symbols,self.name,self.matrix_dim,self.num_term,self.basis = self.load_and_check_matrix(model_path)
        if zero_index == None:
            self.zero_index = []
        else:self.zero_index = zero_index
        self.num_para = 1
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.para = nn.ParameterList([nn.Parameter(torch.ones(self.num_para))for _ in range(self.num_symbols)]).to(self.device)
        
        
        self.__have_get_KP_fix_data = False
        self.set_zero_and_init_matrix_fuction() ### self.matrix_function() have been initialed in self.set_zero()
        self.property_for_opt = ""


    def load_and_check_matrix(self,file_path):
        with open(file_path,"rb") as f:
            # pickletools.dis(f)
            model_dict = pickle.load(f)
        
        rotation = model_dict['rotation']
        model_order = model_dict['model']
        num_var = model_dict['num_symbols']
        basis = model_dict['basis']
        name = model_dict['name']
        matrix_dim = basis.shape[-1]
        model = {}
        for k in model_order.keys():
            model.update(model_order[k])
        return model,rotation,num_var,name,matrix_dim,len(model.keys()),torch.tensor(basis)
    

    def init_para(self,para:torch.tensor):
        should_para_shape = (self.num_para,len(self.para))
        if para.shape == should_para_shape:
            self.para = nn.ParameterList([nn.Parameter(para[:,i]) for i in range(self.num_symbols)]).to(self.device)
            self.set_zero_and_init_matrix_fuction()
        else:
            raise AssertionError("我们需要输入的para.shape是{}，当先输入的形状是{}".format(should_para_shape,para.shape))   
        

    def set_zero_and_init_matrix_fuction(self):
        for index in self.zero_index:
            self.para[index] = nn.Parameter(torch.zeros(self.num_para)).to(self.device)
        self.matrix_function = self.create_model_function()


    def create_model_function(self):
        ### 防止输入另一种hamiltonian
        if self.name == "kp hamiltonian":
            return self.__create_TB_function()
        

    def __get_KP_fix_data(self):
        term_list = list(self.model.keys())
        self.k_term_index = []
        self.k_point_index = []
        self.formula_index = []
        self.formula_symbol_index = []
        self.formula_value = []
        self.basis_index = []
        for i in range(self.num_term):
            num_k = len(term_list[i])
            for j in range(num_k-1):
                self.k_term_index.append([i,j])
                self.k_point_index.append(term_list[i][j])
            self.basis_index.append(term_list[i][-1])

            formula_symbol_list = list(self.model[term_list[i]].keys())
            formula_value_list = list(self.model[term_list[i]].values())
            num_formula = len(formula_symbol_list)
            for j in range(num_formula):
                self.formula_index.append([i,j])
                self.formula_symbol_index.append(formula_symbol_list[j])
                self.formula_value.append(formula_value_list[j])
                
        self.k_point_index = torch.tensor(self.k_point_index).to(self.device)
        self.k_term_index = torch.tensor(self.k_term_index).to(self.device)
        self.formula_index = torch.tensor(self.formula_index).to(self.device)
        print("formula_index",self.formula_index)
        self.formula_symbol_index = torch.tensor(self.formula_symbol_index).to(self.device)
        self.formula_value = torch.tensor(self.formula_value).to(self.device)
        
        self.__have_get_KP_fix_data = True

    def __create_TB_function(self):
        if not self.__have_get_KP_fix_data:
            self.__get_KP_fix_data()
        max_order = torch.max(torch.tensor([len(k) for k in self.model.keys()]))
        max_formula = torch.max(torch.tensor([len(self.model[term]) for term in self.model.keys()]))
        formula = torch.zeros(self.num_term,max_formula,self.num_para,dtype=torch.complex128).to(self.device)
        para_tensor = torch.stack([para for para in self.para])
        formula[self.formula_index[:,0],
                self.formula_index[:,1]] = torch.einsum("ij,i->ij",para_tensor[self.formula_symbol_index,:],
                                                        self.formula_value)
        formula = torch.einsum("ijk->ik",formula)

        print("formula",formula.shape)
        def matrix_function(input_data):
            k_term = torch.ones(self.num_term,max_order,input_data.shape[-1])
            k_term[self.k_term_index[:,0],
                self.k_term_index[:,1]] = input_data[self.k_point_index,:]
            k_term = torch.prod(k_term,dim=1)
            k_term = k_term.type(self.basis.dtype)
            
            matrix = torch.einsum("ik,ij,imn->jkmn",k_term,formula,self.basis[self.basis_index,:,:])
            return matrix
        return matrix_function
            


if __name__ == "__main__":
    # p = ParaKP("/Volumes/KINGSTON/DFTBAI/DFTBAI_code/dftbai/example/test_KP/226.123_L4L4/226.123_L4L4_2.pkl")
    kp = ParaKP("/Volumes/KINGSTON/DFTBAI/DFTBAI_code/dftbai/example/test_KP/test1/kp.pkl")
    print(kp.model)
    input_data = torch.tensor([[0.5,0.5,0.5],[0.2,0.3,0.4]]).transpose(-1,-2)
    print(kp.matrix_function(input_data))
    print(kp.basis)