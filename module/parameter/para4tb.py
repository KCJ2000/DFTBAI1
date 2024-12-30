import pickle 
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# print(sys.path)
import torch
from torch import nn
import numpy as np

from torch import exp

class ParaTB(nn.Module):
    def __init__(self,model_path:str) -> None:
        super(ParaTB,self).__init__()
        self.matrix,self.model_info,self.num_symbols,self.name,self.matrix_dim = self.load_and_check_matrix(model_path)
        self.num_para = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.para = nn.ParameterList([nn.Parameter(torch.ones(self.num_para))for _ in range(self.num_symbols)]).to(self.device)
        
        self.__have_get_TB_fix_data = False
        self.matrix_function = self.create_model_function()
        self.property_for_opt = ""
        
        
    def load_and_check_matrix(self,file_path):
        with open(file_path,"rb") as f:
            # pickletools.dis(f)
            model_dict = pickle.load(f)
        matrix = model_dict['model']
        model_info = model_dict['info']
        num_symbols = model_dict['num_symbols']
        name = model_dict['name']
        matrix = sum(matrix.values())
        matrix_dim = matrix.shape[0]
        return matrix,model_info,num_symbols,name,matrix_dim
        
        
    # def __create_ele_function(self,exp_list,formula_list):
    #     num_exp = len(exp_list)
    #     exp_list = np.array(exp_list)
    #     exp_list = torch.tensor(exp_list).to(self.device)
    #     num_para = self.num_para
    #     if num_exp != 0:
    #         formula = torch.stack([torch.sum(torch.stack([self.para[symbol]*values for symbol,values in formula_list[i].items()]),axis=0)
    #             for i in range(num_exp)
    #         ]).type(torch.complex64).to(self.device)
    #         formula = formula.transpose(dim0=1,dim1=0)
    #         def ele_function(input_data):
    #             ### input_data一定是kx,ky,kz，(3,num_points)
    #             ### 这里实际上就是formula*exp的形式只不过用矩阵的形式，进行优化
    #             input_data = input_data.type(exp_list.dtype).to(self.device)
    #             exp_term = torch.matmul(exp_list,input_data)
    #             exp_term = torch.exp(1j*exp_term).type(torch.complex64)
    #             ele = torch.matmul(formula,exp_term)
    #             return ele
    #     else:
    #         def ele_function(input_data):
    #             num_point = input_data.shape[1]
    #             return torch.zeros(num_para,num_point).to(self.device)
        
    #     return ele_function      
                
    
    # def __create_TB_function(self):
    #     matrix_dim = self.matrix.shape[0]
    #     num_para = self.num_para
    #     function_matrix = np.zeros((matrix_dim,matrix_dim),dtype=object)
    #     for i in range(matrix_dim):
    #         for j in range(matrix_dim):
    #             function_matrix[i,j] = self.__create_ele_function(self.matrix[i,j].exp_list,self.matrix[i,j].formula_list)
    #     def matrix_function(input_data):
    #         num_points = input_data.shape[1]
    #         matrix = torch.zeros(num_para,num_points,matrix_dim,matrix_dim,dtype=torch.complex64).to(self.device)
    #         for i in range(matrix_dim):
    #             for j in range(matrix_dim):
    #                 matrix[:,:,i,j] = function_matrix[i,j](input_data)
    #         return matrix
    #     return matrix_function
    
    def __get_exp(self,exp_term):
        for i in range(self.matrix_dim):
            for j in range(self.matrix_dim):
                exp_list = np.array(self.matrix[i][j].exp_list)
                num_exp = exp_list.shape[0]
                if num_exp == 0:
                    continue
                exp_term[i,j,0:num_exp,:] = torch.tensor(exp_list)
        return exp_term
    
    def __get_values_and_index(self,formula_values):
        index_matrix = []
        index_para = []
        for i in range(self.matrix_dim):
            for j in range(self.matrix_dim):
                num_exp = len(self.matrix[i][j].formula_list)
                if num_exp == 0:
                    continue
                for k in range(num_exp):
                    symbols = list(self.matrix[i][j].formula_list[k].keys())
                    values = list(self.matrix[i][j].formula_list[k].values())
                    n_term = len(self.matrix[i][j].formula_list[k])
                    for n in range(n_term):
                        formula_values[i,j,k,n] = values[n]
                        index_matrix.append([i,j,k,n])
                        index_para.append(symbols[n])
        return formula_values,torch.tensor(index_matrix),index_para                
    
    def __get_TB_fix_data(self):
        self.max_exp_term = torch.max(torch.tensor([[len(self.matrix[i][j].exp_list) for i in range(self.matrix_dim)]for j in range(self.matrix_dim)]))

        exp_term = torch.zeros(self.matrix_dim,self.matrix_dim,self.max_exp_term,3,requires_grad=False)
        self.exp_term = self.__get_exp(exp_term)
        formula_values = torch.zeros(self.matrix_dim,self.matrix_dim,self.max_exp_term,2,requires_grad=False,dtype=torch.complex64)
        self.formula_values,self.index_matrix,self.index_para = self.__get_values_and_index(formula_values)
        self.formula_values = self.formula_values.to(self.device)
        self.index_matrix = self.index_matrix.to(self.device)
        self.exp_term = self.exp_term.to(self.device).type(torch.float64)
        
        self.__have_get_TB_fix_data = True
        
    
    def __create_TB_function(self):
        if not self.__have_get_TB_fix_data:
            self.__get_TB_fix_data()
        formula_symbol = torch.zeros(self.matrix_dim,self.matrix_dim,self.max_exp_term,2,self.num_para,
                                     device=self.device)
        para_tensor = torch.stack([para for para in self.para])
        # print(formula_symbol.shape)
        # print(formula_symbol[torch.tensor([[0,0,0,0,0]])].shape)
        formula_symbol[self.index_matrix[:,0],
                       self.index_matrix[:,1],
                       self.index_matrix[:,2],
                       self.index_matrix[:,3]] = para_tensor[self.index_para,:]
        formula_symbol = formula_symbol.type(self.formula_values.dtype)
        
        def matrix_function(input_data):
            exp_term = torch.matmul(self.exp_term,input_data)
            exp_term = torch.exp(1j*exp_term)
            formula = torch.einsum("ijmnp,ijmn->ijmp",formula_symbol,self.formula_values).type(exp_term.dtype)
            matrix = torch.einsum("ijmp,ijmk->pkij",formula,exp_term)
            return matrix
    
        return matrix_function
            
    
    
    
    def create_model_function(self):
        if self.name == "tight binding hamiltonian":
            return self.__create_TB_function()
        
        
    def init_para(self,para:torch.tensor):
        should_para_shape = (self.num_para,len(self.para))
        if para.shape == should_para_shape:
            self.para = nn.ParameterList([nn.Parameter(para[:,i]) for i in range(self.num_symbols)]).to(self.device)
            self.matrix_function = self.create_model_function()
        else:
            raise AssertionError("我们需要输入的para.shape是{}，当先输入的形状是{}".format(should_para_shape,para.shape))    
    
    
    
class ParaTB_train:
    def __init__(self,model_path:str,mask_index:list=None) -> None:
        """need to init a Parameter class using model_path"""
        self.model_path = model_path
        self.mask_index = mask_index

    def init_model(self,para):
        pass
            
    def mask(self):
        pass
    
    def train(self):
        pass
        
        

if __name__ == "__main__":
    p = ParaTB("/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/Si_sps'.pkl")
    k_point = torch.tensor([[0,0,0],[0.5,0.5,0.5]],dtype=torch.float64).transpose(0,1).to("cuda:0")
    matrix = p.matrix_function(k_point)
    print(matrix.shape)
    print(matrix.dtype)
    print(matrix.requires_grad)
    eigenvalues, eigenvectors = torch.linalg.eig(matrix)
    print(eigenvalues.shape)
    print(eigenvectors.shape)
    zhengjiao = torch.matmul(torch.conj(eigenvectors.transpose(-1,-2)),eigenvectors)
    print(eigenvalues)
    print(zhengjiao.shape)
    print(torch.diagonal(zhengjiao,dim1=-1,dim2=-2))
    transed = torch.matmul(torch.conj(eigenvectors.transpose(-1,-2)),matrix)
    transed = torch.matmul(transed,eigenvectors)
    print(transed.requires_grad)
    print(eigenvalues.requires_grad)
    print(eigenvalues.is_leaf)
    print(eigenvectors.requires_grad)