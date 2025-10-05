import os
import sys
file_path = sys.argv[0]
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path,"module"))
import pickle

import torch
from torch import nn
import torch.optim as opt

from Manager.Manager import manager
from parameter.para4tb import ParaTB


class Stiefel_Frame(nn.Module):
    """为特征向量形成的Frame始终满足X^drag@X = I_p,X belongs to St(n,p)
        n是模型维数,p是需要拟合成的能带的个数
    """
    def __init__(self,eigenvector):
        super().__init__()
        # print(torch.nonzero(torch.abs(eigenvector.transpose(-1,-2).conj()@eigenvector - torch.eye(eigenvector.shape[-1]))>1e-5))
        if torch.nonzero(torch.abs(eigenvector.transpose(-1,-2).conj()@eigenvector - torch.eye(eigenvector.shape[-1],device=eigenvector.device))>1e-5).shape != torch.Size([0, 4]):
            raise ValueError("eigenvector初始化后并不正交，请正交后重新输入")
        self.frame = eigenvector.detach().clone().requires_grad_(True)
        self.M = torch.zeros_like(self.frame) ### 用于Cayley Adam算法

    def QR_retraction(self):
        Q,R = torch.linalg.qr(self.frame)
        # print("Q^H@Q",torch.nonzero(torch.abs(Q.transpose(-1,-2).conj()@Q-torch.eye(4))>1e-4))
        self.frame.data = Q

    @torch.no_grad()
    def fast_cayley_retraction(self,n_step,alpha,lr,beta1=0.9,beta2=0.999,eps=1e-8,q=0.5,max_iter=3):
        G = self.frame.grad
        n_step += 1
        self.M = beta1*self.M + (1-beta1)*G
        # if n_step%1 == 0:
        #     print("v_h",v_h,"r",r,"v",self.v)
        W_h = self.M@self.frame.transpose(-1,-2).conj() - 0.5*self.frame@(self.frame.transpose(-1,-2).conj()@self.M@self.frame.transpose(-1,-2).conj())
        W = (W_h - W_h.transpose(-1,-2).conj())
        self.M = W@self.frame
        alpha = min(lr,2*q/(eps+torch.norm(W)))
        Y = self.frame - alpha*self.M
        for i in range(max_iter):
            Y = self.frame - alpha/2*W@(self.frame+Y)
        del self.frame
        self.frame = Y.detach().clone().requires_grad_(True)
        

    def forward(self,matrices):
        return Stiefel_Frame_Function.apply(matrices,self.frame)
    

class Stiefel_Frame_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx,matrices,frame):
        ctx.save_for_backward(matrices,frame)
        eigen_matrices = torch.matmul(frame.conj().transpose(-1,-2),matrices)
        eigen_matrices = torch.matmul(eigen_matrices,frame)
        return eigen_matrices

    @staticmethod
    def backward(ctx, grad_ouput):
        matrices, frame = ctx.saved_tensors
        sym = (grad_ouput + grad_ouput.transpose(-1,-2).conj())/2
        grad_G = 2*matrices@frame@sym
        grad_M = frame@sym@frame.transpose(-1,-2).conj()
        ### tangent space projection
        # grad_f = grad_G - frame@(frame.transpose(-1,-2).conj()@grad_G+grad_G.transpose(-1,-2).conj()@frame)/2
        return grad_M, grad_G
    

class Eigen_Trans(nn.Module):
    def __init__(self, eigenvector):
        super().__init__()
        self.frame = nn.Parameter(eigenvector)
    def forward(self,matrices):
        eigen_matrices = torch.matmul(self.frame.conj().transpose(-1,-2),matrices)
        eigen_matrices = torch.matmul(eigen_matrices,self.frame)
        return eigen_matrices


class ParaTB4Band(ParaTB):
    def __init__(self, model_path: str,zero_index=None,device:str=None) -> None:
        super(ParaTB4Band,self).__init__(model_path,zero_index,device)
        self.property_for_opt = "band"
        self.have_init_trans = False
        self.set_zero_and_init_matrix_fuction()
        
    def init_frame(self,input_data,model_index,para=None):
        if para != None:
            self.init_para(para)
        matrices = self.matrix_function(input_data)
        eigenvalue, eigenvector = torch.linalg.eigh(matrices)
        eigenvector = eigenvector.detach()
        eigenvalue,idx = torch.sort(eigenvalue,dim=-1)
        eigenvector = torch.gather(eigenvector,dim=-1,index=idx.unsqueeze(-1).expand_as(eigenvector).transpose(-1,-2))
        # print("ss",torch.nonzero(torch.abs(eigenvector.transpose(-1,-2).conj()@eigenvector)-torch.eye(10)>1e-3).shape)
        self.frame_trans = Stiefel_Frame(eigenvector[:,:,:,model_index])
        self.have_init_trans = True

    def init_trans_matrix(self,input_data,para=None):
        if para != None:
            self.init_para(para)
        matrices = self.matrix_function(input_data)
        _ , eigenvector = torch.linalg.eigh(matrices)
        eigenvector = eigenvector.detach()
        self.frame_trans = Eigen_Trans(eigenvector)
        self.have_init_trans = True
    
    
    def forward(self,input_data):
        self.matrix_function = self.create_model_function()
        matrices = self.matrix_function(input_data)
        if self.have_init_trans:
            eigen_matrices = self.frame_trans(matrices)
        else:
            self.init_trans_matrix(input_data)
            eigen_matrices = self.forward(input_data)
        return eigen_matrices
    



class para4band_manager(manager):
    def __init__(self,model_path: str,zero_index=None,device:str=None):
        super(para4band_manager).__init__()
        self.manager_name = "para4band_manager"
        self.tool = self.select_tool(model_path,zero_index,device)

    def select_tool(self,model_path,zero_index,device):
        with open(model_path, 'rb') as file:
            data = pickle.load(file)
        name = data['name']
        if name == "tight binding hamiltonian":
            return ParaTB4Band(model_path,zero_index,device)
        elif name == "kp model":
            return 0
