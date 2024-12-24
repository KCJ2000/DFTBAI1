
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from torch import nn
import torch.optim as opt
torch.autograd.set_detect_anomaly(True)

from parameter.para4tb import ParaTB,ParaTB_train


class Para4Band(ParaTB):
    def __init__(self, model_path: str) -> None:
        super().__init__(model_path)
        self.property_for_opt = "band"
        self.have_init_trans = False
        
    def init_trans_matrix(self,input_data,para=None):
        if para != None:
            self.init_para(para)
        matrices = self.matrix_function(input_data)
        _ , eigenvector = torch.linalg.eig(matrices)
        self.trans_matrix = nn.Parameter(eigenvector.detach())
        self.have_init_trans = True
    
    
    def forward(self,input_data):
        self.matrix_function = self.create_model_function()
        matrices = self.matrix_function(input_data)
        if self.have_init_trans:
            eigen_matrices = torch.matmul(torch.conj(self.trans_matrix.transpose(-1,-2)),matrices)
            eigen_matrices = torch.matmul(eigen_matrices,self.trans_matrix)
        else:
            self.init_trans_matrix(input_data)
            eigen_matrices = self.forward(input_data)
        return eigen_matrices
    
    
class Para4Band_train(ParaTB_train):
    def __init__(self,model_path,mask_index = None):
        super(Para4Band_train,self).__init__(model_path,mask_index)
        self.para4TB = Para4Band(self.model_path)
        self.mask()

    def init_para(self,para):
        self.para4TB.init_para(para)
        
    def mask(self):
        if self.mask_index:
            for index in self.mask_index:
                self.para4TB.para[index].requires_grad = False

    def loss1(self,eigen_matrices):
        eigens = torch.diagonal(eigen_matrices,dim1=-1,dim2=-2)
        eigen_diag = torch.diag_embed(eigens)
        loss0 = torch.abs(eigen_matrices-eigen_diag)
        loss1 = loss0.view(loss0.shape[0],loss0.shape[1],-1)
        loss1,_ = torch.topk(loss1,k=self.para4TB.matrix_dim,dim=2)
        loss = torch.mean(loss0) + torch.mean(loss1)
        return loss
    
    def loss2(self,eye_matrices):
        eigenvector = self.para4TB.trans_matrix
        orth = torch.matmul(torch.conj(eigenvector.transpose(-1,-2)),
                            eigenvector)
        loss0 = torch.abs(orth-eye_matrices)
        loss1 = loss0.view(loss0.shape[0],loss0.shape[1],-1)
        loss1,_ = torch.topk(loss1,k=self.para4TB.matrix_dim,dim=2)
        loss = torch.mean(loss0) + torch.mean(loss1)      
        return loss
        
    def loss3(self,eigen_matrices,band_index,energy):
        eigens = torch.diagonal(eigen_matrices,dim1=-1,dim2=-2).type(torch.float32) ###Hermit矩阵实数特征值
        eigens = torch.sort(eigens,dim=-1)[0]
        
        eigens = eigens[:,:,band_index]
        energy = energy.repeat(eigens.shape[0],1,1)
        loss = torch.mean(torch.abs(eigens-energy))
        return loss
    
    def loss(self,input_data,band_index,energy,eye_matrices):
        
        eigen_matrices = self.para4TB(input_data)
        ### loss1是为了保证特征值矩阵是对角矩阵
        loss1 = self.loss1(eigen_matrices)
        ### loss2是为了保证特征向量的正交性
        loss2 = self.loss2(eye_matrices)
        ### loss3是为了拟合能带
        loss3 = self.loss3(eigen_matrices,band_index,energy)
        if loss2<1e-2 and loss3+loss1 <= 1e-2:
            return "break",-1
        elif loss3+loss1 >= loss2:
            return "特征值优化",loss3 + loss1
        elif loss2 >= loss3 + loss1:
            return "正交优化",loss2

        
    def train(self,epoch,k_points,energy,band_index,para=None):
        """训练band的过程

        Args:
            epoch (_type_): 迭代次数
            k_points (_type_): (3,num_k_points)的形式输入
            energy (_type_): band能级
            band_index (_type_): 确定我们拟合的band对应第几个特征值
        """
        num_k_points = k_points.shape[1]
        if energy.shape[0] != num_k_points and energy.shape[1] != len(band_index):
            raise ValueError("输入的能级(energy)个数应该与k点个数相等,且与band_index个数相等,应输入({},{})型tensor,现在输入{}型tensor".format(num_k_points,
                                                                                                                len(band_index),
                                                                                                                energy.shape))
        if max(band_index)-min(band_index) > self.para4TB.matrix_dim:
            raise AssertionError("能带条数超出模型表达能力范围")
        
        if para!=None:
            self.para4TB.init_para(para)
            self.mask()

        k_points = k_points.to(self.para4TB.device)
        energy = energy.to(self.para4TB.device)
        center = int(self.para4TB.matrix_dim/2)
        band_index = torch.tensor(band_index)
        band_index_center = int(torch.sum(band_index)/band_index.shape[0])
        band_index = band_index - band_index_center + center -3
        print(band_index)
        self.para4TB.init_trans_matrix(k_points)
        
        optimizer = opt.Adam(self.para4TB.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
                           
        eye_matrices = torch.diag_embed(torch.ones(self.para4TB.num_para,
                                                   num_k_points,
                                                   self.para4TB.matrix_dim)).to(self.para4TB.device)    
        
        total_train_band = 0
        loss_band = 0
        for i in range(epoch):
            loss_type,loss = self.loss(k_points,band_index,energy,eye_matrices)
            # print(optimizer.param_groups)
            optimizer.zero_grad()  # 清除旧的梯度 
            # loss.backward(retain_graph=True)       # 计算新的梯度
            loss.backward()
            # for name, param in self.para4TB.named_parameters():
            #     if param.grad is not None:
            #         print(name, 'has gradient', param.grad)
            #     else:
            #         print(name, 'does not have gradient')
            optimizer.step()
            if loss_type == "正交优化":
                loss_band = loss
                total_train_band += 1
            elif loss_type == "break":
                break
            if i %1000 == 0:
                params_values = torch.transpose(torch.stack([param.detach() for param in self.para4TB.para]),dim0=0,dim1=1)
                # print(self.para4TB.para[0].grad)
                print(i,loss_type,loss,params_values,total_train_band,loss_band)
            
            
            
from physics_property.band.band_data_in import BandDataIn

if __name__ == "__main__":
    mask = [1,2,6,7,9,10,12]
    para_train = Para4Band_train("/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/Si_sps'.pkl",
                              mask)
    band_in = BandDataIn("/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/BAND.dat")
    k_points = torch.tensor(band_in.content["k_vector"]).transpose(dim0=0,dim1=1)*2*torch.pi
    band_index = [1,2,3,4]
    energy = torch.tensor(band_in.content["energy"][:,band_index])
    para = torch.tensor([[1,0,0,1,1,1,0,0,1,0,0,1,0,1,1]],dtype=torch.float32)
    # para = torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],dtype=torch.float32)
    para = torch.tensor([[-4.2,0,0,6.6850,1.715,-8.3/4,0,0,-5.7292/4,0,0,-5.3749/4,0,1.715/4,4.575/4]],dtype=torch.float32)
    # para = torch.tensor([[-3.6794,  0.0000,  0.0000,  6.6766,  1.5710, -1.9928,  0.0000,  0.0000,-1.4614,  0.0000,  0.0000, -1.1135,  0.0000,  0.4030,  1.2013]])
    para = torch.tensor([[-4.0771,  0.0000,  0.0000,  3.6879,  1.2270, -1.6666,  0.0000,  0.0000,  -1.4248,  0.0000,  0.0000, -0.4334,  0.0000,  0.3119,  1.1078]])
    para_train.train(epoch = int(1e7),
                     k_points = k_points,
                     energy = energy,
                     band_index=band_index,
                     para=para)
    
    
    