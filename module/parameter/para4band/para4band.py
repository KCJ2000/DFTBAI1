
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from torch import nn
import torch.optim as opt
# torch.autograd.set_detect_anomaly(True)

from parameter.para4tb import ParaTB,ParaTB_train


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


class Para4Band(ParaTB):
    def __init__(self, model_path: str,zero_index=None,device:str=None) -> None:
        super().__init__(model_path,zero_index,device)
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
    
    
class Para4Band_train(ParaTB_train):
    def __init__(self,model_path,mask_index = None,zero_index = None,device=None):
        super(Para4Band_train,self).__init__(model_path,mask_index,zero_index,device)
        self.para4TB = Para4Band(self.model_path,zero_index,device)
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
        eigenvector = self.para4TB.frame_trans.frame
        orth = torch.matmul(torch.conj(eigenvector.transpose(-1,-2)),
                            eigenvector)
        loss0 = torch.abs(orth-eye_matrices)
        loss1 = loss0.view(loss0.shape[0],loss0.shape[1],-1)
        loss1,_ = torch.topk(loss1,k=self.para4TB.matrix_dim,dim=2)
        loss = torch.mean(loss0) + torch.mean(loss1)      
        return loss
        
    def loss3(self,eigen_matrices,model_index,energy):
        eigens = torch.diagonal(eigen_matrices,dim1=-1,dim2=-2).type(torch.float32) ###Hermit矩阵实数特征值
        eigens = torch.sort(eigens,dim=-1)[0]
        eigens = eigens[:,:,model_index]
        
        energy = torch.sort(energy,dim=-1)[0]
        energy = energy.repeat(eigens.shape[0],1,1)
        delta_energy = torch.abs(eigens-energy)
        delta_energy = torch.flatten(delta_energy)
        loss1,_ = torch.topk(delta_energy,eigens.shape[0]*eigens.shape[1],dim=-1)
        loss = torch.mean(loss1) + torch.mean(delta_energy)
        return loss
    
    def loss(self,input_data,model_index,energy,eye_matrices):
        
        eigen_matrices = self.para4TB(input_data)
        ### loss1是为了保证特征值矩阵是对角矩阵
        loss1 = self.loss1(eigen_matrices)
        ### loss2是为了保证特征向量的正交性
        loss2 = self.loss2(eye_matrices)
        ### loss3是为了拟合能带
        loss3 = self.loss3(eigen_matrices,model_index,energy)

        if loss3>1e-2 and (torch.randn(1)<=1 or loss1+loss2<=0.2):
            return "特征值优化",loss3
        elif loss2+loss1 >0.2:
            return "正交保障",loss1+loss2
        elif loss3<=1e-2 and loss1+loss2 <= 0.2:
            return "break",-1


    def loss3_emphasis_fermi(self,eigen_matrices,model_index,energy,
                             emphasis_range=3.0,conv_limit=torch.tensor(1e-1),fermi_energy=0.0):
        """与loss3函数的区别是使用exp(-ax**2)函数做卷积，强调出fermi surface附近的能带结构，并将卷积结果作为loss输出
        fermi_energy是设定费米面的位置
        emphasis_range和conv_limit是用来确定函数形状参数a的，a的确定不大直观
        但emphasis_range是要强调的能量范围，conv_limit是卷积核在强调范围中的最小值
        """
        eigens = torch.diagonal(eigen_matrices,dim1=-1,dim2=-2).type(torch.float32) ###Hermit矩阵实数特征值
        eigens = torch.sort(eigens,dim=-1)[0]
        eigens = eigens[:,:,model_index]
        
        energy = torch.sort(energy,dim=-1)[0]
        energy = energy.repeat(eigens.shape[0],1,1)

        a = -torch.log(conv_limit)/emphasis_range**2
        conv_kernel = torch.exp(-a*(energy-fermi_energy)**2)
        
        delta_energy = torch.abs(eigens-energy)
        delta_energy = torch.mul(delta_energy,conv_kernel)
        delta_energy = torch.flatten(delta_energy)
        loss1,_ = torch.topk(delta_energy,eigens.shape[0]*eigens.shape[1],dim=-1)
        loss = torch.mean(loss1) + torch.mean(delta_energy)
        return loss


    def loss_emphasis_fermi(self,input_data,model_index,energy,eye_matrices,
                            emphasis_range=2.0,conv_limit=torch.tensor(1e-1),fermi_energy=0.0):
        eigen_matrices = self.para4TB(input_data)
        ### loss1是为了保证特征值矩阵是对角矩阵
        loss1 = self.loss1(eigen_matrices)
        ### loss2是为了保证特征向量的正交性
        loss2 = self.loss2(eye_matrices)
        ### loss3是为了拟合能带
        loss3 = self.loss3_emphasis_fermi(eigen_matrices,model_index,energy,
                                          emphasis_range,conv_limit,fermi_energy)

        if loss3>1e-2 and (torch.randn(1)<=1 or loss1+loss2<=0.2):
            return "特征值优化",loss3
        elif loss2+loss1 >0.2:
            return "正交保障",loss1+loss2
        elif loss3<=1e-2 and loss1+loss2 <= 0.2:
            return "break",-1

        
    def train(self,epoch,k_points,energy,model_index,
              para=None,
              covergence_tolerance=0.04,max_iteration=5e5):
        """训练band的过程

        Args:
            epoch (_type_): 迭代次数
            k_points (_type_): (3,num_k_points)的形式输入
            energy (_type_): band能级
            band_index (_type_): 确定我们拟合的band对应第几个特征值
        """
        num_k_points = k_points.shape[1]
        if energy.shape[0] != num_k_points and energy.shape[1] != len(model_index):
            raise ValueError("输入的能级(energy)个数应该与k点个数相等,且与model_index个数相等,应输入({},{})型tensor,现在输入{}型tensor".format(num_k_points,
                                                                                                                len(model_index),
                                                                                                                energy.shape))
        if energy.shape[1] > self.para4TB.matrix_dim:
            raise AssertionError("能带条数超出模型表达能力范围")
        
        if para!=None:
            self.para4TB.init_para(para)
            self.mask()

        k_points = k_points.to(self.para4TB.device)
        energy = energy.to(self.para4TB.device)
        
        self.para4TB.init_trans_matrix(k_points)
        
        optimizer = opt.Adam(self.para4TB.parameters(),lr=0.001)
        # print(optimizer.param_groups)               
        eye_matrices = torch.diag_embed(torch.ones(self.para4TB.num_para,
                                                   num_k_points,
                                                   self.para4TB.matrix_dim)).to(self.para4TB.device)    
        
        orth_error = 0
        eig_loss_list = []
        n_eig_loss = 0
        para_log = []
        loss_log = []
        for i in range(epoch):
            loss_type,loss = self.loss(k_points,model_index,energy,eye_matrices)
            # print(optimizer.param_groups)
            if loss_type == "正交保障":
                self.para4TB.init_trans_matrix(k_points)
                # optimizer = opt.Adam(self.para4TB.parameters(),lr=0.001, weight_decay=1e-5)
                orth_error = loss
            elif loss_type == "特征值优化":
                eig_loss_list.append(loss.item())
                n_eig_loss += 1
            elif loss_type == "break":
                params_values = torch.transpose(torch.stack([param.detach() for param in self.para4TB.para]),dim0=0,dim1=1)
                print("完成优化，收敛，参数为：")
                print(params_values)
                break
            optimizer.zero_grad()  # 清除旧的梯度 
            loss.backward()
            optimizer.step()

            if n_eig_loss > 1e4:
                eig_end = torch.tensor(eig_loss_list[n_eig_loss-1000:n_eig_loss])
                eig_mean = torch.mean(eig_end)
                eig_centered = eig_end - eig_mean
                if torch.norm(eig_centered) < covergence_tolerance:
                    print("已收敛，重新迭代")
                    params_values = torch.transpose(torch.stack([param.detach() for param in self.para4TB.para]),dim0=0,dim1=1)
                    para_log.append(params_values)
                    loss_log.append(("收敛",eig_mean))
                    random_para = torch.randn(1,self.para4TB.num_symbols)
                    self.para4TB.init_para(random_para)
                    self.mask()
                    
                    optimizer = opt.Adam(self.para4TB.parameters(),lr=0.001)
                    
                    eig_loss_list = []
                    n_eig_loss = 0
                elif n_eig_loss > max_iteration:
                    print("超出最大迭代")
                    params_values = torch.transpose(torch.stack([param.detach() for param in self.para4TB.para]),dim0=0,dim1=1)
                    para_log.append(params_values)
                    loss_log.append(("超出最大迭代次数",eig_mean))
                    random_para = torch.randn(1,self.para4TB.num_symbols)
                    self.para4TB.init_para(random_para)
                    self.mask()
                    
                    optimizer = opt.Adam(self.para4TB.parameters(),lr=0.001)
                                        
                    eig_loss_list = []
                    n_eig_loss = 0
                    
                                        
            if i %1000 == 0:
                params_values = torch.transpose(torch.stack([param.detach() for param in self.para4TB.para]),dim0=0,dim1=1)
                # print(self.para4TB.para[0].grad)
                print(i,loss_type,loss,params_values,orth_error)
        if loss_type != "break":
            print("完成共{}轮迭代".format(epoch),"推荐参数如下：")
            ### 给loss排序后输出
            loss_values = torch.tensor([loss_value for loss_type,loss_value in loss_log])
            loss_values,sort_index = torch.sort(loss_values)
            sort_index = [t_value.item() for t_value in sort_index]
            loss_log = [loss_log[index] for index in sort_index]
            para_log = [para_log[index] for index in sort_index]
            for loss,para in zip(loss_log,para_log):
                print("收敛loss为：",loss)
                print(para)
        

    def train_Magnetic(self,epoch,k_points,energy_up,model_index_up,energy_dn,model_index_dn,
              para=None,
              covergence_tolerance=0.04,max_iteration=5e5):
        """训练band的过程

        Args:
            epoch (_type_): 迭代次数
            k_points (_type_): (3,num_k_points)的形式输入
            energy (_type_): band能级
            band_index (_type_): 确定我们拟合的band对应第几个特征值
        """
        num_k_points = k_points.shape[1]
        if energy_up.shape[0] != num_k_points and energy_up.shape[1] != len(model_index_up):
            raise ValueError("输入的能级(energy_up)个数应该与k点个数相等,且与model_index个数相等,应输入({},{})型tensor,现在输入{}型tensor".format(num_k_points,
                                                                                                                len(model_index_up),
                                                                                                                energy_up.shape))
        if energy_dn.shape[0] != num_k_points and energy_dn.shape[1] != len(model_index_dn):
            raise ValueError("输入的能级(energy_dn)个数应该与k点个数相等,且与model_index个数相等,应输入({},{})型tensor,现在输入{}型tensor".format(num_k_points,
                                                                                                                len(model_index_dn),
                                                                                                                energy_dn.shape))        
        if energy_up.shape[1] > self.para4TB.matrix_dim or energy_dn.shape[1] > self.para4TB.matrix_dim:
            raise AssertionError("能带条数超出模型表达能力范围")
        
        if para!=None:
            self.para4TB.init_para(para)
            self.mask()

        k_points = k_points.to(self.para4TB.device)
        energy_up = energy_up.to(self.para4TB.device)
        energy_dn = energy_dn.to(self.para4TB.device)
        
        self.para4TB.init_trans_matrix(k_points)
        
        optimizer = opt.Adam(self.para4TB.parameters(),lr=0.001)
        # print(optimizer.param_groups)               
        eye_matrices = torch.diag_embed(torch.ones(self.para4TB.num_para,
                                                   num_k_points,
                                                   self.para4TB.matrix_dim)).to(self.para4TB.device)    
        
        orth_error = 0
        eig_loss_list = []
        n_eig_loss = 0
        para_log = []
        loss_log = []
        for i in range(epoch):
            loss_type1,loss1 = self.loss(k_points,model_index_up,energy_up,eye_matrices)
            loss_type2,loss2 = self.loss(k_points,model_index_dn,energy_dn,eye_matrices)
            if loss_type1 == "正交保障" or loss_type2 == "正交保障":
                loss_type = "正交保障"
                loss = (loss1 + loss2)/2
            elif loss_type1 == "特征值优化" or loss_type == "特征值优化":
                loss_type = "特征值优化"
                loss = (loss1 + loss2)/2
            elif loss_type1 == "break" and loss_type2 == "break":
                loss_type = "break"
                
            # print(optimizer.param_groups)
            if loss_type == "正交保障":
                self.para4TB.init_trans_matrix(k_points)
                # optimizer = opt.Adam(self.para4TB.parameters(),lr=0.001, weight_decay=1e-5)
                orth_error = loss
            elif loss_type == "特征值优化":
                eig_loss_list.append(loss.item())
                n_eig_loss += 1
            elif loss_type == "break":
                params_values = torch.transpose(torch.stack([param.detach() for param in self.para4TB.para]),dim0=0,dim1=1)
                print("完成优化，收敛，参数为：")
                print(params_values)
                break
            optimizer.zero_grad()  # 清除旧的梯度 
            loss.backward()
            optimizer.step()

            if n_eig_loss > 1e3:
                eig_end = torch.tensor(eig_loss_list[n_eig_loss-200:n_eig_loss])
                eig_mean = torch.mean(eig_end)
                eig_centered = eig_end - eig_mean
                if torch.norm(eig_centered) < covergence_tolerance:
                    print("已收敛，重新迭代")
                    params_values = torch.transpose(torch.stack([param.detach() for param in self.para4TB.para]),dim0=0,dim1=1)
                    para_log.append(params_values)
                    loss_log.append(("收敛",eig_mean))
                    random_para = torch.randn(1,self.para4TB.num_symbols)
                    self.para4TB.init_para(random_para)
                    self.mask()
                    
                    optimizer = opt.Adam(self.para4TB.parameters(),lr=0.001)
                    
                    eig_loss_list = []
                    n_eig_loss = 0
                elif n_eig_loss > max_iteration:
                    print("超出最大迭代")
                    params_values = torch.transpose(torch.stack([param.detach() for param in self.para4TB.para]),dim0=0,dim1=1)
                    para_log.append(params_values)
                    loss_log.append(("超出最大迭代次数",eig_mean))
                    random_para = torch.randn(1,self.para4TB.num_symbols)
                    self.para4TB.init_para(random_para)
                    self.mask()
                    
                    optimizer = opt.Adam(self.para4TB.parameters(),lr=0.001)
                                        
                    eig_loss_list = []
                    n_eig_loss = 0
                    
                                        
            if i %1000 == 0:
                params_values = torch.transpose(torch.stack([param.detach() for param in self.para4TB.para]),dim0=0,dim1=1)
                # print(self.para4TB.para[0].grad)
                print(i,loss_type,loss,params_values,orth_error)
        if loss_type != "break":
            print("完成共{}轮迭代".format(epoch),"推荐参数如下：")
            ### 给loss排序后输出
            loss_values = torch.tensor([loss_value for loss_type,loss_value in loss_log])
            loss_values,sort_index = torch.sort(loss_values)
            sort_index = [t_value.item() for t_value in sort_index]
            loss_log = [loss_log[index] for index in sort_index]
            para_log = [para_log[index] for index in sort_index]
            for loss,para in zip(loss_log,para_log):
                print("收敛loss为：",loss)
                print(para)
 


    def train_emphasis_fermi(self,epoch,k_points,energy,model_index,
              para=None,
              covergence_tolerance=0.04,max_iteration=5e5,
              emphasis_range=2.0,conv_limit=torch.tensor(1e-1),fermi_energy=0.0):
        """训练band的过程
            与self.train()的区别只在于用的loss是loss_emphasis_fermi
        Args:
            epoch (_type_): 迭代次数
            k_points (_type_): (3,num_k_points)的形式输入
            energy (_type_): band能级
            band_index (_type_): 确定我们拟合的band对应第几个特征值
        """
        num_k_points = k_points.shape[1]
        if energy.shape[0] != num_k_points and energy.shape[1] != len(model_index):
            raise ValueError("输入的能级(energy)个数应该与k点个数相等,且与model_index个数相等,应输入({},{})型tensor,现在输入{}型tensor".format(num_k_points,
                                                                                                                len(model_index),
                                                                                                                energy.shape))
        if energy.shape[1] > self.para4TB.matrix_dim:
            raise AssertionError("能带条数超出模型表达能力范围")
        
        if para!=None:
            self.para4TB.init_para(para)
            self.mask()

        k_points = k_points.to(self.para4TB.device)
        energy = energy.to(self.para4TB.device)
        
        self.para4TB.init_trans_matrix(k_points)
        
        optimizer = opt.Adam(self.para4TB.parameters(),lr=0.001)
        # print(optimizer.param_groups)               
        eye_matrices = torch.diag_embed(torch.ones(self.para4TB.num_para,
                                                   num_k_points,
                                                   self.para4TB.matrix_dim)).to(self.para4TB.device)    
        
        orth_error = 0
        eig_loss_list = []
        n_eig_loss = 0
        para_log = []
        loss_log = []
        for i in range(epoch):
            loss_type,loss = self.loss_emphasis_fermi(k_points,model_index,energy,eye_matrices,
                                                    emphasis_range,conv_limit,fermi_energy)
            # print(optimizer.param_groups)
            if loss_type == "正交保障":
                self.para4TB.init_trans_matrix(k_points)
                # optimizer = opt.Adam(self.para4TB.parameters(),lr=0.001, weight_decay=1e-5)
                orth_error = loss
            elif loss_type == "特征值优化":
                eig_loss_list.append(loss.item())
                n_eig_loss += 1
            elif loss_type == "break":
                params_values = torch.transpose(torch.stack([param.detach() for param in self.para4TB.para]),dim0=0,dim1=1)
                print("完成优化，收敛，参数为：")
                print(params_values)
                break
            optimizer.zero_grad()  # 清除旧的梯度 
            loss.backward()
            optimizer.step()

            if n_eig_loss > 1e4:### 这段是在判断是否收敛
                eig_end = torch.tensor(eig_loss_list[n_eig_loss-1000:n_eig_loss])
                eig_mean = torch.mean(eig_end)
                eig_centered = eig_end - eig_mean
                if torch.norm(eig_centered) < covergence_tolerance:
                    print("已收敛，重新迭代")
                    params_values = torch.transpose(torch.stack([param.detach() for param in self.para4TB.para]),dim0=0,dim1=1)
                    para_log.append(params_values)
                    loss_log.append(("收敛",eig_mean))
                    random_para = torch.randn(1,self.para4TB.num_symbols)
                    self.para4TB.init_para(random_para)
                    self.mask()
                    
                    optimizer = opt.Adam(self.para4TB.parameters(),lr=0.001)
                    
                    eig_loss_list = []
                    n_eig_loss = 0
                elif n_eig_loss > max_iteration:
                    print("超出最大迭代")
                    params_values = torch.transpose(torch.stack([param.detach() for param in self.para4TB.para]),dim0=0,dim1=1)
                    para_log.append(params_values)
                    loss_log.append(("超出最大迭代次数",eig_mean))
                    random_para = torch.randn(1,self.para4TB.num_symbols)
                    self.para4TB.init_para(random_para)
                    self.mask()
                    
                    optimizer = opt.Adam(self.para4TB.parameters(),lr=0.001)
                                        
                    eig_loss_list = []
                    n_eig_loss = 0
                    
                                        
            if i %30000 == 0:
                params_values = torch.transpose(torch.stack([param.detach() for param in self.para4TB.para]),dim0=0,dim1=1)
                # print(self.para4TB.para[0].grad)
                print(i,loss_type,loss,params_values,orth_error)
        if loss_type != "break":
            print("完成共{}轮迭代".format(epoch),"推荐参数如下：")
            ### 给loss排序后输出
            loss_values = torch.tensor([loss_value for loss_type,loss_value in loss_log])
            loss_values,sort_index = torch.sort(loss_values)
            sort_index = [t_value.item() for t_value in sort_index]
            loss_log = [loss_log[index] for index in sort_index]
            para_log = [para_log[index] for index in sort_index]
            for loss,para in zip(loss_log,para_log):
                print("收敛loss为：",loss)
                print(para)

    
    def loss_Stiefel(self,input_data,energy,knn_idx,rho=10):
        eigen_matrices = self.para4TB(input_data)

        delta1 = torch.abs(eigen_matrices-energy)
        loss1 = torch.mean(delta1) + torch.mean(torch.diagonal(delta1,dim1=-1,dim2=-2))

        rand = torch.randint(0,10,(1,),device = self.device)
        index = knn_idx[:,rand]
        delta_e = torch.diagonal(eigen_matrices - eigen_matrices[:,index] - (energy - energy[index]),dim1=-1,dim2=-2)
        loss2 = torch.mean(torch.abs(delta_e))

        loss3 = self.loss1(eigen_matrices)

        return loss1 + loss2 + 1e6*loss3, loss1, loss2, loss3



    def train_Stiefel(self,epoch,k_points,energy,model_index,
              para=None,
              covergence_tolerance=0.01,max_iteration=5e5):
        """
        使用general Stiefel manifold 的切平面公式来作为约束loss,Z^drag@M@X+X^drag@\patialM@X+X^drag@M@Z = \patial diagE
        在 ||X^drag@M@X - diagE||_L1 作为拟合
        使用Stiefel manifold Restraction回传
        """
        num_k_points = k_points.shape[1]
        if energy.shape[0] != num_k_points and energy.shape[1] != len(model_index):
            raise ValueError("输入的能级(energy)个数应该与k点个数相等,且与model_index个数相等,应输入({},{})型tensor,现在输入{}型tensor".format(num_k_points,
                                                                                                                len(model_index),
                                                                                                                energy.shape))
        if energy.shape[1] > self.para4TB.matrix_dim:
            raise AssertionError("能带条数超出模型表达能力范围")
        
        if para!=None:
            self.para4TB.init_para(para)
            self.mask()

        k_points = k_points.to(self.para4TB.device)
        energy = energy.to(self.para4TB.device)
        energy = torch.diag_embed(energy)
        
        self.para4TB.init_frame(k_points,model_index)

        optimizer = torch.optim.Adam(self.para4TB.para, lr=1e-2)
        kdm = torch.cdist(k_points.t(),k_points.t(),p=2)
        knn_idx = kdm.topk(10 + 1, largest=False, dim=-1).indices[:, 1:]  # [n, k]
        

        loss_log = []
        para_log = []
        eig_loss_list = []
        n_iter = 0
        for i in range(epoch):
            loss, loss1, loss2, loss3 = self.loss_Stiefel(k_points,energy,knn_idx,rho=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.para4TB.frame_trans.fast_cayley_retraction(n_iter,alpha=1,lr = 0.01)
            
            
            eig_loss_list.append(loss.item())
            n_iter += 1
            if n_iter > 1e3:
                eig_end = torch.tensor(eig_loss_list[n_iter-200:n_iter])
                eig_mean = torch.mean(eig_end)
                eig_centered = eig_end - eig_mean
                if torch.norm(eig_centered) < covergence_tolerance:
                    print("已收敛，重新迭代")
                    params_values = torch.transpose(torch.stack([param.detach() for param in self.para4TB.para]),dim0=0,dim1=1)
                    para_log.append(params_values)
                    loss_log.append(("收敛",eig_mean))
                    random_para = torch.randn(1,self.para4TB.num_symbols)
                    self.para4TB.init_para(random_para)
                    self.mask()      

                    optimizer = torch.optim.Adam(self.para4TB.para, lr=1e-2)

                    eig_loss_list = []
                    n_iter = 0
                elif n_iter > max_iteration:
                    print("超出最大迭代")
                    params_values = torch.transpose(torch.stack([param.detach() for param in self.para4TB.para]),dim0=0,dim1=1)
                    para_log.append(params_values)
                    loss_log.append(("超出最大迭代次数",eig_mean))
                    random_para = torch.randn(1,self.para4TB.num_symbols)
                    self.para4TB.init_para(random_para)
                    self.mask()
                    
                    optimizer = opt.Adam(self.para4TB.para,lr=0.01)
                                        
                    eig_loss_list = []
                    n_iter = 0                                


            if i%1000 == 0:
                print("frame^H@frame",torch.nonzero(torch.abs(self.para4TB.frame_trans.frame.transpose(-1,-2).conj()@self.para4TB.frame_trans.frame - torch.eye(self.para4TB.frame_trans.frame.shape[-1],device=self.para4TB.frame_trans.frame.device))>1e-5).shape)
                params_values = torch.transpose(torch.stack([param.detach() for param in self.para4TB.para]),dim0=0,dim1=1)
                print(i,"loss",loss,"loss1",loss1,"loss2",loss2,"loss3",loss3,params_values)
                


        print("完成共{}轮迭代".format(epoch),"推荐参数如下：")
        ### 给loss排序后输出
        loss_values = torch.tensor([loss_value for loss_type,loss_value in loss_log])
        loss_values,sort_index = torch.sort(loss_values)
        sort_index = [t_value.item() for t_value in sort_index]
        loss_log = [loss_log[index] for index in sort_index]
        para_log = [para_log[index] for index in sort_index]
        for loss,para in zip(loss_log,para_log):
            print("收敛loss为：",loss)
            print(para)       
        

from physics_property.band.band_data_in import BandDataIn
import time
if __name__ == "__main__":
    mask = [1,2,6,7,9,10,12]
    # mask = []
    para_train = Para4Band_train("/data/home/kongfh/DFTBAI1/example/test_TB/Si_like/Si_PC/Si_sps'.pkl",
                                 zero_index=mask,
                              mask_index=mask)
    band_in = BandDataIn("/data/home/kongfh/DFTBAI1/example/test_TB/Si_like/Si_PC/BAND.dat")
    k_points = torch.tensor(band_in.content["k_vector"]).transpose(dim0=0,dim1=1)*2*torch.pi
    band_index = [1,2,3,4]
    energy = torch.tensor(band_in.content["energy"][:,band_index,0])
    # print(band_in.content)
    print(energy.shape)
    model_index = [1,2,3,4]
    # para = torch.tensor([[1,0,0,1,1,1,0,0,1,0,0,1,0,1,1]],dtype=torch.float32)
    para = torch.randn(1,15)

    # para = torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],dtype=torch.float32)
    # para = torch.tensor([[-4.2,0,0,6.6850,1.715,-8.3/4,0,0,-5.7292/4,0,0,-5.3749/4,0,1.715/4,4.575/4]],dtype=torch.float32)
    # para = torch.tensor([[-3.6794,  0.0000,  0.0000,  6.6766,  1.5710, -1.9928,  0.0000,  0.0000,-1.4614,  0.0000,  0.0000, -1.1135,  0.0000,  0.4030,  1.2013]])
    # para = torch.tensor([[-4.0771,  0.0000,  0.0000,  3.6879,  1.2270, -1.6666,  0.0000,  0.0000,  -1.4248,  0.0000,  0.0000, -0.4334,  0.0000,  0.3119,  1.1078]])
    # para = torch.tensor([[-2.6193,  0.0000,  0.0000,  3.8806,  2.3433, -0.9512,  0.0000,  0.0000,  -1.5930,  0.0000,  0.0000, -0.0275,  0.0000, -0.0144,  1.5422]])### 无法描述band gap
    start_time = time.time()
    para_train.train_Stiefel(epoch = int(1e4),
                     k_points = k_points,
                     energy = energy,
                     model_index=model_index,
                     para=para)
    end_time = time.time()
    print(end_time-start_time)
    
    