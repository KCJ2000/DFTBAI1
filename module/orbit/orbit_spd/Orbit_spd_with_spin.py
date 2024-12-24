import os
import sys
file_path = sys.argv[0]
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))

from orbit.Orbit import orbit
from orbit.orbit_spd.Orbit_spd_without_spin import orbit_spd_without_spin

import sympy as sp
import numpy as np
from numpy import exp,cos,sin
from sympy import simplify,Matrix,symbols,total_degree,expand,parse_expr

import json
from scipy.spatial.transform import Rotation



class orbit_spd_with_spin(orbit):
    def __init__(self,orbit_list = None,spin_dict = None):
        """orbit_spd_with_spin类是考虑自旋的轨道类,与Magnetic Space Group和Spin Space Group相匹配
            该类需要依赖orbit_spd_without_spin类,对非自旋部分进行分析,方便代码简洁与代码复用
            这样orbit_spd_with_spin只需要考虑自旋的部分即可
            
        Args:
            orbit_list:用于记录一个wyckoff position上的一套轨道
            spin_dict:用于记录对应的轨道上面要不要加自旋
            self.orbit_table 是从文件中读取的spd轨道信息
        """
        folder_path = os.path.dirname(__file__)
        with open(os.path.join(folder_path,"orbit_rep_table.json")) as f:
            self.orbit_table = json.load(f)    
        
        self.no_spin_orbit = orbit_spd_without_spin(orbit_list)
        
        self.check_input(orbit_list,spin_dict)
        self.orbit_list = orbit_list
        self.spin_dict = spin_dict
        self.orbit_num_dict = self.no_spin_orbit.orbit_num_dict
        self.orbit_trans_dict = self.no_spin_orbit.orbit_trans_dict
        self.order_dict = self.no_spin_orbit.order_dict
        
        self.have_complete_list = []
        
    def check_input(self,orbit_list,spin_dict):
        if not spin_dict:
            raise AssertionError("忘记输入是否考虑轨道自旋了，请输入。如果不需要轨道自旋，请选择orbit_spd_without_spin类。")
        
        spin_orbit = spin_dict.keys()
        orbit_equal1 = all(spin_orbit0 in orbit_list for spin_orbit0 in spin_orbit)
        orbit_equal2 = all(orbit in spin_orbit for orbit in orbit_list)
        if not (orbit_equal1 and orbit_equal2):
            raise AssertionError("orbit list{}和spin dict{}中的轨道应该是一样的,请重新输入".format(orbit_list,spin_orbit))
        
        
    def complete_basis(self,orbit_rot_dict,rot_name):
        """需要用到无自旋的self.complete_basis函数，得到在某个旋转操作下相互关联的轨道

        Args:
            orbit_rot_dict (_type_): 我们通过同一个阶的一套完备基得到了一个基于这套完备基的rotation matrix
                            也就是输入的orbit_rot_dict中的内容，完备基的顺序储存在self.orbit_trans_dict中
            rot_name (str): 已经完成对称性测试的旋转操作名称
                            !!!一定要确保名称和得到orbit_rot_dict的操作对应
        """
        related_dict = self.no_spin_orbit.complete_basis(orbit_rot_dict,rot_name)
        self.orbit_list = self.no_spin_orbit.orbit_list
        self.orbit_num_dict = self.no_spin_orbit.orbit_num_dict
        
        need_order = related_dict.keys()
        for order in need_order:
            for relate_orbit_list in related_dict[order]:
                if len(relate_orbit_list) == 1:
                    if self.spin_dict[relate_orbit_list[0]] > self.orbit_num_dict[relate_orbit_list[0]]:
                        raise AssertionError("该轨道{}的个数 {} 不应该小于该轨道所需要自旋的数目 {},请重新输入".format(relate_orbit_list[0],
                                                                                        self.orbit_num_dict[relate_orbit_list[0]],
                                                                                        self.spin_dict[relate_orbit_list[0]]))
                else:
                    orbit_num = self.orbit_num_dict[relate_orbit_list[0]]
                    spin_num = [self.spin_dict[relate_orbit] for relate_orbit in relate_orbit_list if relate_orbit in self.spin_dict.keys()]
                    if set(spin_num) != 1:
                        max_num = max(spin_num)
                        if max_num > orbit_num:
                            raise AssertionError("轨道基组{}中有轨道的个数{}少于该轨道所需要的自旋数目{},请重新输入".format(relate_orbit_list,
                                                                                               orbit_num,max_num))
                        else:
                            for relate_orbit in relate_orbit_list:
                                if relate_orbit not in self.spin_dict.keys():
                                    self.spin_dict[relate_orbit] = max_num
                                    print("轨道{}没设置自旋，根据对称性，设置自旋轨道数为{}".format(relate_orbit,max_num))
                                elif self.spin_dict[relate_orbit]<max_num:
                                    print("轨道{}自旋轨道个数因为对称性原因不足，当增加到{}才足够".format(relate_orbit,max_num))
                                    self.spin_dict[relate_orbit] = max_num
        self.n_orbit = self.no_spin_orbit.n_orbit
        self.have_complete_list = self.no_spin_orbit.have_completed_list                       
        
        
    def so3_2_su2(self,rot_matrix):
        """
        这个函数还是有问题,先往下写24.6.28
        严重怀疑：
            Mathematica 的EulerAngles函数和python的scipy中的as_euler函数得到的结果
            在结果中存在pi的时候,对于SO3而言,pi和-pi是一样的
            但是对于SU2而言,sin(beta/2)正好差了个正负号,所以旋转矩阵的结果也经常差了正负号,
            不过不影响最后Hamiltonian的旋转结果，可以接着往下写
            
        
            rot_matrix是实空间正交矩阵(Jones 表示下),这是个SO3群元的3维表示
            我们要得到SU2群元的2*2酉表示
        """
        if np.linalg.det(rot_matrix)<0:
            rot_matrix = - rot_matrix
        # print("SO3_matrix",rot_matrix)
        rot_matrix = Rotation.from_matrix(rot_matrix)
        # 提取欧拉角，可以指定旋转顺序，例如 'zyz' 表示先绕z轴旋转，然后是y轴，最后是z轴
        euler_angles = rot_matrix.as_euler('zyz', degrees=False)  # degrees=False 返回弧度制角度
        euler_angles = - euler_angles
        
        alpha = -euler_angles[0]
        beta = euler_angles[1]
        gamma = euler_angles[2]
        ### 按照zyz方向旋转exp(-i*simga_z*alpha/2)@exp(-i*simga_y*beta/2)@exp(-i*simga_z*gamma/2)
        matrix = np.array([[exp(-1j*gamma/2)*cos(beta/2)*exp(-1j*alpha/2),-exp(1j*gamma/2)*sin(beta/2)*exp(-1j*alpha/2)],
                [exp(-1j*gamma/2)*sin(beta/2)*exp(1j*alpha/2),exp(1j*gamma/2)*cos(beta/2)*exp(1j*alpha/2)]]
                        )

        ### 滤掉一些原本是0的数值误差
        threshold = 1e-9
        matrix[np.abs(matrix) < threshold] = 0
        matrix[np.abs(matrix.real)<threshold] -= matrix[np.abs(matrix.real)<threshold].real
        matrix[np.abs(matrix.imag)<threshold] -= matrix[np.abs(matrix.imag)<threshold].imag*1j
        
        ###加上这句话就和MagnticTB对上了，我暂时还不知道为什么，原因在该函数的注释里
        ### 这个地方有没有负号区别不大，因为在对Hamiltonian进行旋转的时候负号消掉了
        matrix = matrix.conj()
        # print("alpha,beta,gamma:",alpha,beta,gamma)
        # print("SU2_matrix",matrix)
        return matrix        
      
        
    def get_orbit_rot_dict(self,rot_op_dict,latt_vector,group_operation_list):
        """
        对于整个群操作进行轨道分析，
        这里是针对自旋体系，需要用到self.so3_2_su2函数，然后对需要spin的轨道进行直积
        
        Args:
            rot_op_dict (dict): 定义的点群中所有操作的名称和matrix 表示(Jones表示下),可以用群类中的get_rot_op函数得到
            latt (np.array): 晶格矢量
            group_operation_list: 磁群的操作列表,有磁群的get_operation函数得到
            
        return group_orbit_rot_matrix_dict (dict) :每个operation的旋转操作名称和是否Time Reverse 对应orbit基组的rotation matrix
        """
        if len(group_operation_list[0]) != 3:### 在该类中，group_operation_list没什么用，加上1是为了函数形式上的统一，2是加判断条件，以免造成人为编程上的差错
            raise AssertionError("输入的可能是Space Group或者其他群的操作，orbit_spd_with_spin类只负责Magnetic Group的操作，如有需要，请使用orbit_spd_without_spin类")    
        
        rot_op_dict = self.no_spin_orbit.rotation_basis_trans(rot_op_dict,latt_vector)

        ### 完成所有的群旋转操作
        rot_op_name = rot_op_dict.keys()
        group_orbit_rot_dict = {} ### 用于记录群中所有的旋转操作所对应的该阶所有orbit的旋转矩阵
        group_orbit_rot_matrix_dict = {} ###(函数返回值)用于记录我们输入的轨道在某个操作下所对应的rotation matrix
        for key0 in rot_op_name:
            orthometric_rotation = rot_op_dict[key0]
            ### 得到不同阶的basis的旋转矩阵
            orbit_rot_dict = self.no_spin_orbit.get_orbit_rotation_matrix(orthometric_rotation)
            # print("orbit_rot_dict",orbit_rot_dict)
            group_orbit_rot_dict[key0] = orbit_rot_dict
            ### 利用每个basis的旋转矩阵，使基完备
            self.complete_basis(orbit_rot_dict,key0)    
        self.spin_orbit_list = [value for key,value in self.spin_dict.items() if value == 1]              
        
        
        ### 完成对于完备化后的orbit_list轨道的分块旋转矩阵构造
        order_orbit_index_dict = {}###用于不同order分块矩阵的构造的index
        orbit_matrix_index = {}### 用于最后把所有的orbit放入最终得到的matrix的index
        self.orbit_num_dict_with_spin = {}
        matrix_index = 0
        orbit_name = self.orbit_num_dict.keys()
        for orbit in orbit_name:
            index = self.orbit_trans_dict[self.order_dict[orbit]].index(orbit)
            if self.order_dict[orbit] in order_orbit_index_dict.keys():
                order_orbit_index_dict[self.order_dict[orbit]].append(index)
            else:
                order_orbit_index_dict[self.order_dict[orbit]] = [index]
            if self.spin_dict[orbit] == self.orbit_num_dict[orbit]:### 不会有超出的情况，否则之前就有报错了
                orbit_matrix_index[orbit+" with spin"] = [matrix_index,matrix_index + 2*self.spin_dict[orbit]]
                self.orbit_num_dict_with_spin[orbit+" with spin"] = self.spin_dict[orbit]
                matrix_index = matrix_index + 2*self.spin_dict[orbit]
            elif self.spin_dict[orbit] != self.orbit_num_dict[orbit] and self.spin_dict[orbit] != 0:
                orbit_matrix_index[orbit+" with spin"] = [matrix_index,matrix_index + 2*self.spin_dict[orbit]]
                matrix_index = matrix_index + 2*self.spin_dict[orbit]
                self.orbit_num_dict_with_spin[orbit+" with spin"] = self.spin_dict[orbit]
                
                orbit_matrix_index[orbit+" without spin"] = [matrix_index,matrix_index + self.orbit_num_dict[orbit] - self.spin_dict[orbit]]
                matrix_index = matrix_index + self.orbit_num_dict[orbit] - self.spin_dict[orbit]
                self.orbit_num_dict_with_spin[orbit+" without spin"] = self.orbit_num_dict[orbit] - self.spin_dict[orbit]
            
            elif self.spin_dict[orbit] == 0:
                orbit_matrix_index[orbit+" without spin"] = [matrix_index,matrix_index + self.orbit_num_dict[orbit]]
                matrix_index = matrix_index + self.orbit_num_dict[orbit]
                self.orbit_num_dict_with_spin[orbit+" without spin"] = self.orbit_num_dict[orbit]

        self.orbit_matrix_index = orbit_matrix_index    
        self.matrix_dim = sum(self.orbit_num_dict_with_spin.values()) + sum(self.spin_dict.values())
        
        ### 对每一个operation构建矩阵，储存在group_orbit_rot_matrix_dict中（函数返回值）
        for op in group_operation_list:
            orbit_rot_matrix = np.zeros((self.matrix_dim,self.matrix_dim),dtype = np.complex64)
            su2 = self.so3_2_su2(rot_op_dict[op[0]]) ### 之后构造spin matrix的时候需要用到的su2矩阵
            sigma_y = np.array([[0,-1j],[1j,0]]) ### 之后如果有时间反演的话，需要用到的sigma_y矩阵
            orbit_rot_dict = group_orbit_rot_dict[op[0]]
            orbit_name_spin = orbit_matrix_index.keys()
            need_order = set(orbit_rot_dict.keys())
            # print(need_order)
            # print(self.orbit_num_dict)
            # print("orbit_matrix_index",orbit_matrix_index)
            # print(order_orbit_index_dict)
            # print("orbit_num_dict_with_spin",self.orbit_num_dict_with_spin)
            for order in need_order:
                index = order_orbit_index_dict[order]
                block_matrix = orbit_rot_dict[order]
                n = block_matrix.shape[0]
                block_matrix_spin = np.kron(block_matrix,su2)
                block_matrix_spin_TR = np.matmul(block_matrix_spin,np.kron(np.eye(n),1j*sigma_y))
                ###在可能存在多个相同轨道的情况下，判断是否具有spin和时间反演对称性，然后用su2群的结果进行直积
                ###对于spin而言有两种情况
                ###第1种：with spin，相互之间有关联的轨道相互作用；without spin，相互之间有关联的轨道相互作用
                ###第2种：相互独立的轨道之间的旋转，block_matrix中该矩阵元为0，不影响
                orbit_list_at_order = self.orbit_trans_dict[order]
                for orbit_name_spin1 in orbit_name_spin:
                    for orbit_name_spin2 in orbit_name_spin:
                        orbit_orgin_name1 = orbit_name_spin1.split()[0]
                        orbit_orgin_name2 = orbit_name_spin2.split()[0]
                        if orbit_orgin_name1 not in orbit_list_at_order or orbit_orgin_name2 not in orbit_list_at_order:
                            continue
                        
                        block_index1 = self.orbit_trans_dict[order].index(orbit_orgin_name1)
                        block_index2 = self.orbit_trans_dict[order].index(orbit_orgin_name2)
                        index1 = orbit_matrix_index[orbit_name_spin1]
                        index2 = orbit_matrix_index[orbit_name_spin2]
                        orbit_num1 = index1[1] - index1[0]
                        orbit_num2 = index2[1] - index2[0]

                        if "with spin" in orbit_name_spin1 and "with spin" in orbit_name_spin2:
                            with_spin = True
                        else:with_spin = False
                        
                        if orbit_num1 != orbit_num2:
                            continue
                        
                        # print("index1",index1)
                        # print("index2",index2)
                        # print("block_index1",block_index1)
                        # print("block_index2",block_index2)
                        
                        if with_spin and op[2] == False:
                            orbit_rot_matrix[index1[0]:index1[1],index2[0]:index2[1]] = np.kron(block_matrix_spin[2*block_index1:2*(block_index1+1),
                                                                                                          2*block_index2:2*(block_index2+1)],
                                                                                                np.eye(self.orbit_num_dict_with_spin[orbit_name_spin1]))### 如果是这种情况，spin个数一定相等，之前在complete_basis中强制约定过
                            # print(block_matrix_spin[2*block_index1:2*(block_index1+1),2*block_index2:2*(block_index2+1)])
                        elif with_spin and op[2] == True:
                            orbit_rot_matrix[index1[0]:index1[1],index2[0]:index2[1]] = np.kron(block_matrix_spin_TR[2*block_index1:2*(block_index1+1),
                                                                                                             2*block_index2:2*(block_index2+1)],
                                                                                                np.eye(self.orbit_num_dict_with_spin[orbit_name_spin1]))
                        elif not with_spin:
                            orbit_rot_matrix[index1[0]:index1[1],index2[0]:index2[1]] = np.kron(block_matrix[block_index1:block_index1+1,
                                                                                                    block_index2:block_index2+1],
                                                                                                np.eye(self.orbit_num_dict_with_spin[orbit_name_spin1]))
            group_orbit_rot_matrix_dict[str(op)] = orbit_rot_matrix                                                                                        
        return group_orbit_rot_matrix_dict
    
    def __len__(self):
        """轨道加自旋，一共有多少个轨道，也就是所对应的matrix维数有多少"""
        return self.matrix_dim                 
                     
                     
                     
                     
                     
import time

from symmetry_operation.mag_group.mg_operation import MagneticGroupOp
if __name__=="__main__":
    start_time = time.time()
    
    file_path_out = "G:\\DFTBAI\\DFAITB1\\develop_test\\test_orbit\\output24_9_29.txt"
    f = open(file_path_out,"w")
    
    test_group = MagneticGroupOp("191.234")
    rot_table = test_group.get_rot_op_rep()
    op_list = test_group.group_operation
    
    # spd_spin_orbit = orbit_spd_with_spin(["s","s","px","px","pz"],{"s":0,"px":2,"pz":0})
    spd_spin_orbit = orbit_spd_with_spin(["pz"],{"pz":1})
    latt = np.array([[1,0,0],[-1/2,3**0.5/2,0],[0,0,3]])
    group_rot_matrix = spd_spin_orbit.get_orbit_rot_dict(rot_table,latt,op_list)

    f.write(str(spd_spin_orbit.orbit_num_dict_with_spin)+"\n")
    f.write(str(spd_spin_orbit.matrix_dim)+"\n")
    f.write(str(group_rot_matrix)+"\n")
    f.write(str(spd_spin_orbit.orbit_list)+"\n")
    f.write(str(spd_spin_orbit.spin_dict)+"\n")
    f.write(str(spd_spin_orbit.orbit_trans_dict)+"\n")
    f.write(str(spd_spin_orbit.orbit_matrix_index)+"\n")
    end_time = time.time()
    print("run_time:",end_time-start_time)