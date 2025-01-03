import os
import sys
file_path = sys.argv[0]
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))

import numpy as np

import pickle
import json
import numbers

from Hamiltonian.hamiltonian import Hamiltonian
from Hamiltonian.matrix_element import Matrix_Ele
from physics_system.periodicity.Periodicity_System import PeriodicityPhysicsSystem
from orbit.orbit_spd.Orbit_spd_manager import orbit_spd_manager


class TBHamiltonian(Hamiltonian):
    def __init__(self, dim=1,orbit_init:list = None,sysinit:dict = None):
        super(TBHamiltonian,self).__init__(dim)
        """
            orbit_list有两种用法(这里默认是用spd类型轨道)
                1、是每个Wcykoff position上分别应该用什么轨道
                2、每个Wcykoff position上统一用什么轨道
            sysinit是需要输入的初始化研究体系的变量
        """
        self.name = "tight binding hamiltonian"
        self.use_orbit_type = "not defined"
        self.periodicitysystem = PeriodicityPhysicsSystem(**sysinit)
        self.wyck = self.periodicitysystem.wyckoffpos
        self.orbit_init = self.__check_model_init(orbit_init)
        
        self.orbit_init = orbit_init
        self.sysinit = sysinit
        
        self.orbit_list,self.orbit_rotation_list = self.get_orbit_rotation_list()
        print("atom_rep:",self.periodicitysystem.atom_rep)
        print("neighbour_table:",self.periodicitysystem.neighbour_table)
        print("orbit_list",self.orbit_list)

        self.threshold = 1e-9
        self.unsym_dict = {}
        self.sys_atom_index = []
        self.unsym_dict = self.unsym_construct()
        print("完成无对称性约束的Hamiltonian生成")
        self.rotation_matrix_dict = self.generate_orbit_rotation_matrix()
        print("完成Hamiltonian旋转矩阵的构建")
        self.sym_hamiltonian_dict = {}
        self.sym_hamiltonian_dict = self.__generator()
        print("完成Hamiltonian构建")
        self.num_symbols = self.__modify_symbols()
    
    def __check_model_init(self,orbit_init):
        if orbit_init == None:
            raise AssertionError("忘记输入orbit定义信息，请重新输入")
        n_orbit = len(orbit_init)
        if n_orbit == 1:
            orbit_init = orbit_init*len(self.wyck)
            self.use_orbit_type = "Uniform"
        elif n_orbit != len(self.wyck):
            raise AssertionError("经过分析，一共有{}个wyck占位,但是给出了{}个orbit定义,二者应该数目一致".format(len(self.wyck),n_orbit))
        else: self.use_orbit_type = "Respective"
        
        return orbit_init
        
    def get_orbit_rotation_list(self):
        n_orbit_set = len(self.orbit_init)
        orbit_list = []
        for i in range(n_orbit_set):
            orbit_list.append(orbit_spd_manager(**self.orbit_init[i]))
            
        rot_dict = self.periodicitysystem.group.get_rot_op_rep()
        op_list = self.periodicitysystem.group.group_operation
        latt = np.array(self.periodicitysystem.lattice.lattice_vector.subs(self.periodicitysystem.lattice_parameter),dtype = float)
        latt = np.linalg.inv(self.periodicitysystem.lattice.Conven2Prim_matrix)@latt###Prim2Convent的变换成Orbit需要的Concentional Cell的latt vector
        orbit_rotation_list = [wyck_orbit.tool.get_orbit_rot_dict(rot_dict,latt,op_list) for wyck_orbit in orbit_list]

        return orbit_list,orbit_rotation_list
    
    def unsym_construct(self):
        """通过orbit和Wcykoff position的atom_bond来构建没有对称性约束的Hamiltonian矩阵
        """
        unsym_matrix = {}
        
        n_neighbour = self.periodicitysystem.n_neighbour
        neighbour_table = self.periodicitysystem.neighbour_table
        atom_pos = self.periodicitysystem.atompos
        n_atom = len(atom_pos)
        n_wyck = len(self.wyck)
        
        n_sys_orbit = 0
        sys_atom_list = [0]
        ### 同一个wyckoff position 用同一套 orbit basis,所以对不同wyckoff position需要进行分类
        for i in range(n_wyck):
            n_atom_on_wyck = len(self.wyck[i]) ### 在某个wyckoff position上的原子个数
            for j in range(n_atom_on_wyck):
                n_sys_orbit += len(self.orbit_list[i].tool)
                sys_atom_list.append(sys_atom_list[-1]+len(self.orbit_list[i].tool))
        self.sys_atom_index = sys_atom_list

        
        for neighbour0 in range(n_neighbour):
            symbol_index = 0
            matrix = np.zeros((n_sys_orbit,n_sys_orbit),dtype=object)
            for i in range(n_atom):
                if str(neighbour0) in neighbour_table[i].keys():
                    neighbour_list = neighbour_table[i][str(neighbour0)]
                else:
                    continue
                atom_related = set([atom_index0 for atom_index0,_ in neighbour_list])
                for j in atom_related:
                    matrix[sys_atom_list[i]:sys_atom_list[i+1],
                           sys_atom_list[j]:sys_atom_list[j+1]],symbol_index = self.__matrix_block_orbit(symbol_index,j,
                                                                                            sys_atom_list[i+1]-sys_atom_list[i],
                                                                                            sys_atom_list[j+1]-sys_atom_list[j],
                                                                                            neighbour_list,atom_pos[i])
            for i in range(n_sys_orbit):
                for j in range(n_sys_orbit):
                    if matrix[i,j] == 0:
                        matrix[i,j] = Matrix_Ele([],[])
            unsym_matrix[neighbour0] = matrix
        return unsym_matrix
    
    def __matrix_block_orbit(self,symbol_index,atom2_index,n_orbit1,n_orbit2,neighbour_list,atom_pos):
        matrix_block = np.zeros((n_orbit1,n_orbit2),dtype=object)
        neighbour_list_index = [(pos-atom_pos).transpose() for atom_index0,pos in neighbour_list if atom_index0 == atom2_index]
        n_neighbour = len(neighbour_list_index)
        # print(neighbour_list_index)
        for i in range(n_orbit1):
            for j in range(n_orbit2):
                exp_list = []
                formula_list = []
                for n_neigh in range(n_neighbour):
                    exp_list.append(neighbour_list_index[n_neigh])
                    formula_list.append({symbol_index:1,symbol_index+1:1j})
                    symbol_index += 2
                matrix_block[i,j] = Matrix_Ele(exp_list,formula_list)
        return matrix_block,symbol_index
        
    def generate_orbit_rotation_matrix(self):
        """
        得到所有操作下的Hamiltonian旋转矩阵\n
        当某个操作转到某wyckoff position上的atom时，对应原子的轨道应该旋转
        比如，0号原子转到了1号原子的位置上面，那么旋转矩阵0-1的block上面应该是轨道的对应的旋转操作矩阵
        
        """
        rot_op_dict = self.periodicitysystem.group.group_operation_rep
        rot_op_dict = self.periodicitysystem.group.basis_trans(np.linalg.inv(self.periodicitysystem.lattice.Conven2Prim_matrix))
        op_names  = rot_op_dict.keys()
        atom_rep = self.periodicitysystem.atom_rep
        n_wyck = len(self.wyck)
        rotation_matrix_dict = {}
        matrix_dim = self.sys_atom_index[-1]
        latt = np.array(self.periodicitysystem.lattice.lattice_vector.subs(self.periodicitysystem.lattice_parameter),dtype=float)
        recp_latt = np.array(self.periodicitysystem.lattice.repi_lattice_vector.subs(self.periodicitysystem.lattice_parameter),dtype=float)
        # print(self.orbit_rotation_list)
        for op_name in op_names:
            rotation_matrix = np.zeros((matrix_dim,matrix_dim),dtype=complex)
            count_atom = 0 
            for num_wyck in range(n_wyck):
                n_atom_in_wyck = len(self.wyck[num_wyck])
                for num_atom in range(n_atom_in_wyck):
                    output = np.matmul(rot_op_dict[op_name],self.wyck[num_wyck][num_atom]) ###作用于对称性操作后，得到旋转后的原子坐标，需要对1取模才是在原胞中的原子坐标
                    output[0:3] = output[0:3]%1
                    # print(output)
                    index = [i for i in range(self.periodicitysystem.n_atom) if np.sum(np.abs(np.array(output)-np.array(atom_rep[i])))<1e-9][0]
                    # # print(index)
                    rotation_matrix[self.sys_atom_index[count_atom]:self.sys_atom_index[count_atom+1],
                                    self.sys_atom_index[index]:self.sys_atom_index[index+1]] = self.orbit_rotation_list[num_wyck][op_name]
                    count_atom += 1
                    
            rotation_matrix_dict[op_name] = rotation_matrix
            ### 倒空间矢量的旋转,并滤掉数值误差
            # print("op_name:",op_name,"rot_op_dict:",rot_op_dict[op_name][0:3,0:3])
            ### 先变换到Conventional Cell然后再变换到倒空间
            k_matrix = np.linalg.inv(np.transpose(recp_latt))@np.transpose(latt)@rot_op_dict[op_name][0:3,0:3]@np.linalg.inv(np.transpose(latt))@np.transpose(recp_latt)
            k_matrix[np.abs(k_matrix) < self.threshold] = 0
            # print("k_matrix:",k_matrix)
            rotation_matrix_dict["k "+op_name] = k_matrix
        return rotation_matrix_dict
    
    def statics_symbol(self,matrix):
        """
        用于统计matrix中现有的变量和变量个数
        """
        n_dim = matrix.shape[0]
        var_list = []
        for ele in np.nditer(matrix,flags=['refs_ok']):
            ele = ele.item()
            if isinstance(ele,Matrix_Ele):
                ele_var_list = ele.var_list
                new_var = [var for var in ele_var_list if var not in var_list]
                var_list = var_list + new_var
        num_var = len(var_list)
        return var_list,num_var
            
    def __parameter_reduce_solver(self,matrix):
        """这个函数用来求解厄密和对称性约束下，matrix的独立变量，最后返回约化后的矩阵，和每个index对应的list

        Args:
            matrix (np.array): 需要处理的矩阵
        """
        n_dim = matrix.shape[0]      
        for i in range(n_dim):###在确定公式前，先检查一遍，以免有空值
            for j in range(n_dim):
                if isinstance(matrix[i,j],Matrix_Ele):
                    if matrix[i,j].empty:
                        matrix[i,j] = 0   
                           
        var_list,var_num = self.statics_symbol(matrix)
        num_var = len(var_list)
        equations = []
        for ele in np.nditer(matrix,flags=['refs_ok']):
            ele = ele.item()
            if not isinstance(ele,Matrix_Ele):
                continue
            for formula in ele.formula_list:
                symbol_index = formula.keys()
                equation_array = np.zeros((num_var),dtype=np.complex64)
                for index_s in symbol_index:###一个是符号索引，一个是该符号在矩阵中的索引
                    index_n = var_list.index(index_s)
                    equation_array[index_n] = formula[index_s]
                equations.append(equation_array)
        equations = np.array(equations)
        equations = self.__gaussian_elimination(equations)
        
        return equations,var_list
        
    def __gaussian_elimination(self,A):
        if A.shape[0] == 0:
            return A
        num_equation = A.shape[0]
        num_symbol = A.shape[1]  
        num = min(num_equation,num_symbol)
        print(A.shape)
        ### 向下消元，化为行阶梯矩阵
        equation_anchor = 0
        for i in range(num_symbol):
            # 寻找绝对值最大的行，用于行交换
            max_row = max(range(equation_anchor, num_equation), key=lambda r: abs(A[r][i]))
            pivot = A[max_row,i]
            if abs(pivot) == 0:
                continue
            else:
                A[[equation_anchor, max_row]] = A[[max_row, equation_anchor]]# 交换行
            # 归一化当前行的主元
            A[equation_anchor] = A[equation_anchor] / pivot
            # 消去当前列的其他元素
            for j in range(equation_anchor+1, num_equation):
                    A[j] = A[j] - A[j][i] * A[equation_anchor]
            equation_anchor += 1   
            if equation_anchor == num_equation:
                break 
            
        ### 向上消元，化为最简形式
        for i in range(num_equation-1,0,-1):
            index = np.where(A[i])[0]
            if not np.any(index):### 判断是否是空数组
                continue
            else:
                index = index[0]
            for j in range(i):
                A[j] = A[j] - A[j][index]*A[i]
        
        ### 消除误差
        for i in range(num_equation):
            for j in range(num_symbol):
                if abs(A[i,j]) < self.threshold:
                    A[i,j] = 0
                # if abs(A[i,j].real) < self.threshold:
                #     A[i][j] = A[i][j].imag*1j
                # if abs(A[i,j].imag) < self.threshold:
                #     A[i][j] = A[i][j].real
        return A
    
    def __equations_replace(self,equations,var_list,matrix):
        matrix_dim = matrix.shape[0]
        num_equations = equations.shape[0]
        if num_equations == 0:
            return matrix
        for i in range(matrix_dim):
            for j in range(matrix_dim):
                if isinstance(matrix[i,j],Matrix_Ele):
                    for k in range(num_equations):
                        matrix[i,j].replace(equations[k],var_list)
        return matrix
        
    def __k_rotation(self,matrix,k_rot):
        matrix_dim = matrix.shape[0]
        new_matrix = np.zeros((matrix_dim,matrix_dim),dtype=object)
        for i in range(matrix_dim):
            for j in range(matrix_dim):
                if not matrix[i,j].empty:
                    new_matrix[i,j] = matrix[i,j].k_rotation(k_rot)
                else:new_matrix[i,j] = Matrix_Ele([],[])
        return new_matrix
    
    def __generator(self):
        """
        现在我们有了对称性操作的旋转矩阵self.rotation_matrix_dict和没有对称性的hamiltonian矩阵self.unsym_dict\n
        我们现在需要用公式
            1 H^{T*} = H Hamiltonian是厄密矩阵
            2 G*H(R*k)*G^{-1} = H(k)来求解符合对称性的Hamiltonian矩阵\n
        """
        neighbours = self.unsym_dict.keys()
        sym_hamiltonian = {}
        rot_op_dict = self.periodicitysystem.group.group_operation_rep
        for neighbour in neighbours:
            matrix = self.unsym_dict[neighbour]
            
            # matrix_test = np.zeros_like(matrix,dtype=object)
            # for i in range(matrix.shape[0]):
            #     for j in range(matrix.shape[1]):
            #         if isinstance(matrix[i,j],Matrix_Ele):
            #                 matrix_test[i,j] = matrix[i,j].var_list
            # print(matrix_test)
            
            delta_conj_hamiltonian = matrix.T.conj() - matrix
            equations,var_list = self.__parameter_reduce_solver(delta_conj_hamiltonian)
            print(var_list)
            matrix = self.__equations_replace(equations,var_list,matrix)
            print(self.statics_symbol(matrix))
            
            op_names = rot_op_dict.keys()
            for op_name in op_names:
                print("neighbour:",neighbour,"op_name:",op_name)
                k_rot = np.linalg.inv(self.rotation_matrix_dict["k "+op_name])
                matrix_rot_k = self.__k_rotation(matrix,k_rot)
                if "True" in op_name:
                    matrix_rot_k = self.__k_rotation(matrix_rot_k,np.array([[-1,0,0],[0,-1,0],[0,0,-1]]))
                    matrix_rot_k = matrix_rot_k.conjugate()

                matrix_rot = np.linalg.inv(self.rotation_matrix_dict[op_name])@matrix@self.rotation_matrix_dict[op_name]
                delta_matrix = matrix_rot - matrix_rot_k
                equations,var_list = self.__parameter_reduce_solver(delta_matrix)
                matrix = self.__equations_replace(equations,var_list,matrix)
                print(self.statics_symbol(matrix))
            sym_hamiltonian[neighbour] = matrix
                
        return sym_hamiltonian
    
    def __modify_symbols(self):
        neighbours = self.sym_hamiltonian_dict.keys()
        index_anchor = 0
        for neighbour in neighbours:
            var_list,var_num = self.statics_symbol(self.sym_hamiltonian_dict[neighbour])
            matrix = self.sym_hamiltonian_dict[neighbour]
            matrix_dim = matrix.shape[0]
            symbol_replace_dict = {var_list[i]:index_anchor+i for i in range(var_num)}
            index_anchor += var_num
            print(symbol_replace_dict)
            for i in range(matrix_dim):
                for j in range(matrix_dim):
                    if not matrix[i,j].empty:
                        new_formula_list = []
                        for k in range(matrix[i,j].exp_num):
                            new_formula = {symbol_replace_dict[key]:value for key,value in matrix[i,j].formula_list[k].items() if key in symbol_replace_dict.keys()}
                            new_formula_list.append(new_formula)
                        matrix[i,j] = Matrix_Ele(matrix[i,j].exp_list,new_formula_list)
            self.sym_hamiltonian_dict[neighbour] = matrix                  
        return index_anchor
        
    def save_model(self,save_path):
        file_name = os.path.join(save_path,self.sysinit["sys_name"]+".pkl")
        content = {
            "info":{"sysinit":self.sysinit,"orbit_init":self.orbit_init},
            "model":self.sym_hamiltonian_dict,
            "num_symbols":self.num_symbols,
            "name":self.name
        }
        with open(file_name,"wb") as f:
            pickle.dump(content,f)
        
        
    
import time
if __name__ == "__main__":
    # start_time = time.time()
    # exp_list1 = [np.array([0,0,0]),np.array([0.5,0.5,0.5]),np.array([-0.25,0.5,0.5])]
    # formula_list1 = [{1:1+2j},{2:2+0.5j},{3:0.5+1j}]
    # exp_list2 = [np.array([0,0,-0]),np.array([0.5,-0.5,0.5]),np.array([-0.25,0.5,0.5])]
    # formula_list2 = [{10:1+2j},{7:2+0.5j},{3:-0.5+1e-8-1j}]
    # matrix_ele1 = Matrix_Ele(exp_list1,formula_list1)
    # matrix_ele2 = Matrix_Ele(exp_list2,formula_list2)
    # matrix_ele = matrix_ele1 + matrix_ele2 + 2j
    
    # matrix_ele_sub = matrix_ele1 - matrix_ele2
    # # matrix_ele_mul = matrix_ele*(2.3423+1.3234j)
    # matrix_ele_mul = 2*matrix_ele
    # end_time = time.time()
    # print(end_time-start_time)
    
    # print(matrix_ele1.exp_list)
    # print(matrix_ele1.formula_list)
    # print(matrix_ele_sub.var_list)
    
    # print(matrix_ele.exp_list)
    # print(matrix_ele.formula_list)
    # print(matrix_ele_sub.exp_list)
    # print(matrix_ele_sub.formula_list)
    # print(matrix_ele_mul.exp_list)
    # print(matrix_ele_mul.formula_list)
    
    # a = np.array([0,1,0,0,0,0,0,0.5,0,0,0])
    # matrix_ele_mul.replace(a)
    # print(matrix_ele_mul.exp_list)
    # print(matrix_ele_mul.formula_list)
    
    # start_time = time.time()
    
    # a = np.array([[0.5,-0.8660254],
    #               [0.8660254,0.5]],dtype=np.complex128)
    # a = np.array([[1,0],
    #               [0,1]],dtype = np.complex128)
    # symbol_matrix = np.array([[Matrix_Ele(exp_list1,formula_list1),Matrix_Ele([],[])],
    #                           [Matrix_Ele(exp_list1,formula_list1),Matrix_Ele(exp_list1,formula_list1)]])   
    # print(symbol_matrix.dtype)
    # symbol_matrix = a@symbol_matrix
    
    # print(symbol_matrix[0,1].exp_list)
    # print(symbol_matrix[0,1].formula_list)
    
    
    sysinit = {
                "sys_name":"Si_sps'",
                "group_type":"Space Group",
                "group_name":"227",
                "lattice_type":"CubiFace",
                "lattice_parameter":{"a":1},
                "atompos":[[1/8,1/8,1/8]],
                # "magdirect":[[0,0,0]],
                "neighbour_list":[2]
                }
    
    # sysinit = {
    #             "sys_name":"Si",
    #             "group_type":"Magnetic Group",
    #             "group_name":"227.128",
    #             "lattice_type":"CubiFace",
    #             "lattice_parameter":{"a":1},
    #             "atompos":[[1/8,1/8,1/8]],
    #             "magdirect":[[0,0,0]],
    #             "n_neighbour":2
    #             }
        
    # orbitinit = [{"orbit_list":["s","px","py","pz"],"spin_dict":{"s":0,"px":0,"py":0,"pz":0}}]
    
    orbitinit = [{"orbit_list":["s","s","px","py","pz"]}]
    start_time = time.time()
    model = TBHamiltonian(orbit_init=orbitinit,sysinit = sysinit)
    end_time = time.time()
    run_time = end_time - start_time
    print(run_time)
    print(model.statics_symbol(model.sym_hamiltonian_dict[0]))
    print(model.statics_symbol(model.sym_hamiltonian_dict[1]))
    model.save_model("/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC")
    # print(model.unsym_dict)
    # print(model.rotation_matrix_dict)
    
    
    