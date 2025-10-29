import os
import sys
file_path = sys.argv[0]
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))

from Hamiltonian.hamiltonian import Hamiltonian

import numpy as np
from numpy import sqrt


from collections.abc import Mapping
import numbers
import itertools
from collections import Counter
import math
import pickle

class MyDict(dict):
    def __add__(self, other):
        if not isinstance(other, Mapping) and not isinstance(other, numbers.Number):  # 检查 other 是否是映射类型
            return NotImplemented
        if isinstance(other,numbers.Number):
            if abs(other) < 1e-7:
                return self 
        
        # 定义一个接近0的阈值
        threshold = 1e-7
        
        tmp = self.copy()
        for k, v in other.items():
            if k in tmp:
                tmp[k] += v
                # 如果值接近0，删除这个键
                if np.isclose(tmp[k], 0, atol=threshold):
                    del tmp[k]
            else:
                # 如果值接近0，不添加这个键
                if not np.isclose(v, 0, atol=threshold):
                    tmp[k] = v
        if tmp == {}:
            return 0
        else:
            return MyDict(tmp)

    def __mul__(self, scalar):
        if not isinstance(scalar, numbers.Number):  # 检查 scalar 是否是数字类型
            return NotImplemented
        if scalar == 0:
            return 0
        else:
            return MyDict({k: v * scalar for k, v in self.items()})

    __rmul__ = __mul__
    __radd__ = __add__

    def __sub__(self, other):
        return self.__add__((-1)*other)
    

class kpHamiltonian(Hamiltonian):
    def __init__(self, dim=1, order=None ,rotation_list= []):
        if order==None:
            raise("忘记输入order")
        elif not isinstance(order,list):
            raise("请输入一个list")
        
        self.name = "kp hamiltonian"
        self.dim = dim
        self.order = order
        self.threshold = 1e-7
        self.basis = self.su_n_generators(dim)
        self.n_basis = len(self.basis)
        self.rotation_list = rotation_list
        self.kp_hamiltonian_dict = self.init_kp_hamiltonian()
        print("完成初始化构建")
        print("kp_hamiltonian",self.kp_hamiltonian_dict)
        self.__generator()
        print("完成对称性约束")
        self.__modify_symbols()
        print("kp_hamiltonian",self.kp_hamiltonian_dict)


    def su_n_generators(self,n):
        """
        生成 SU(n) 的生成元
        :param n: SU(n) 的维度
        :return: 一个包含 n²−1 个生成元的列表
        """
        # 生成所有可能的 (i, j) 对
        idx = np.triu_indices(n, 1)
        num_off_diag = len(idx[0])
        
        # 初始化生成元列表
        generators = []
        # generators = []
        
        # 生成非对角线上的生成元
        for k in range(num_off_diag):
            i, j = idx[0][k], idx[1][k]
            
            # 生成 X 类生成元
            X = np.zeros((n, n), dtype=complex)
            X[i, j] = 0.5
            X[j, i] = 0.5
            generators.append(1j * X)
            
            # 生成 Y 类生成元
            Y = np.zeros((n, n), dtype=complex)
            Y[i, j] = -0.5j
            Y[j, i] = 0.5j
            generators.append(1j * Y)
        
        # 生成对角线上的生成元
        for m in range(1, n):
            D = np.zeros((n, n), dtype=complex)
            D[:m, :m] = np.eye(m) / np.sqrt(2 * m * (m + 1))
            D[m, m] = -m / np.sqrt(2 * m * (m + 1))
            generators.append(1j * D)

        generators = [(-1j)*np.sqrt(2)*generator0 for generator0 in generators]
        # print([generator0.conj().T - generator0 for generator0 in generators])
        generators.insert(0,np.eye(n,dtype=complex))
        
        return np.array(generators)


    def init_kp_hamiltonian(self):
        kp_ham_dict = {}
        for order0 in self.order:
            kp_ham_dict[order0] = self.__init_kp_order_ham(order0)
        return kp_ham_dict


    def __init_kp_order_ham(self,order0):
        shape_k = [3]*order0
        shape = shape_k.copy()
        shape.append(self.n_basis)
        shape = tuple(shape)
        symbol_index = 0
        have_in = []
        ham = {}
        for index in np.ndindex(shape):
            if index in have_in:
                continue
            perms = list(set(itertools.permutations(index[:-1])))
            have_in += [perm+(index[-1],) for perm in perms]
            ham[index] = MyDict({symbol_index:1})
            symbol_index += 1
        return ham


    def __exact_var_array(self,ham_array):
        var_list = []
        it = np.nditer(ham_array, flags=['multi_index','refs_ok'], op_flags=['readwrite'])
        for ele in it:
            if ele == 0:
                continue
            ele = ham_array[it.multi_index]
            for key0 in ele.keys():
                if key0 not in var_list:
                    var_list.append(key0)
        num_var = len(var_list)
        return var_list, num_var
    
    def __exact_var_dict(self,ham_dict):
        sym_list = []
        for order0 in self.order:
            ham_dict0 = ham_dict[order0]
            for k,formula in ham_dict0.items():
                for sym in formula.keys():
                    if sym not in sym_list:
                        sym_list.append(sym)
        return sym_list,len(sym_list)

    def __exact_eqaution(self,var_list,num_var,ham_array):
        equations = []
        it = np.nditer(ham_array, flags=['multi_index','refs_ok'], op_flags=['readwrite'])
        for ele in it:
            if ele == 0:
                continue
            ele = ham_array[it.multi_index]
            equation = np.zeros(num_var,dtype = np.complex128)
            for key0,value0 in ele.items():
                index = var_list.index(key0)
                equation[index] = value0
            equations.append(equation)
        return np.array(equations)


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

            ### 消除误差
            A[np.abs(A)< self.threshold] = 0
            
            if equation_anchor == num_equation:
                break 
            # print(equation_anchor,A)
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
        A[np.abs(A)< self.threshold] = 0
                # if abs(A[i,j].real) < self.threshold:
                #     A[i][j] = A[i][j].imag*1j
                # if abs(A[i,j].imag) < self.threshold:
                #     A[i][j] = A[i][j].real
        return A
    

    def replace(self,equations,var_list,ham_dict):
        n_equ = equations.shape[0]
        for i in range(n_equ):
            if np.sum(np.abs(equations[i])) > 1e-7:
                index = np.where(equations[i])[0]
                index_symbol = [var_list[index0] for index0 in index]
                num_index = len(index)
                for k,v in ham_dict.items():
                    if v == 0:
                        continue
                    if index_symbol[0] in v.keys():
                        para = v[index_symbol[0]]
                        del v[index_symbol[0]]
                        formula_replace = {index_symbol[j]:-para*equations[i][index[j]] for j in range(1,num_index)}
                        ham_dict[k] = v + formula_replace
        ham_dict = {k:v for k,v in ham_dict.items() if v != 0}
        return ham_dict


    def basis_trans(self,trans_matrix,if_anitunitary):
        """
        得到basis的旋转得到旋转矩阵
        trans_matrix: the unitary matrix to rotate the basis
        """
        if trans_matrix.shape != (self.dim,self.dim):
            raise ValueError("trans_matrix的形状不对,"+str(trans_matrix.shape)+",应该是"+str((self.dim,self.dim)))
        if np.sum(np.abs(trans_matrix.conj().T@trans_matrix - np.eye(self.dim))) > 1e-4:
            raise ValueError("not unitary")
        
        n_basis = self.basis.shape[0]
        basis = np.expand_dims(self.basis, axis=0)
        if not if_anitunitary:
            basis = trans_matrix@basis@trans_matrix.conj().T
        else:
            basis = trans_matrix@basis.conj()@trans_matrix.conj().T
        basis = basis.repeat(n_basis,axis=0)

        basis = np.transpose(basis,axes=(0, 1, 3, 2)).conj()
        basis_matrix = np.einsum("ijmn,inm -> ij",basis,self.basis)
        basis_matrix = np.transpose(basis_matrix,axes=(1,0))
        basis_matrix[0] = basis_matrix[0]/basis_matrix[0,0]
        return basis_matrix


    def symmetrize_efficient(self,A):
        # 获取数组的维度
        dims = len(A.shape) - 1
        # 创建一个对称化的数组
        symmetric_A = np.zeros_like(A)
        perm_seed = tuple([i for i in range(dims)])
        perms = itertools.permutations(perm_seed)
        perms = [perm + (dims,) for perm in perms]
        n_perms = len(perms)
        # 遍历所有可能的转置组合
        for perm in perms:
            # 计算所有转置的和
            symmetric_A += np.transpose(A, axes=perm)
        # 除以转置组合的数量
        indices = np.ndindex(A.shape)
        for index in indices:
            counter = Counter(index[:-1])
            count_dict = {item: count for item, count in counter.items() if count > 1}
            symmetric_A[index] = symmetric_A[index]/math.prod([math.factorial(v) for k,v in count_dict.items()])
        return symmetric_A


    def rotation(self,basis_rotation,k_rotation,ham,if_anitunitary:bool):
        ham_index = list(ham.keys())
        ham_index_array = np.array(ham_index).T
        trans_index = tuple([ham_index_array[i] for i in range(ham_index_array.shape[0])])
        rotation_equation_array = np.zeros(len(ham_index),dtype=object)
        basis_transform = self.basis_trans(basis_rotation,if_anitunitary)
        for index in ham_index:
            n = len(index)
            if not if_anitunitary:
                vector_list = [k_rotation[index[i]] for i in range(n-1)]
                vector_list.append(basis_transform[index[-1]])
            else:
                vector_list = [-k_rotation[index[i]] for i in range(n-1)]
                vector_list.append(basis_transform[index[-1]])
            indices = ','.join([chr(97 + i) for i in range(n)])  # 'i,j,k'
            output_indices = ''.join([chr(97 + i) for i in range(n)])  # 'ijk'
            einsum_expr = f'{indices}->{output_indices}'
            tensor_product = np.einsum(einsum_expr, *vector_list)
            tensor_product = self.symmetrize_efficient(tensor_product)
            # print(ham[index],index,"\n",tensor_product)
            rotation_equation_array += ham[index]*tensor_product[trans_index]
        return rotation_equation_array, ham_index


    def __generator(self):
        for order0 in self.order:
            ham = self.kp_hamiltonian_dict[order0]
            
            for rotation in self.rotation_list:
                ham_rotation,ham_index = self.rotation(rotation[0],rotation[1],ham,rotation[2])
                ham_array = np.array([ham[ham_index0] for ham_index0 in ham_index])
                delta = ham_array - ham_rotation
                # print("ham_final",ham_rotation)
                # print("ham_array",ham_array)
                #print("delta",delta)
                var_list, num_var = self.__exact_var_array(delta)
                equations = self.__exact_eqaution(var_list,num_var,delta)
                equations = self.__gaussian_elimination(equations)
                ham = self.replace(equations,var_list,ham)
                #print("ham",ham)
            self.kp_hamiltonian_dict[order0] = ham
            

    def __modify_symbols(self):
        orders = self.kp_hamiltonian_dict.keys()
        index_anchor = 0
        for order in orders:
            ham = self.kp_hamiltonian_dict[order]
            self.var_list,self.var_num = self.__exact_var_dict(self.kp_hamiltonian_dict)
            symbol_replace_dict = {self.var_list[i]:index_anchor+i for i in range(self.var_num)}
            index_anchor += self.var_num
            for k,v in ham.items():
                new_formula = {symbol_replace_dict[key]:value for key,value in v.items() if key in symbol_replace_dict.keys()}
                ham[k] = new_formula
            self.kp_hamiltonian_dict[order] = ham

    def save_model(self,save_path):
        file_name = os.path.join(save_path,self.sysinit["sys_name"]+".pkl")
        content = {
            "rotation":self.rotation_list,
            "model":self.kp_hamiltonian_dict,
            "num_symbols":self.var_num,
            "name":self.name,
            "basis":self.basis
        }
        with open(file_name,"wb") as f:
            pickle.dump(content,f)


import time
if __name__ == "__main__":
    rotation_list = [(np.array([[1,  0],[0, -1]]),np.array([[0,-1,0],[1,0,0],[0,0,1]]),False),(np.array([[1,  0],[0, 1]]),-np.eye(3),True)]
    start_time = time.perf_counter()
    kp = kpHamiltonian(dim = 2 ,order = [1],rotation_list=rotation_list)
    end_time = time.perf_counter()
    print(end_time-start_time)
    ham = kp.kp_hamiltonian_dict[1]
    print(len(ham))

    ##L4L4
    # C31=np.array([[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]])
    # C2b=np.array([[1j,0,0,0],[0,-1j,0,0],[0,0,1j,0],[0,0,0,-1j]])
    # Inv=np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])
    # T=np.array([[0,0,0,-1],[0,0,-1,0],[0,1,0,0],[1,0,0,0]])
    # rotation_list = [(C31,np.linalg.inv(np.array([[0,0,1],[1,0,0],[0,1,0]])),False),
    #                  (C2b,np.linalg.inv(np.array([[0,-1,0],[-1,0,0],[0,0,-1]])),False),
    #                  (Inv,np.linalg.inv(np.array([[-1,0,0],[0,-1,0],[0,0,-1]])),False),
    #                  (T,np.eye(3),True)]
    # start_time = time.perf_counter()
    # kp = kpHamiltonian(dim = 4 ,order = [1],rotation_list=rotation_list)
    # end_time = time.perf_counter()
    # print(end_time-start_time)
    # print(kp.basis_trans(T,True))

