import os
import sys
file_path = sys.argv[0]
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))

import numpy as np

import pickle
import json
import numbers
import math

from Hamiltonian.Hamiltonian4kp.k_poly_term import kp_k_term
from Hamiltonian.hamiltonian import Hamiltonian

def random_unitary_matrix(n):
    """
    生成一个 n×n 的随机酉矩阵。
    """
    # 生成一个随机的复数矩阵
    z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    # 进行 QR 分解
    q, r = np.linalg.qr(z)
    # 修正 R 的对角线，使其为正数
    d = np.diag(np.diag(r) / np.abs(np.diag(r)))
    # 修正后的 Q 就是一个酉矩阵
    u = q @ d
    return u


class kpHamiltonian(Hamiltonian):
    def __init__(self, dim=1, order=1 ,rotation_list= []):
        super().__init__(dim)
        self.name = "kp hamiltonian"
        self.order = order
        self.threshold = 1e-7
        self.basis = self.su_n_generators(dim)
        self.n_basis = len(self.basis)
        self.rotation_list = rotation_list

        self.kp_hamiltonian_dict = self.init_kp_hamiltonian()
        print("完成无对称性约束的Hamiltonian生成")
        self.sym_hamiltonian_dict = {}
        self.sym_hamiltonian_dict = self.__generator()


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
        generators = [np.eye(n, dtype=complex)]
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

        generators = [np.sqrt(2)*generator0 for generator0 in generators]

        return np.array(generators)


    def is_su_n_basis(self, matrices, tol=1e-8):
        """
        检查一组矩阵是否构成 SU(n) 的基
        :param matrices: 一个包含 n²−1 个 n×n 复矩阵的列表
        :param tol: 数值容差
        :return: 是否构成 SU(n) 的基
        """
        n = matrices[0].shape[0]
        num_matrices = len(matrices)
        
        # 检查矩阵数量是否正确
        if num_matrices != n**2 - 1:
            print("矩阵数量不正确，应为 n²−1。")
            return False
        
        # 检查每个矩阵是否无迹
        for T in matrices:
            if not np.isclose(np.trace(T), 0, atol=tol):
                print("矩阵不是无迹的。")
                return False
        
        # 检查每个矩阵是否反厄米
        for T in matrices:
            if not np.allclose(T.conj().T, -T, atol=tol):
                print("矩阵不是反厄米的。")
                return False
        
        # 检查正交归一性
        for i in range(num_matrices):
            for j in range(num_matrices):
                inner_product = np.trace(matrices[i].conj().T @ matrices[j])
                if i == j:
                    if not np.isclose(inner_product, 1.0, atol=tol):
                        print(str(i)+","+str(j)+"矩阵不是归一化的。"+str(inner_product))
                        return False
                else:
                    if not np.isclose(inner_product, 0, atol=tol):
                        print("矩阵不是正交的。")
                        return False
        
        print("所有条件均满足，这是一组 SU(n) 的基。")
        return True


    def basis_trans(self,trans_matrix):
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
  
        basis = trans_matrix@basis@trans_matrix.conj().T
        basis = basis.repeat(n_basis,axis=0)

        basis = np.transpose(basis,axes=(0, 1, 3, 2)).conj()
        basis_matrix = np.einsum("ijmn,inm -> ij",basis,self.basis)
        basis_matrix = np.transpose(basis_matrix,axes=(1,0))
        basis_matrix[0] = basis_matrix[0]/basis_matrix[0,0]
        return basis_matrix


    def init_kp_hamiltonian(self):
        kp_ham_dict = {}
        for order0 in range(1,self.order+1):
            ham_list = []
            for i in range(self.n_basis):
                ham_list.append(kp_k_term(order=order0, basis_index=i))          
            kp_ham_dict[order0] = np.array(ham_list)
        return kp_ham_dict


    def rotation(self,kp_list,basis_rotation,k_rotation):
        basis_matrix = self.basis_trans(basis_rotation)
        ham_list =[]
        for kp_term in kp_list:
            ham_list.append(kp_term.rotation(k_rotation))
        kp_rot_ham_list = basis_matrix@np.array(ham_list)
        # print("basis_matrix",basis_matrix)
        return kp_rot_ham_list


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
    
    
    def statics_symbol(self,kp_term_list):
        """
        用于统计matrix中现有的变量和变量个数
        """
        var_list = []
        for ele in kp_term_list:
            if isinstance(ele,kp_k_term):
                ele_var_list = ele.var_list
                new_var = [var for var in ele_var_list if var not in var_list]
                var_list = var_list + new_var
        num_var = len(var_list)
        return var_list,num_var


    def __parameter_reduce_solver(self,kp_term_list):
        """这个函数用来求解厄密和对称性约束下，matrix的独立变量，最后返回约化后的矩阵，和每个index对应的list

        Args:
            matrix (np.array): 需要处理的矩阵
        """
        n_dim = kp_term_list.shape[0]      
        for i in range(n_dim):###在确定公式前，先检查一遍，以免有空值
            if isinstance(kp_term_list[i],kp_k_term):
                if kp_term_list[i].empty:
                    kp_term_list[i] = 0   
                           
        var_list,var_num = self.statics_symbol(kp_term_list)
        num_var = len(var_list)
        equations = []
        for ele in kp_term_list:
            if ele == 0:
                continue
            for poly in ele.poly_list:
                if poly == 0:
                    continue
                symbol_index = poly.symbol_dict.keys()
                equation_array = np.zeros((num_var),dtype=np.complex64)
                for index_s in symbol_index:###一个是符号索引，一个是该符号在矩阵中的索引
                    index_n = var_list.index(index_s)
                    equation_array[index_n] = poly.symbol_dict[index_s]
                equations.append(equation_array)
        equations = np.array(equations)
        equations = self.__gaussian_elimination(equations)

        return equations,var_list

    def __equations_repalce(self,equations,var_list,kp_list):
        num_term = kp_list.shape[0]
        num_equations = equations.shape[0]
        for i in range(num_term):
            if isinstance(kp_list[i],kp_k_term):
                for k in range(num_equations):
                    kp_list[i].replace(equations[k],var_list)
                    # print("i:"+str(i)+" "+"k:"+str(k))
                    # print("kp_list",kp_list[i])
        return kp_list


    def __generator(self):
        orders = self.kp_hamiltonian_dict.keys()
        sym_ham_dict = {}
        for order in orders:
            ham_list = self.kp_hamiltonian_dict[order]
            for rotation in self.rotation_list:
                ham_rot_list = self.rotation(ham_list,basis_rotation=rotation[0],k_rotation=rotation[1])
                delta_ham_list = ham_rot_list + (-1)*ham_list
                equations,var_list = self.__parameter_reduce_solver(delta_ham_list)
                print("equaitons",equations)
                print("var_list",var_list)
                ham_list = self.__equations_repalce(equations,var_list,ham_list)

                # print("rotation",rotation)
                # print(self.statics_symbol(ham_list))
                # print("ham_list",ham_list)
            sym_ham_dict[order] = ham_list
            
        return sym_ham_dict




import time
if __name__ == "__main__":
    # dim = 3
    # start_time = time.time()
    # kp = kpHamiltonian(dim)
    # # print(kp.is_su_n_basis(kp.basis))
    # # print(kp.basis)
    # # ran_unitary = random_unitary_matrix(dim)
    # ran_unitary = np.array([[0,0,-1],[0,1,0],[-1,0,0]])
    # basis_matrix = kp.basis_trans(ran_unitary)
    # print(basis_matrix)
    # transed = np.einsum("ij,jmn->imn",basis_matrix,kp.basis)
    # rotated = ran_unitary@kp.basis@ran_unitary.conj().T
    # print(np.sum(np.abs(transed-rotated)))
    # end_time = time.time()
    # print(end_time-start_time)

    # rotation_list = [(np.array([[0,-1,0],[1,0,0],[0,0,-1]]),np.array([[0,-1,0],[1,0,0],[0,0,-1]])),
    #                  (np.array([[1,0,0],[0,-1,0],[0,0,-1]]),np.array([[1,0,0],[0,-1,0],[0,0,-1]])),
    #                  (np.array([[-1,0,0],[0,1,0],[0,0,-1]]),np.array([[-1,0,0],[0,1,0],[0,0,-1]])),
    #                  (np.array([[-1,0,0],[0,-1,0],[0,0,1]]),np.array([[-1,0,0],[0,-1,0],[0,0,1]])),
    #                  (np.array([[0,0,-1],[0,1,0],[-1,0,0]]),np.array([[0,0,-1],[0,1,0],[-1,0,0]]))]
    # kp = kpHamiltonian(dim=3,order=2,rotation_list=rotation_list)
    # print(kp.sym_hamiltonian_dict)

    rotation_list = [(np.array([[1,  0],[0, -1]]),np.array([[0,-1,0],[1,0,0],[0,0,1]]))]
    start_time= time.time()
    kp = kpHamiltonian(dim=2,order=2,rotation_list=rotation_list)
    end_time = time.time()
    print(end_time-start_time)
    print(kp.basis)
    print(kp.sym_hamiltonian_dict)
