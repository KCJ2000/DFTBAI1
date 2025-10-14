import os
import sys
file_path = sys.argv[0]
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))

import numpy as np

import pickle
import json
import numbers

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


class TBHamiltonian(Hamiltonian):
    def __init__(self, dim=1):
        super().__init__(dim)
        self.name = "kp hamiltonian"
        self.basis = self.su_n_generators(dim)


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
        # generators = [np.zeros((n,n), dtype=complex)]
        generators = []
        
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
        print(basis.shape)
        basis = np.transpose(basis,axes=(0, 1, 3, 2)).conj()
        basis_matrix = np.einsum("ijmn,inm -> ij",basis,self.basis)
        basis_matrix = np.transpose(basis_matrix,axes=(1,0))
        return basis_matrix



import time
if __name__ == "__main__":
    dim = 2
    start_time = time.time()
    kp = TBHamiltonian(dim)
    # print(kp.is_su_n_basis(kp.basis))
    # print(kp.basis)
    ran_unitary = random_unitary_matrix(dim)
    basis_matrix = kp.basis_trans(ran_unitary)
    print(basis_matrix)
    transed = np.einsum("ij,jmn->imn",basis_matrix,kp.basis)
    rotated = ran_unitary@kp.basis@ran_unitary.conj().T
    print(np.sum(np.abs(transed-rotated)))
    end_time = time.time()
    print(end_time-start_time)