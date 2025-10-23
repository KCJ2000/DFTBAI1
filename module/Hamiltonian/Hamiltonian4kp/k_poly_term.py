import numpy as np 
import math
import numbers


class poly_term:
    def __init__(self,k_power_list=[0,0,0],symbol_dict={0:1}):
        self.k_power_list = np.array(k_power_list)
        if symbol_dict == {}:
            self.empty = True
        else:
            self.empty = False
        self.symbol_dict = symbol_dict
        self.threshold = 1e-7
        self.var_list,self.var_num = self.__extract_var()

    def __extract_var(self):
        var_list = []
        for key in self.symbol_dict.keys():
            if key not in var_list:
                var_list.append(key)
        return var_list,len(var_list)


    def multinomial_coefficients(self, n):
        """
        生成所有可能的 (i, j, k) 组合，使得 i + j + k = n。
        """
        combinations = []
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                combinations.append((i, j, k))
        return combinations


    def expand_polynomial(self, a, b, c, n):
        """
        展开 (a k_x + b k_y + c k_z)^n 并计算每一项的系数。
        """
        combinations = self.multinomial_coefficients(n)
        terms = []
        
        for i, j, k in combinations:
            coefficient = (math.factorial(n) / (math.factorial(i) * math.factorial(j) * math.factorial(k))) * (a**i) * (b**j) * (c**k)
            if np.isclose(coefficient, 0.0, atol=1e-7):
                continue
            term = (np.array([i,j,k]),coefficient)
            terms.append(term)
        
        return terms
        

    def multi_poly(self,poly_list1,poly_list2):
        poly_list0 = []
        for poly1 in poly_list1:
            for poly2 in poly_list2:
                poly_power = poly1[0] + poly2[0]
                coef = poly1[1]*poly2[1]
                poly_list0.append((poly_power,coef))
        poly_list = []
        have_index = []
        n_term = len(poly_list0)
        for i in range(n_term):
            if i in have_index:
                continue
            coef = poly_list0[i][1]
            have_index.append(i)
            for j in range(i+1,n_term):
                if j in have_index:
                    continue
                if np.sum(np.abs(poly_list0[i][0]-poly_list0[j][0]))<1e-7:
                    coef += poly_list0[j][1]
                    have_index.append(j)
            poly_list.append((poly_list0[i][0],coef))
        return poly_list


    def rotation(self,rotation_matrix):
        if rotation_matrix.shape == np.ones(3).shape:
            raise ValueError("不是3*3类型的矩阵")
        if np.sum(np.abs(rotation_matrix.T@rotation_matrix-np.eye(3)))>1e-8:
            raise ValueError("不是归一矩阵")

        poly_x = self.expand_polynomial(rotation_matrix[0][0],
                                        rotation_matrix[0][1],
                                        rotation_matrix[0][2],self.k_power_list[0])
        
        poly_y = self.expand_polynomial(rotation_matrix[1][0],
                                        rotation_matrix[1][1],
                                        rotation_matrix[1][2],self.k_power_list[1])
        
        poly_z = self.expand_polynomial(rotation_matrix[2][0],
                                        rotation_matrix[2][1],
                                        rotation_matrix[2][2],self.k_power_list[2])
        
        poly = self.multi_poly(poly_x,poly_y)
        poly = self.multi_poly(poly,poly_z)

        poly_list = []
        for poly0 in poly:
            symbol = {key:poly0[1]*value for key,value in self.symbol_dict.items()}
            poly_list.append(poly_term(poly0[0],symbol))
        return poly_list
    

    def __formula_add(self,formula1:dict,formula2:dict):
        new_formula = formula1.copy()
        symbol1 = formula1.keys()
        symbol2 = formula2.keys()
        for key in symbol2:
            if key in symbol1:
                new_formula[key] += formula2[key]
            else:
                new_formula[key] = formula2[key]
        
        new_formula = {key:value for key,value in new_formula.items() if abs(value)>=self.threshold}
        return new_formula


    def __mul__(self,thing):
        if isinstance(thing,numbers.Number):
            if abs(thing) < self.threshold:
                return 0
            else:
                new_symbol_dict = {key:thing*value for key,value in self.symbol_dict.items()}
                return poly_term(self.k_power_list,new_symbol_dict)
        else:
            raise TypeError("在tight binding model的计算中，G*H(R*k)*G^{-1} = H(k),不可能出现非数据类型变量相乘的情况，请检查代码")    
    
    def __rmul__(self,thing):
        return self.__mul__(thing)

    def __add__(self,thing):
        if isinstance(thing,poly_term):
            if np.sum(np.abs(self.k_power_list-thing.k_power_list)) < self.threshold:
                formula = self.__formula_add(thing.symbol_dict,self.symbol_dict)
                return poly_term(self.k_power_list,symbol_dict=formula)
            else:
                raise TypeError(f"{thing.k_power_list} term and {self.k_power_list} term 不能相加")
        elif abs(thing) < self.threshold:
            return poly_term(self.k_power_list,self.symbol_dict)
        else:
            raise TypeError("只能和poly_term class相加")

    def __radd__(self,thing):
        return self.__add__(thing)

    def __repr__(self):
        return str(self.symbol_dict) + "*" + f"k_x^{self.k_power_list[0]} * k_y^{self.k_power_list[1]} * k_z^{self.k_power_list[2]}"

    def replace(self,formula_array,index_symbol):
        num_index = len(index_symbol)
        para = self.symbol_dict[index_symbol[0]]
        del self.symbol_dict[index_symbol[0]]
        formula_replace = {index_symbol[j]:-para*formula_array[j] for j in range(1,num_index)}
        self.symbol_dict = self.__formula_add(self.symbol_dict,formula_replace)
        self.var_list,self.var_num = self.__extract_var()
        # print("symbol_dict",self.symbol_dict)
        # print("replace:",self.k_power_list,self.symbol_dict)
        if self.symbol_dict == {}:
            self.empty = True



class kp_k_term:
    def __init__(self,order=0,basis_index=0,poly_list=None):
        """
        order 是多项式最高的阶数
        basis_index 是对应的basis的编号，用于设置变量的序号,只在init的时候有用
        """
        self.order = order
        self.basis_index = basis_index
        self.symbol_num, self.xyz_term_list = self.init_term(order)
        if poly_list == None:
            self.poly_list = self.init_poly()
        else:
            self.poly_list = poly_list
        self.threshold = 1e-7
        self.var_list,self.symbol_num = self.statics_symbol()
        self.empty = True
        for poly in self.poly_list:
            if poly != 0:
                self.empty = False
                break


    def statics_symbol(self):
        """
        用于统计matrix中现有的变量和变量个数
        """
        var_list = []
        for ele in self.poly_list:
            if isinstance(ele,poly_term):
                ele_var_list = ele.var_list
                new_var = [var for var in ele_var_list if var not in var_list]
                var_list = var_list + new_var
        num_var = len(var_list)
        return var_list,num_var


    def init_term(self, order):
        """
        生成所有满足 i + j + k = n 的非负整数组合 (i, j, k)。
        """
        combinations = []
        for i in range(order + 1):
            for j in range(order + 1 - i):
                k = order - i - j
                combinations.append(np.array([i, j, k]))
        return math.comb(order + 2, 2),combinations

    def init_poly(self):
        poly_list = []
        symbol_index = self.basis_index*self.symbol_num
        for xyz_power in self.xyz_term_list:
            poly_list.append(poly_term(xyz_power,{symbol_index:1}))
            symbol_index += 1
        return poly_list

    def __formula_add(self,formula1:dict,formula2:dict):
        new_formula = formula1.copy()
        symbol1 = formula1.keys()
        symbol2 = formula2.keys()
        for key in symbol2:
            if key in symbol1:
                new_formula[key] += formula2[key]
            else:
                new_formula[key] = formula2[key]
        
        new_formula = {key:value for key,value in new_formula.items() if abs(value)>=self.threshold}
        return new_formula

    def rotation(self,rotation):
        polys_list = []
        for poly0 in self.poly_list:
            if isinstance(poly0,poly_term):
                polys_list += poly0.rotation(rotation)# list拼接
        n_poly = len(polys_list)
        poly_final_list = []
        have_detect = []
        for k_power in self.xyz_term_list:
            new_formula = {}
            for i in range(n_poly):
                if i in have_detect:
                    continue
                if np.sum(np.abs(polys_list[i].k_power_list - k_power)) < self.threshold:
                    have_detect.append(i)
                    new_formula = self.__formula_add(new_formula,polys_list[i].symbol_dict)
            poly_final_list.append(poly_term(k_power_list=k_power,symbol_dict=new_formula))

        return  kp_k_term(order=self.order,basis_index=self.basis_index,poly_list=poly_final_list)


    def __mul__(self,thing):
        if isinstance(thing,numbers.Number):
            if abs(thing) < self.threshold:
                return 0
            else:
                poly = self.poly_list.copy()
                n_poly = len(self.poly_list)
                for i in range(n_poly):
                    poly[i] = self.poly_list[i]*thing
                return kp_k_term(self.order,self.basis_index,poly)
        else:
            raise TypeError("在tight binding model的计算中，G*H(R*k)*G^{-1} = H(k),不可能出现非数据类型变量相乘的情况，请检查代码")       

    def __rmul__(self,thing):
        return self.__mul__(thing)

    def __add__(self,thing):
        if isinstance(thing,kp_k_term):
            if thing.order == self.order:
                poly1_list = thing.poly_list.copy()
                poly2_list = self.poly_list.copy()
                poly_list = []
                for k_power in self.xyz_term_list:
                    poly_need1 = 0
                    poly_need2 = 0
                    for poly1 in poly1_list:
                        if not isinstance(poly1,poly_term):
                            continue
                        if np.sum(np.abs(k_power-poly1.k_power_list)) < self.threshold:
                            poly_need1 = poly1
                            break
                    for poly2 in poly2_list:
                        if not isinstance(poly2,poly_term):
                            continue
                        if np.sum(np.abs(k_power-poly2.k_power_list)) < self.threshold:
                            poly_need2 = poly2
                            break
                    poly_list.append(poly_need1 + poly_need2)
                return kp_k_term(self.order,self.basis_index,poly_list)
            else:
                raise TypeError("不同阶的k term不可以相加")
        elif abs(thing) < self.threshold:
            return kp_k_term(self.order,self.basis_index,self.poly_list)
        else:
            raise TypeError("不是 kp_k_term class")

    def __radd__(self,thing):
        return self.__add__(thing)

    def __repr__(self):
        return str(self.poly_list)

    def replace(self,formula_array,var_list):
        """
        用于变量替换，输入的array是经过高斯消元法提出变量之间相关性的
        因为我们只需要变量替换后的kp_k_term,所以这里我们进行本地操作即可，无需返回新的Matrix_Ele类
        """
        if type(formula_array) != type(np.array([0])):
            raise TypeError("应该是numpy.array类型变量，而非{}".format(type(formula_array)))
        if len(formula_array.shape) != 1:
            raise ValueError("输入的numpy.array的shape应该是1，请检查")

        index = np.where(formula_array)[0]
        index_symbol = [var_list[index0] for index0 in index]
        if len(index) == 0:###等式为空，不操作
            return 0
        if index_symbol[0] not in self.var_list:###没有操作对象，不操作
            return 0
        
        formula_array = formula_array/formula_array[index[0]]
        
        n_poly = len(self.poly_list)
        for i in range(n_poly):
            if not isinstance(self.poly_list[i],poly_term):
                continue
            if index_symbol[0] in self.poly_list[i].symbol_dict.keys():
                self.poly_list[i].replace(formula_array[index],index_symbol)
        


        ### 让所有空项，得0
        n_poly = len(self.poly_list)
        for i in range(n_poly):
            if isinstance(self.poly_list[i],poly_term):
                if self.poly_list[i].empty:
                    self.poly_list[i] = 0

        self.var_list,self.symbol_num = self.statics_symbol()



import time
if __name__ == "__main__":
    poly1 = poly_term(k_power_list=[2,0,0],symbol_dict={0:1})
    poly2 = poly_term(k_power_list=[0,1,0],symbol_dict={1:0.5})
    print(poly1+poly2)
    # print(poly1.expand_polynomial(0.5,np.sqrt(3)/2,0,2))
    polya = [(np.array([1,1,0]),0.89),(np.array([2,1,0]),0.45)]
    polyb = [(np.array([0,1,0]),1.2),(np.array([1,0,0]),0.2),(np.array([1,1,0]),0.5)]
    rotation = np.array([[np.sqrt(3)/2,0.5,0],[-0.5,np.sqrt(3)/2,0],[0,0,1]])
    print(rotation.shape)

    start = time.time()
    kp_mat_ele1 = kp_k_term(order=3,basis_index=0)
    sigma_y = np.array([[0,-1j],[1j,0]])
    print(sigma_y*np.array(kp_mat_ele1))
    kp_mat_ele1.rotation(rotation)
    print(kp_mat_ele1.poly_list)
    kp_mat_ele2 = kp_k_term(order=3,basis_index=1)
    print("-------------------------------------")
    kp_mat_ele = kp_mat_ele1+kp_mat_ele2
    print(kp_mat_ele.var_list)
    end = time.time()
    print(end-start)
    # print(poly1.multi_poly(polya,polyb))

