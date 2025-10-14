import numpy as np
import numbers


class Matrix_Ele:
    """
        该类是用于用数值标记表示tight-binding hamitonian matrix中的矩阵元
        每个矩阵元都按照如下方式组成：
            1、因为是tight-binding hamiltonian matrix中的矩阵元，所以只需要分exp的类型，并记录每种类型前的多项式.使用list对应储存
            2、多项式用字典储存，键是变量序号(从0开始)，值是对应变量前的系数.常数的变量序号是-1
            3、需要与加减乘除函数
            4、需要变量替换函数
    """
    def __init__(self,exp_list,formula_list):
        self.threshold = 1e-7
        self.empty = False### 默认有值
        self.exp_list,self.formula_list,self.exp_num = self.check_input(exp_list,formula_list)
        self.var_list,self.var_num = self.__extract_var()
        
        
    def check_input(self,exp_list,formula_list):
        if not isinstance(exp_list,list):
            raise AssertionError("输入exp_list类型错误，应当为list")
        elif exp_list == [] or formula_list == []:
            self.empty = True
        if not isinstance(formula_list,list):
            raise AssertionError("输入formula_list类型错误,应当为list")
        if  not self.empty:
            if not isinstance(formula_list[0],dict):
                raise AssertionError("输入formula_list内容类型错误，应当为dict")
        
        num_exp = len(exp_list)
        num_formula = len(formula_list)
        if num_exp != num_formula:
            raise AssertionError("输入的exp种类有{}种，而对应的formula有{}种".format(num_exp,num_formula))
        for i in range(num_exp):
            for j in range(i+1,num_exp):
                if np.allclose(exp_list[i],exp_list[j],rtol=self.threshold,atol= self.threshold):
                    raise AssertionError("有相同的exp项，请检查代码")
                
        exp_list = [np.array(exp_list[i]) for i in range(num_exp)]
        return exp_list,formula_list,num_exp
        
        
    def __check_dupli(self,exp_list,formula_list):
        """该函数为了判断是否有同样的exp项，并合并，但是写得不好，之后改，暂时不用

        Args:
            exp_list (_type_): _description_
            formula_list (_type_): _description_

        Returns:
            _type_: _description_
        """
        new_exp_list = []
        new_formula_list = []
        
        dupli_exp = {}
        
        num_exp = len(exp_list)
        judge_dupli = False
        for i in range(num_exp):
            judge_in = False
            for exp in new_exp_list:
                if np.allclose(exp,exp_list,rtol=self.threshold,atol=self.threshold):
                    str_exp = str(exp)
                    if str_exp in dupli_exp.keys():
                        dupli_exp[str_exp] = dupli_exp.append(i)
                        judge_dupli = True
                        break
                    else:dupli_exp[str_exp] = [i]
                    judge_in = True
            if not judge_in:
                new_exp_list.append(exp_list[i])
        if not judge_dupli:
            return exp_list,formula_list        
        
        num_new_exp = len(new_exp_list)
        for i in range(num_new_exp):
            str_exp = str(new_exp_list[i])
            index_formula = dupli_exp[str_exp]
            num_index = len(index_formula)
            formula0 = {}
            for j in range(num_index):
                formula0 = self.__formula_add(formula0,formula_list[j])
            new_formula_list.append(formula0)
        return new_exp_list,new_formula_list
                            
          
    def __extract_var(self):
        var_list = []
        for formula in self.formula_list:
            for key in formula.keys():
                if key not in var_list:
                    var_list.append(key)
        return var_list,len(var_list)
        
        
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
        
        
    def __formula_sub(self,formula1:dict,formula2:dict):
        new_formula = formula1.copy()
        symbol1 = formula1.keys()
        symbol2 = formula2.keys()
        for key in symbol2:
            if key in symbol1:
                new_formula[key] -= formula2[key]
            else:
                new_formula[key] = -formula2[key]
        
        new_formula = {key:value for key,value in new_formula.items() if abs(value)>=self.threshold}
        return new_formula
        
        
    def __add__(self,thing):
        if isinstance(thing,numbers.Number):
            formula_list_copy = self.formula_list.copy()
            for i in range(self.exp_num):
                if np.allclose(np.array([0,0,0]),self.exp_list[i],rtol=self.threshold, atol=self.threshold):
                    if -1 in self.formula_list[i].keys():
                        formula_list_copy[i][-1] += thing
                    else:
                        formula_list_copy[i][-1] = thing
                if np.abs(formula_list_copy[i][-1]) <= self.threshold:
                    del formula_list_copy[i][-1]
            return Matrix_Ele(self.exp_list,formula_list_copy)
        elif isinstance(thing,Matrix_Ele):
            new_exp = self.exp_list.copy()
            new_formula = self.formula_list.copy()
            num_exp1 = len(thing.exp_list)
            num_exp2 = len(self.exp_list)
            for i in range(num_exp1):
                judge_in = False ###标记新加进来的Matrix_Ele中是否有原本没有的exp项
                for j in range(num_exp2):
                    if np.allclose(thing.exp_list[i],self.exp_list[j],rtol=self.threshold,atol=self.threshold):
                        judge_in = True
                        new_formula[j] = self.__formula_add(thing.formula_list[i],self.formula_list[j])
                        break
                if not judge_in:### 出现了新的exp项
                    new_exp.append(thing.exp_list[i])
                    new_formula.append(thing.formula_list[i])
            
            ### 删除计算中被削掉的exp项
            num_new_exp = len(new_exp)
            indexes_to_del = [i for i in range(num_new_exp) if new_formula[i]=={}]
            indexes_to_del = sorted(indexes_to_del, reverse=True)
            for index in indexes_to_del:
                del new_exp[index]
                del new_formula[index]        
            
            return Matrix_Ele(new_exp,new_formula) 
                        
        else:raise TypeError("只能是Number类型变量或者Matrix_Ele类型变量，请检查程序")
    
    def __radd__(self,thing):
        if isinstance(thing,numbers.Number):
            return self.__add__(thing)
        
    def __sub__(self,thing):
        if isinstance(thing,numbers.Number):
            formula_list_copy = self.formula_list.copy()
            for i in range(self.exp_num):
                if np.allclose(np.array([0,0,0]),self.exp_list[i],rtol=self.threshold, atol=self.threshold):
                    if -1 in self.formula_list[i].keys():
                        formula_list_copy[i][-1] += thing
                    else:
                        formula_list_copy[i][-1] = thing
            return Matrix_Ele(self.exp_list,formula_list_copy)
        elif isinstance(thing,Matrix_Ele):
            new_exp = self.exp_list.copy()
            new_formula = self.formula_list.copy()
            num_exp1 = len(thing.exp_list)
            num_exp2 = len(self.exp_list)
            for i in range(num_exp1):
                judge_in = False ###标记新加进来的Matrix_Ele中是否有原本没有的exp项
                for j in range(num_exp2):
                    if np.allclose(thing.exp_list[i],self.exp_list[j],rtol=self.threshold,atol=self.threshold):
                        judge_in = True
                        new_formula[j] = self.__formula_sub(self.formula_list[j],thing.formula_list[i])
                        break
                if not judge_in:### 出现了新的exp项
                    new_exp.append(thing.exp_list[i])
                    new_formula.append(thing.formula_list[i])
            
            ### 删除计算中被削掉的exp项
            num_new_exp = len(new_exp)
            indexes_to_del = [i for i in range(num_new_exp) if new_formula[i]=={}]
            indexes_to_del = sorted(indexes_to_del, reverse=True)
            for index in indexes_to_del:
                del new_exp[index]
                del new_formula[index]        
            
            return Matrix_Ele(new_exp,new_formula) 
                        
        else:raise TypeError("只能是Number类型变量或者Matrix_Ele类型变量，请检查程序")
        
    
    def __mul__(self,thing):
        if isinstance(thing,numbers.Number):
            if abs(thing) < self.threshold:
                return Matrix_Ele([],[])
            else:
                num_exp = len(self.exp_list)
                formula_list_copy = self.formula_list.copy()
                indexes_to_del = []
                for i in range(num_exp):
                    formula_list_copy[i] = {key:thing*value for key,value in self.formula_list[i].items()}
                return Matrix_Ele(self.exp_list,formula_list_copy)
        else:
            raise TypeError("在tight binding model的计算中，G*H(R*k)*G^{-1} = H(k),不可能出现非数据类型变量相乘的情况，请检查代码")
        
    
    
    def __rmul__(self,thing):
        return self.__mul__(thing)
      
      
    def conjugate(self):
        new_exp_list = [-exp for exp in self.exp_list]
        new_formula_list = [{key:value.conjugate() for key,value in formula.items()}for formula in self.formula_list]
        return Matrix_Ele(new_exp_list,new_formula_list)
        
    def k_rotation(self,k_rot):
        self.exp_num = len(self.exp_list)
        new_exp_list = []
        for i in range(self.exp_num):
            new_exp_list.append(self.exp_list[i]@k_rot)
        return Matrix_Ele(new_exp_list,self.formula_list)  
                 
    def replace(self,formula_array,var_list):
        """用于变量替换，输入的array是经过高斯消元法提出变量之间相关性的
            因为我们只需要变量替换后的Matrix_Ele,所以这里我们进行本地操作即可，无需返回新的Matrix_Ele类

        Args:
            formula_array (_type_): _description_

        Raises:
            TypeError: _description_
            ValueError: _description_
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
        num_index = len(index)
        formula_array = formula_array/formula_array[index[0]]
            
        for i in range(self.exp_num):
            if index_symbol[0] in self.formula_list[i].keys():
                para = self.formula_list[i][index_symbol[0]]
                del self.formula_list[i][index_symbol[0]]
                formula_replace = {index_symbol[j]:-para*formula_array[index[j]] for j in range(1,num_index)}
                self.formula_list[i] = self.__formula_add(self.formula_list[i],formula_replace)

        ### 删除计算中被削掉的exp项
        indexes_to_del = [i for i in range(self.exp_num) if self.formula_list[i]=={}]
        indexes_to_del = sorted(indexes_to_del, reverse=True)
        for index in indexes_to_del:
            del self.exp_list[index]
            del self.formula_list[index]        
            
        if self.exp_list == []:
            self.empty = True
            self.var_list = []
            self.var_num = 0 
            self.exp_num = 0
        else:
            self.var_list,self.var_num = self.__extract_var()
            self.exp_num = len(self.exp_list)
 
    def __repr__(self):
        return str((self.exp_list,self.formula_list))
            
            
