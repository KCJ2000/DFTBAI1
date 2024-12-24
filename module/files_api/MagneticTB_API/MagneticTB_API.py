import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from files_api.API import API
import json
import sympy as sp
from sympy import Matrix,simplify,symbols
import pickle


class MagneticTB_API(API):
    def __init__(self):
        self.software_type = "Mathematica"
        self.cmd = ["math"]
        self.description = "接入"+self.software_type+"的接口\n"+"使用MagneticTB"
        
    def input_adaptor(self,orbit_init,system_init):
        """该函数本来的功能是把本项目的输入和MagneticaTB的项目进行对接,但是我现在懒得写,输入直接用MagneticaTB规定的内容24.07.14
        """
        init_info = {}
        init_info["name"] = system_init["name"]
        init_info["group_num"] = system_init["group_num"]
        init_info["lattice_vector"] = system_init["lattice_vector"]#.replace("{","{{")
        init_info["lattice_vector"] = init_info["lattice_vector"]#.replace("}","}}")        
        init_info["latpar"] = system_init["latpar"]#.replace("{","{{")
        init_info["latpar"] = init_info["latpar"]#.replace("}","}}")
        init_info["wyckpos"] = system_init["wyckpos"]#.replace("{","{{")
        init_info["wyckpos"] = init_info["wyckpos"]#.replace("}","}}")
        orbit_info = []
        for orbit_wyck in orbit_init:
            orbit_wyck_info = []
            for orbits in orbit_wyck["orbit_list"]:
                if orbit_wyck["spin_dict"][orbits] == 1:
                    orbit_wyck_info.append(orbits+"up")
                    orbit_wyck_info.append(orbits+"dn")
                else:
                    orbit_wyck_info.append(orbits)
            orbit_info.append(orbit_wyck_info)
        init_info["orbit"] = str(orbit_info).replace("[","{")
        init_info["orbit"] = init_info["orbit"].replace("]","}")
        init_info["orbit"] = init_info["orbit"].replace("\'","\"")
        init_info["n_neighbour"] = system_init["n_neighbour"]
        return init_info
    
    
    def prepare_code(self,init_info,save_file_path):
        code = ""
        code += "Quiet@Needs[\"MagneticTB`\"];\n"
        input_key = ["group_num","lattice_vector","latpar","wyckpos"]
        need_assert = not all([key in init_info.keys() for key in input_key  ])
        if not need_assert:
            AssertionError("请输入{},具体请参照MagneticTB中的群编号,输入群名称等功能敬请期待".format(init_info))
        init_code = "sgop=msgop[{group_num}];\n"
        init_code +="init[lattice->{lattice_vector},lattpar->{latpar},wyckoffposition->{wyckpos},symminformation->sgop,basisFunctions->{orbit}];\n"
        init_code = init_code.format(**init_info)
        code += init_code
        hammiltonian_code = "ham = Sum[symham[n], {{n, 1, {}}}];\n".format(init_info["n_neighbour"])###根据python语法这里需要{{和}}来对{和}进行转义
        code += hammiltonian_code
        code += self.__save_ham_code(save_file_path = save_file_path)
        return code
    
    
    def __save_ham_code(self,save_file_path):
        save_file_path = os.path.join(save_file_path,"hamiltonian.json")
        save_file_path = save_file_path.replace("\\","\\\\")
        save_code = "Nrow = Length[ham];\n"
        save_code+= "Ncol = Length[ham[[1]]];\n"
        save_code+= "hamAssociation = <||>;\n"
        save_code+= "Do[Do[hamAssociation[ToString[{row, col}]] = ToString[Expand[ham[[row, col]]], InputForm], {col, Ncol}], {row, Nrow}];\n"
        save_code+= "Export[\"{}\",   hamAssociation, \"JSON\"];".format(save_file_path)        
        return save_code
    
    
    def output_adaptor(self,init_info,save_file_path):
        with open(os.path.join(save_file_path,"hamiltonian.json"),"r") as f:
            ham_dict = json.load(f)
        ham_index_str = ham_dict.keys()
        ham_index_int = Matrix([[int(ham_index_str0[1:-1].split(",")[0]),int(ham_index_str0[1:-1].split(",")[1])] for ham_index_str0 in ham_index_str])
        n_row = max(ham_index_int[:,0])
        n_col = max(ham_index_int[:,1])
        if n_row != n_col:
            AssertionError("文件所提供的不是一个方阵，请检查文件")
        ham_matrix = sp.zeros(n_row, n_col)
        for index in ham_index_str:
            index_int = [int(index[1:-1].split(",")[0])-1,int(index[1:-1].split(",")[1])-1]
            ham_matrix[index_int] = self.__formula_trans(ham_dict[index])
        
        model_dict = {"info":init_info,"matrix":ham_matrix}
        with open(os.path.join(save_file_path,init_info["name"]+'.pkl'), 'wb') as f:
            pickle.dump(model_dict, f)
        print("完成hamiltonian构建,详情请加载{}".format(os.path.join(save_file_path,'matrix.pkl')))
        return 0
    
    
    def __formula_trans(self,formula_str):
        formula_str = formula_str.replace("E^","exp(1)**")
        formula_str = formula_str.replace("[","(")
        formula_str = formula_str.replace("]",")")        
        formula_str = formula_str.replace("Sqrt","sqrt")
        formula = simplify(formula_str)
        free_symbols = list(formula.free_symbols)
        free_symbols_list_name = [var.name for var in free_symbols]
        free_symbols_list = symbols(free_symbols_list_name,real=True)
        n_var = len(free_symbols)
        replace_dict = {free_symbols[i]:free_symbols_list[i] for i in range(n_var)}
        formula = formula.subs(replace_dict)
        return formula
    
    
    def run(self,orbit_init,system_init,save_file_path):
        args = self.input_adaptor(orbit_init,system_init)
        code = self.prepare_code(args,save_file_path)
        result = self.start_process(code)
        self.output_adaptor(args,save_file_path)
        return 0
    
    
    
    
    
    
    
if __name__ == "__main__":
    import sys
    sys.path.append("G:\\DFTBAI\\DFAITB1")
    # sysinit = {
    #     "name":"Si",
    #     "group_num":1631,
    #     "lattice_vector":"{{0,a/2,a/2},{a/2,0,a/2},{a/2,a/2,0}}",
    #     "latpar":"{a->1}",
    #     "wyckpos":"{{{1/8,1/8,1/8},{0,0,0}}}",
    #     "n_neighbour" : 2
    # }
    # orbitinit = [{"orbit_list":["px","py","pz"],"spin_dict":{"px":0,"py":0,"pz":0}}]
    # magnetic_model = MagneticTB_API()
    # file = "G:\\DFTBAI\\DFAITB1\\test\\hamiltonian\\TB\\test2"
    # result = magnetic_model.run(orbitinit,sysinit,file)
    
    
    
    