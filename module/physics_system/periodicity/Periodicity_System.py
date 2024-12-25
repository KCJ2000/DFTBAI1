import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import json
import numpy as np
from lattice.Bravais_Lattice.Bravais_Lattice import bravaislattice
from symmetry_operation.mag_group.mg_operation import MagneticGroupOp
from symmetry_operation.point_group.pg_operation import PointGroupOp
from symmetry_operation.space_group.sg_operation import SpaceGroupOp
from physics_system.Physics_System import Phy_Sys

class PeriodicityPhysicsSystem(Phy_Sys):
    def __init__(self,sys_name=None,group_type=None,group_name=None,
                 lattice_type=None,lattice_parameter={},atompos = [],magdirect = [],
                 n_neighbour = 3):
        super(PeriodicityPhysicsSystem,self).__init__(sys_name)
        self.group_type = group_type  ### 定义这里定义哪个群类，比如是点群还是空间群还是磁群
        self.group_name = group_name
        self.lattice_type = lattice_type
        self.lattice_parameter = lattice_parameter
        self.group = self.define_group()
        self.lattice = self.define_lattice()
        self.atompos = atompos
        self.magdirect = magdirect
        self.wyckoffpos = []
        self.have_check = False ###是否通过对称性操作生成了所有的原子占位
        self.atom_symmetry_check()
        self.n_neighbour = n_neighbour
        ###提取原子之间的近邻关系，分别得到不同晶格间原子的distance,原子间近邻关系，和对应近邻的距离
        self.atom_distance ,self.atom_neighbour,self.have_detect = self.atom_bond()
        ###通过上一步的结果整理，返回的是记录self.atompos中的第几号原子以及原子坐标的dict
        self.neighbour_table = self.__neighbour_table_construct()
        # self.wcykoff_position, self.n_wcykoff = self.__wcykoff_analysis()
        
        
    def define_group(self):
        group_type_list = ["Point Group","Space Group","Magnetic Group"]
        if self.group_type == None:
            raise AssertionError("没定义group_type")
        else:
            if self.group_type == "Point Group":
                return PointGroupOp(self.group_name)
            elif self.group_type == "Space Group":
                return SpaceGroupOp(self.group_name)
            elif self.group_type == "Magnetic Group":
                return MagneticGroupOp(self.group_name)
            else:
                raise AssertionError("还没定义这种类，请从以下类中选取\n{}".format(group_type_list))

            
    def define_lattice(self):
        if self.group != None and self.lattice_type != None:
            group_info = self.group.__repr__().split()
            point_group_name = group_info[-1].split(":")[-1]
            lattice = bravaislattice(bravais_lattice_type=self.lattice_type)
            folder_path = os.path.dirname(__file__)
            with open(os.path.join(folder_path,"Crystall2point_group.json")) as f:
                crystall2point_group_table = json.load(f)
            if point_group_name not in crystall2point_group_table[lattice.lattice_type]:
                recommend = [key for key,value in crystall2point_group_table.items() if point_group_name in value]
                ### 到这一步，point_group_name 一定可以在这里找到，因为是程序的定义直接得到的
                raise AssertionError("所定义的晶格{}和所定义的对称性操作点群{}并不匹配，按照Crystall2point_group.json中的记录，应该是{}类型的".format(
                    lattice.lattice_type,point_group_name,recommend))
            else:
                return lattice
            
        if self.group != None and self.lattice_type == None:
            return bravaislattice(bravais_lattice_type=self.lattice_type)
            
        if self.group == None:
            return bravaislattice(bravais_lattice_type=self.lattice_type)
            
            
    def __repr__(self):
        if self.group_type == "Magnetic Group":
            return "sys_name:"+self.sys_name+"\nlattice information:"+self.lattice.__repr__()+"\natom_position:\n"+str(self.atompos)+"\nmag_direction:\n"+str(self.magdirect)+"\nGroup information:\n"+self.group.__repr__()
        else:
            return "sys_name:"+self.sys_name+"\nlattice information:"+self.lattice.__repr__()+"\natom_position:\n"+str(self.atompos) +"\nGroup information:"+self.group.__repr__()

    
    def atom_symmetry_check(self):
        """
            所有输入的安全性检测
            Magnetic Group、Space Group、Point Group的定义和输入定义之间冲突的安全性检测
            Wyckoff positon的补全和分析,结果储存在self.wyckoffpos中
            所有的atompos和magdirect的坐标表示都使用Jones表示(方向按照晶格矢量来)
        """
        if self.have_check:
            return 0
        if not self.atompos :###无论什么群，原子坐标必须有
            raise AssertionError("没输入atom position(记得输入分数坐标)")
        if self.group == None:
            raise AssertionError("没定义group，请定义")
        if self.lattice == None:
            raise AssertionError("没定义lattice，请定义")
        
        
        ### 准备self.atom_rep
        ### Magnetic Group的
        n_atom = len(self.atompos)            
        if (not self.magdirect) and self.group_type == "Magnetic Group":
            raise AssertionError("现在使用的是Magnetic Group,但是没输入magdirect。可以选择更改群类型比如Space Group，或者输入magdirect")
        elif self.atompos  and self.magdirect :###如果都有输入，证明是磁群操作（上面有稳定性判断）        
            if len(self.magdirect) != len(self.atompos):
                raise AssertionError("magdirct和atompos的原子数不同，请检查输入")
            joint = np.ones((n_atom,1))
            atom_pos = np.array(self.atompos)
            mag_direct = np.array(self.magdirect)
            if len(atom_pos.shape) == 1 or len(mag_direct.shape) == 1:
                raise AssertionError("输入的常见错误，一个原子的list是[[float,float,float]],而不是[float,float,float]")
            self.atom_rep = np.concatenate((atom_pos, joint,mag_direct), axis=-1)
            
        ### Space Group的
        if self.group_type == "Space Group":
            atom_pos = np.array(self.atompos)
            joint = np.ones((n_atom,1))
            if len(atom_pos.shape) == 1:
                raise AssertionError("输入的常见错误，一个原子的list是[[float,float,float]],而不是[float,float,float]")
            self.atom_rep = np.concatenate((atom_pos,joint),axis=-1)
            
        ### Point Group的
        if self.group_type == "Point Group":
            self.atom_rep = np.array(self.atompos)
        
        
        n_atom = self.atom_rep.shape[0]###self.atom_rep的维度(n_atom,dim_rep)
        operations_table = self.group.group_operation_rep
        operations_table = self.group.basis_trans(np.linalg.inv(self.lattice.Conven2Prim_matrix))
        operations_name = operations_table.keys()
        
        ###通过旋转操作找到所有的等价位置，生成Wyckoff position
        Wyckoff_position = []###应该生成（n_wyck,n_atom,3）的list记录同属于一个Wyckoff position的坐标
        i = 0
        while i <n_atom:
            Wyckoff_position0 = []
            Wyckoff_position0.append(self.atom_rep[i])
            # print(i)
            for operation0 in operations_name:###可以证明所有操作对orgin_pos全部一次就能找全其Wyckoff position相对应的atompos
                output = np.matmul(operations_table[operation0],self.atom_rep[i])
                output[0:3] = (output[0:3])%1
                need_update = not any(np.allclose(output, atom_rep0,rtol=1.e-5, atol=1.e-7) for atom_rep0 in Wyckoff_position0)###不在已发现的坐标里，说明发现新的，更新
                if need_update:
                    Wyckoff_position0.append(output)
                    unequality = []###记录self.atom_rep中和output不等的坐标，赋值为新的self.atom_rep，相当于把相等的删除了
                    for j in range(n_atom):
                        if not np.allclose(output,self.atom_rep[j],rtol=1.e-5,atol=1.e-7):
                            unequality.append(self.atom_rep[j])
                    self.atom_rep = np.array(unequality)
                n_atom = self.atom_rep.shape[0]
            i += 1
            Wyckoff_position0 = np.array(Wyckoff_position0)
            Wyckoff_position.append(Wyckoff_position0)
        
        self.wyckoffpos = Wyckoff_position
        self.atom_rep = np.concatenate(Wyckoff_position,axis=0)      
        self.n_atom = len(self.atom_rep)
        self.dim_rep = len(self.atom_rep[0])
                
            
        ###把所有的位置再提取出来
        if self.group_type == "Magnetic Group":
            self.atompos = self.atom_rep[:,0:3]
            self.magdirect = self.atom_rep[:,4:7]
        elif self.group_type == "Space Group" or self.group_type == "Point Group":
            self.atompos = self.atom_rep[:,0:3]
        self.have_check = True     
        
        
    def atom_bond(self):
        """
        atom_bond是输出某原子是某原子几阶近邻的函数
        输出形式是
        1、 atom_distance: n_atom1*n_atom2(二者相等)的list,每个element都是dict，记录某平移晶格中的atom2是中心晶格中atom1的距离
        2、 atom_neighbour: n_atom1*n_atom2(二者相等)的list,每个element都是dict，记录某平移晶格中的atom2是中心晶格中atom1的第几阶近邻
        """
        if self.lattice_parameter == {}:
            raise AssertionError("没输入bravaise lattice坐标参数，无法计算原子之间的连接")
        free_symbol = set()###用set可以直接排除相同元素,得到lattice_vector中的独立参数
        for elem in self.lattice.lattice_vector:
            free_symbol.update(elem.free_symbols) 
        free_symbol = [str(symbol) for symbol in free_symbol]
        paras = self.lattice_parameter.keys()
        exist = all(para0 in free_symbol for para0 in paras)
        if not exist:
            raise AssertionError("该晶格中独立参数为{}，请全部输入".format(free_symbol))    
        ### 经过一系列安全性检测后，我们为lattice vector赋值
        lattice_vector_with_para = self.lattice.lattice_vector.subs(self.lattice_parameter)
        lattice_vector_with_para = np.array(lattice_vector_with_para,dtype=float)
        ### 把self.atompos中的分数坐标用赋过值的晶格转化成实际坐标(以Primitive Cell的lattice_vector为基组)
        atom_pos = np.array(self.atompos)
        atom_pos = np.matmul(atom_pos,lattice_vector_with_para)
        n_atom = atom_pos.shape[0]
        ### 初始化atom_distance和atom_neighbour,have_detect
        ### have_detect 是记录已经探索到的距离数值，可以用它对距离数值进行排序，然后用atom_distance来直接得到atom_neighbour
        atom_distance = []        
        atom_neighbour = []
        have_detect = []
        for i in range(n_atom):
            atom_distance0 = []
            atom_neighbour0 = []
            have_detect.append([])
            for j in range(n_atom):
                distance_ele = {}
                neighbour_ele = {}
                atom_distance0.append(distance_ele)
                atom_neighbour0.append(neighbour_ele)
            atom_distance.append(atom_distance0)
            atom_neighbour.append(atom_neighbour0)
            
        ###调用self.atom_bond_lattice_distance生成atom_distance的矩阵内容
        ###对于lattice的遍历，我们采取index的绝对值求和从0往上排的策略(一圈圈向外扩展)
        ### 因为python循环的index性质，abs_index的值比实际index大了1
        ### 这个while的作用是得到self.n_neighbour范围之内的所有原子，会遍历到所需要的更外面的一圈（需要这种冗余），以确保没有self.n_neighbour范围内的原子被拉下
        
        abs_index = 1
        while True:
            ### 某abs_index限制下得到平移的lattice的index
            index_list = []
            need_break_dict = {}###初始化全是False，当下面判断已经全部超出了self.n_neighbour的所需范围，就全为true,break
            for x_index in range(-abs_index,abs_index):
                for y_index in range(-abs_index,abs_index):
                    for z_index in range(-abs_index,abs_index):
                        if max([abs(x_index),abs(y_index),abs(z_index)]) == abs_index-1:
                            index_list.append(np.array([x_index,y_index,z_index]))
                            need_break_dict[str(np.array([x_index,y_index,z_index]))] = False
            
            for index0 in index_list:
                trans_vector = np.matmul(index0,lattice_vector_with_para)
                D = self.atom_bond_lattice_distance(atom_pos,trans_vector)
                ###have_touched 判断是否have_dectect中所有原子都已经触碰到了n_neighbour所要求接触到的近邻
                have_touched = all(len(have_detect[i])>=self.n_neighbour for i in range(n_atom))
                for i in range(n_atom):
                    need_break = True
                    for j in range(n_atom):
                        if have_touched:### 在该index中还有所要探索的近邻数，那么说明不应该break
                            if D[i][j] <= have_detect[i][self.n_neighbour-1]:  
                                ### print(index0,D[i][j],have_detect[i][self.n_neighbour])
                                need_break = False   
                        else:need_break = False###还没接触到所要求接触到的近邻，不应该break                
                        atom_distance[i][j][str(index0)] = D[i][j]
                        ### print(D[i][j],[abs(D[i][j] - have_detect[i][k])<1e-8 for k in range(len(have_detect[i]))],have_detect[i])
                        ### have_detect空或者没探查到过这个距离 并且 这个距离没有超出self.n_neighbour需要的探查距离
                        ###     然后添加元素，并进行排序
                        if (not have_detect[i] or not any(abs(D[i][j] - have_detect[i][k])<1e-8 for k in range(len(have_detect[i])))) and not need_break:
                            have_detect[i].append(D[i][j])
                            have_detect[i].sort()
                need_break_dict[str(index0)] = need_break
            # print(need_break_dict)
            if all(need_break_dict.values()):
                print("atom间距搜索达到近邻要求")
                break            
            
            if abs_index>self.n_neighbour:###就算是全跟另一个晶格的自己做邻居也足够了
                print("达到迭代极限,请检查体系或者代码，确定不是程序问题导致的该情况，这种情况一般不易出现")
                break
            
            abs_index += 1
        ###开始通过have_detect和atom_distance得到atom_neighbour
        for i in range(n_atom):
            for j in range(n_atom):
                index_key = atom_distance[i][j].keys()
                n_detected = len(have_detect[i])
                # print("n_detected",n_detected)
                for index0 in index_key:
                    for k in range(n_detected):
                        # print(atom_distance[i][j][index0],have_detect[i][k],atom_distance[i][j][index0] == have_detect[i][k],index0)
                        if abs(atom_distance[i][j][index0] - have_detect[i][k])<=1e-9:
                            atom_neighbour[i][j][index0] = k
                            break
        
        
        # print("have_detect:\n",have_detect)
        # print("atom_distance:\n",atom_distance)
        # print("atom_neighbour:\n",atom_neighbour)
        # print(atom_distance)
        # for i in range(n_atom):        
        #     print(have_detect[i][0:self.n_neighbour])
        # print("abs_index",abs_index)

        return atom_distance,atom_neighbour,have_detect

           
    def atom_bond_lattice_distance(self,atom_pos,vector):
        """
            该函数是为了配合self.atom_bond函数所建立的
            作用是计算某个平移的lattice中的atom2和中心lattice中的atom1的距离
            矩阵元ij，代表中心晶格原子i到平移晶格原子j的距离
            return (np.array):
                shape:n_atom,n_atom
            
        """
        n_atom = atom_pos.shape[0]
        atom_pos1 = np.expand_dims(atom_pos,axis=0)
        atom_pos1 = np.repeat(atom_pos1,axis=0,repeats=n_atom)
        atom_pos2 = np.expand_dims(atom_pos,axis=1)
        atom_pos2 = np.repeat(atom_pos2,axis=1,repeats=n_atom)
        vector = np.expand_dims(vector,axis=0)
        vector = np.expand_dims(vector,axis=0)
        vector = np.repeat(vector,axis=0,repeats=n_atom)
        vector = np.repeat(vector,axis=1,repeats=n_atom)
        delta_atom_pos = atom_pos2 - atom_pos1 + vector
        D = (np.sum(delta_atom_pos**2,axis=-1)**0.5).transpose()
        return D
        
   
    def __neighbour_table_construct(self):
        """
            针对self.atom_neighbour进行处理
            return(list->dict->list):
                neighbour_table:
                    按照self.atompos中的原子个数和顺序构建list，list中每个元素是一个dict
                    每个dict是对应原子的第几近邻->近邻原子的索引和相对坐标的列表list
        """
        neighbour_table = []
        n_atom = len(self.atompos)
        for i in range(n_atom):
            neighbour_table.append({})
        for i in range(n_atom):
            for j in range(self.n_neighbour):
                neighbour_table[i][str(j)] = []
        for i in range(n_atom):
            for j in range(n_atom):
                index_key =  self.atom_neighbour[i][j].keys()
                for index0 in index_key:
                    if self.atom_neighbour[i][j][index0] < self.n_neighbour:
                        index = index0[1:-1].split()
                        index = np.array([float(index[0]),float(index[1]),float(index[2])])
                        neighbour_table[i][str(self.atom_neighbour[i][j][index0])].append((j,self.atompos[j]+index))
        return neighbour_table
    
    
    def neighbour_table_plot(self):
        pass
        
    
    # def __wcykoff_analysis(self):
    #     """
    #         分析有多少个Wcykoff position,进行分类，相同的Wcykoff position在TB中只能用一套轨道
    #     """
    #     if not self.have_check:
    #         raise AssertionError("请先运行self.atom_symmertry_check()")
    #     ### 
    #     for 
        
        
import time
if __name__ == "__main__":
    start_time = time.time()
    # atompos_list = [[1/3,2/3,0],[0,0,0]]
    # magdirect_list = [[1/2,0,0],[0,0,0]]
    # input = {
    #     "sys_name":"test",
    #     "group_type":"Magnetic Group",
    #     "group_name":"191.234",
    #     "lattice_type":"HexaPrim",
    #     "lattice_parameter":{"a":1,"c":3},
    #     "atompos":atompos_list,
    #     "magdirect":magdirect_list
    # }
    atompos_list = [[1/8,1/8,1/8]]
    magdirect_list = [[0,0,0]]
    sysinit = {
                "sys_name":"Si",
                "group_type":"Magnetic Group",
                "group_name":"227.128",
                "lattice_type":"CubiFace",
                "lattice_parameter":{"a":1},
                "atompos":[[1/8,1/8,1/8]],
                "magdirect":[[0,0,0]],
                "n_neighbour":2
                }
    # sysinit = {
    #             "sys_name":"Si",
    #             "group_type":"Space Group",
    #             "group_name":"227",
    #             "lattice_type":"CubiFace",
    #             "lattice_parameter":{"a":1},
    #             "atompos":[[1/8,1/8,1/8]],
    #             # "magdirect":[[0,0,0]],
    #             "n_neighbour":2
    #             }
    system_test = PeriodicityPhysicsSystem(**sysinit)
    # print(system_test)
    # system_test.atom_symmetry_check()
    # system_test.atom_bond()
    end_time = time.time()
    # print(system.atom_distance[0][1])
    # print(system.atom_neighbour)
    # print(system.have_detect)
    # print(system_test.neighbour_table)
    file_path = "G:\\DFTBAI\\DFAITB1\\develop_test\\test_Periodicity_sys\\output_Periodicity_System.txt"
    f = open(file_path,"w")
    f.write(f"{str(system_test.wyckoffpos)}"+"\n")
    # print(np.matmul(system_test.lattice,np.transpose(system_test.lattice)))
    f.write(f"{str(system_test.atom_neighbour)}"+"\n")
    f.write(f"{str(system_test.neighbour_table)}"+"\n")
    f.write(f"{str(system_test.atom_distance)}"+"\n")
    f.write(f"{str(system_test.group.basis_trans(np.linalg.inv(system_test.lattice.Conven2Prim_matrix)))}")
    # print(np.array(system_test.lattice.lattice_vector.subs(system_test.lattice_parameter),dtype=float))
    print(np.linalg.inv(np.array(system_test.lattice.lattice_vector.subs(system_test.lattice_parameter),dtype=float)))
    print(system_test.lattice.repi_lattice_vector)
    print(system_test.lattice.repi_lattice_vector@system_test.lattice.lattice_vector.transpose())
    print(system_test.group_type)
    print("program time",end_time-start_time)
    # system.atom_bond_lattice()