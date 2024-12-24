这个文件夹是为了处理周期性体系而建立的，为处理科研中实际遇到的周期性体系提供了一个接口

Periodicity_System.py中定义了周期性体系中的需要处理的格子和处理格子的群操作,继承自Physics_System
        self.group_type = group_type
        self.group_name = group_name
        self.group = self.define_group()
        self.lattice = self.define_lattice()
        在Physics_System中定义了系统的名字（自己起的，便于工作的时候和写高通量workflow的时候区分）
        
（
    Periodicity_System.py中把所有可能的群都放在一起考虑了，因为周期性体系的理论比较确定只有这几种群，
    实际上这种写法并不flexible，这一切基于不会出现更多的群操作了，或者说之后可能会拓展的各种群都可以通过Point Space Magnetic群实现
    当然，之后也完全可以通过类继承把这几个群分开，增加整个项目的可维护性
）



Bravais_symop_table.json   
    属于某种Bravais Lattice的对称性操作，所有的对称性操作都在 class bravaislattice中定义

Bravais_Lattice_Jones.json 
    定义了每个Bravais Lattice的Jones标准基矢量下的坐标变换矩阵
    CubiPrim、TetrPrim、TricPrim、MonoPrim、HexaPrim定义为各自晶系的标准基
    Orthorhombic晶系标准基为[[a,0,0],[0,b,0],[0,0,c]]
    TrigPrim的标准基也是HexaPrim




