这个文件夹定义了跟格子（lattice）有关的类
Lattice.py中定义了lattice类是接口类，负责定义7大晶系（之后涉及无序体系和准晶体系会承担更多功能），一切与晶格有关的类都继承lattice
Bravais_Lattice.py中定义bravaiselattice
    1、定义该晶格时14种Bravais Lattice 中的哪一种
    2、从Bravais_Lattice_Vector.json中读取该晶格的晶格矢量
    3、计算晶格体积
    4、计算该晶格的倒空间的晶格矢量


Bravais_Lattice_Jones.json 定义了每个Bravais Lattice的Jones标准基矢量下的坐标变换矩阵
    CubiPrim、TetrPrim、TricPrim、MonoPrim、HexaPrim定义为各自晶系的标准基
    Orthorhombic晶系标准基为[[a,0,0],[0,b,0],[0,0,c]]
    TrigPrim的标准基也是HexaPrim



