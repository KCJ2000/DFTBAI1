files_api
掌管整个程序包文件读写工作流的模板接口，所有包中的文件IO需要从这里继承类
Hamiltonian
按要求构造各种Hamiltonian的地方，比如tight binding hamitlonian|Heisenberg hamiltonian|Hubbard model hamiltonian|Andson model等
lattice
晶格的一系列信息
parameter
负责更新学习Hamiltonian里的参数去拟合能带
physics_system 
搭建处理各种凝聚态体系的文件夹，像一个“工厂”的模板（Factory Pattern）
    Periodicity System ：
        原料：Bravais_lattice或Crystall lattice、原子或Wyckoff position
            （*这里把原子或者Wyckoff position和晶格分开是因为二者可以分开，
            如果把原子或者Wyckoff position加到晶格中，会在研究晶格的时候必须定义原子位置，但这并不必要*）
        操作：群元操作

symmetry_operation
各种群的定义和操作，比如：点群、空间群、磁群等
work
使用这个包里的种种工具定义一项功能，或者定义一套工作流，使用者可以根据案例自己写
develop_test
程序包搭建过程中的手脚架，草稿纸，测试我们要用到的函数功能
test 
针对程序包某个模块的测试



在构建晶格以及寻找近邻的时候用的是Primitive Cell基组，可以保证用最少的原子描述体系，降低模型复杂度。
在构建Oribt旋转矩阵的时候用的是Conventional Cell basis,和正交坐标系保持一致。
因为我们所用的磁群信息是从magnetic_table_bns.txt中来的，这里用的全部都是Conventional Cell的群操作，
在程序编写过程中，需要注意basis的变换，使得basis保持一致


整个程序在开发初期为了方便调试，没有用私有成员，后期需要改进，以维护程序的稳定性