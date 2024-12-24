Point_Group.json中记录32个点群的名称和每个点群中各自的旋转对称性操作,点群名称是Schoenflies表示
Shubnikov2Schoenflies.json中是Shubnikov名称表示和Schoenflies名称表示之间的对应关系
xyz_operation.json中是所有旋转操作的xyz表示
xyz_operation_Litvin.json中是Litvin书中定义的所有旋转操作的xyz表示

pg_operation.py中有PointGroupOP类
    初始化：
        group_name 
        记录了群元的旋转矩阵Jones' faithful representation
    
    操作：
        矩阵乘法
