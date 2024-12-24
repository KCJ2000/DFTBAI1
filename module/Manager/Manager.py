class manager:
    """manager类的模板类(接口)
        用于管理同一个接口下多个工具的类
    """
    def __init__(self) -> None:
        self.manager_name = None
        self.tool = self.select_tool()       
        
    def select_tool(self):
        pass
        