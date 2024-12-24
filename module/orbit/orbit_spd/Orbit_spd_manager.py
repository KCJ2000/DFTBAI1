import os
import sys
file_path = sys.argv[0]
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))


from Manager.Manager import manager
from orbit.orbit_spd.Orbit_spd_with_spin import orbit_spd_with_spin
from orbit.orbit_spd.Orbit_spd_without_spin import orbit_spd_without_spin

class orbit_spd_manager(manager):
    def __init__(self,orbit_list=None,spin_dict=None):
        self.manager_name = "orbit_spd_manager"
        self.tool = self.select_tool(orbit_list,spin_dict)
        
        
    def select_tool(self,orbit_list,spin_dict):
        if spin_dict == None:
            return orbit_spd_without_spin(orbit_list)
        else:
            return orbit_spd_with_spin(orbit_list,spin_dict)
            
        