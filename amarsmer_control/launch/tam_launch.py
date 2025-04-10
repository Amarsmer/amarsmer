from simple_launch import SimpleLauncher

def generate_launch_description():

    sl = SimpleLauncher(use_sim_time = True)

    with sl.group(ns = 'amarsmer'):

        sl.node('thruster_manager', 'thruster_manager_node',
                parameters = {'control_frame': 'amarsmer/base_link'})




    return sl.launch_description()




