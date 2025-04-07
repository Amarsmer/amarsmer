from simple_launch import SimpleLauncher


def generate_launch_description():
    
    sl = SimpleLauncher(use_sim_time = False)
    
    sl.declare_arg('namespace', default_value='amarsmer')
    sl.declare_arg('jsp', True)
    sl.declare_arg('rviz', True)
    
    namespace = sl.arg('namespace')
    
    with sl.group(ns=namespace):

        # xacro parsing + change moving joints to fixed if no Gazebo here
        xacro_args = {'namespace': namespace, 'simulation': sl.sim_time}
        sl.robot_state_publisher('amarsmer_description', 'amarsmer.xacro', xacro_args=xacro_args)

        with sl.group(if_arg='jsp'):
            sl.joint_state_publisher(True)

    with sl.group(if_arg='rviz'):
        sl.rviz(sl.find('amarsmer_description', 'rov.rviz'))
        
    return sl.launch_description()
