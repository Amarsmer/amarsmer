from simple_launch import SimpleLauncher


sl = SimpleLauncher(use_sim_time = False)

sl.declare_arg('namespace', default_value='amarsmer')
sl.declare_arg('jsp', True)
sl.declare_arg('rviz', True)
sl.declare_arg('thr','thrusters_plasmar2')


def launch_setup():
    
    namespace = sl.arg('namespace')
    thr_file = sl.arg('thr')
    xacro_name = 'bluerov2.xacro' if thr_file == 'thrusters_blueROV2' else 'amarsmer.xacro'
    
    with sl.group(ns=namespace):

        # xacro parsing + change moving joints to fixed if no Gazebo here
        xacro_args = {'namespace': namespace, 'simulation': sl.sim_time, 'thrusters': thr_file}
        sl.robot_state_publisher('amarsmer_description', xacro_name, xacro_args=xacro_args)

        with sl.group(if_arg='jsp'):
            sl.joint_state_publisher(True)

    with sl.group(if_arg='rviz'):
        sl.rviz(sl.find('amarsmer_description', 'rov.rviz'))
        
    return sl.launch_description()


generate_launch_description = sl.launch_description(launch_setup)
