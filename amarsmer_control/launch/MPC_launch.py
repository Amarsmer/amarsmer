from simple_launch import SimpleLauncher


def generate_launch_description():

    sl = SimpleLauncher(use_sim_time = True)

    sl.declare_arg('namespace', default_value='amarsmer')
    sl.declare_arg('rviz', default_value=True)

    sl.include('amarsmer_description','world_launch.py', launch_arguments={'sliders': False})

    
    # sl.node('thruster_manager', 'publish_wrenches',
    #         parameters={'control_frame': sl.arg('namespace') / 'base_link',
    #                     'use_gz_topics': sl.sim_time})

    sl.node('amarsmer_control', 'path_generation.py')

    sl.node('amarsmer_control', 'path_publisher.py')

    sl.node('amarsmer_control', 'mpc_control.py')

    return sl.launch_description()
