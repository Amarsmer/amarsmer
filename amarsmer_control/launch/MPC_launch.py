from simple_launch import SimpleLauncher


def generate_launch_description():

    sl = SimpleLauncher(use_sim_time = True)

    sl.include('amarsmer_description', 'world_launch.py', launch_arguments={'sliders': False})

    sl.node('amarsmer_control', 'path_generation.py')

    sl.node('amarsmer_control', 'path_publisher.py')

    # sl.node('amarsmer_control', 'mpc_control.py')
    sl.node('amarsmer_control', 'ur_mpc_control.py')

    return sl.launch_description()
