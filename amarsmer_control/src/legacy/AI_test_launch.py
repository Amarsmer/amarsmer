from simple_launch import SimpleLauncher


def generate_launch_description():

    sl = SimpleLauncher(use_sim_time = True)

    sl.include('amarsmer_description', 'world_launch.py', launch_arguments={'sliders': False})

    sl.node('amarsmer_control', 'path_generation.py', launch_arguments={'trajectory': 'sin'})

    sl.node('amarsmer_control', 'path_publisher.py')

    sl.node('amarsmer_control', 'AI_test_run.py')

    return sl.launch_description()
