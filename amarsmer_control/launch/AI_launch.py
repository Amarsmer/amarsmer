from simple_launch import SimpleLauncher
from pathlib import Path

def generate_launch_description():

    sl = SimpleLauncher(use_sim_time = True)

    sl_trajectory = sl.declare_arg('trajectory', default_value = 'station_keeping') # Load specific trajectory from path_generation
    sl_network_name = sl.declare_arg('network_name', default_value = '')            # Load saved network (name only, no ".json")
    sl_train = sl.declare_arg('train', default_value = True)                        # Whether to train or test the network
    sl_automate = sl.declare_arg('automate', default_value = False)                 # Will teleport the robot if the criteria is small enough for a set duration, mostly used with the station_keeping task

    sl.include('amarsmer_description', 'world_launch.py', launch_arguments={'sliders': False})

    sl.node('amarsmer_control', 'path_generation.py', parameters={'trajectory' : sl_trajectory})

    sl.node('amarsmer_control', 'path_publisher.py') # Used to display the path in rviz

    sl.node('amarsmer_control', 'AI_run.py', parameters={'network_name' : sl_network_name,
                                                         'train': sl_train,
                                                         'automate': sl_automate})
    
    return sl.launch_description()
