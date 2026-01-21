from simple_launch import SimpleLauncher
from pathlib import Path


def generate_launch_description():

    sl = SimpleLauncher(use_sim_time = True)

    sl_trajectory = sl.declare_arg('trajectory', default_value = 'station_keeping')
    # sl_load_weights = sl.declare_arg('load_weights', default_value = False)
    sl_weight_name = sl.declare_arg('weights_name', default_value = '')
    sl_train = sl.declare_arg('train', default_value = True)


    sl.include('amarsmer_description', 'world_launch.py', launch_arguments={'sliders': False})

    sl.node('amarsmer_control', 'path_generation.py', parameters={'trajectory' : sl_trajectory})

    sl.node('amarsmer_control', 'path_publisher.py')

    sl.node('amarsmer_control', 'AI_run.py', parameters={'weight_name' : sl_weight_name,
                                                         'train': sl_train})

    """
    layout_file = str(Path(__file__).parents[2] / "plotJuggler_2D_BP_monitoring.xml")

    sl.node(
        package="plotjuggler",
        executable="plotjuggler",
        name="plotjuggler_with_layout",
        arguments=["--layout", layout_file]
    )
    """
    
    return sl.launch_description()
