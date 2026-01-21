from simple_launch import SimpleLauncher
from pathlib import Path


def generate_launch_description():

    sl = SimpleLauncher(use_sim_time = True)

    trajectory = sl.declare_arg('trajectory', default_value = 'station_keeping')
    load_weights = sl.declare_arg('load_weights', default_value = False)
    train = sl.declare_arg('train', default_value = True)


    sl.include('amarsmer_description', 'world_launch.py', launch_arguments={'sliders': False})

    sl.node('amarsmer_control', 'path_generation.py', parameters={'trajectory' : trajectory})

    sl.node('amarsmer_control', 'path_publisher.py')

    sl.node('amarsmer_control', 'AI_run.py', parameters={'load_weights' : load_weights,
                                                         'train': train})

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
