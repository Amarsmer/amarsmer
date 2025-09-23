from simple_launch import SimpleLauncher
from pathlib import Path


def generate_launch_description():

    sl = SimpleLauncher(use_sim_time = True)

    sl.include('amarsmer_description', 'world_launch.py', launch_arguments={'sliders': False})

    # sl.node('amarsmer_control', 'path_generation.py')

    # sl.node('amarsmer_control', 'path_publisher.py')

    sl.node('amarsmer_control', 'AI_train_run.py')

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
