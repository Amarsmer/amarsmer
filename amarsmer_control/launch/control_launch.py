from simple_launch import SimpleLauncher


def generate_launch_description():

    sl = SimpleLauncher(use_sim_time = True)

    with sl.group(ns = 'amarsmer'):

        # load body controller anyway
        sl.node('auv_control', 'cascaded_pid',
                parameters=[sl.find('amarsmer_control', 'cascaded_pid.yaml')],
                output='screen')

        sl.node('slider_publisher', 'slider_publisher', name='pose_control',
                arguments=[sl.find('auv_control', 'pose_setpoint.yaml')])

    return sl.launch_description()
