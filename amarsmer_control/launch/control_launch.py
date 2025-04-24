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

        # sl.node('slider_publisher', 'slider_publisher', name='joint_control',
        #         arguments=[sl.find('amarsmer_control', 'thrusters_plasmar2_joints.yaml')])


        sl.node('thruster_manager', 'publish_wrenches',
                parameters={'control_frame': 'amarsmer/base_link',
                        'use_gz_topics': sl.sim_time})

    return sl.launch_description()
