from simple_launch import SimpleLauncher, GazeboBridge

sl = SimpleLauncher(use_sim_time = True)

sl.declare_arg('namespace', default_value='amarsmer')
sl.declare_arg('ground_truth',default_value=True)
sl.declare_arg('sliders',default_value=True)
sl.declare_arg('camera', True)
sl.declare_arg('gazebo_world_name', 'none')

sl.declare_arg('thr','thrusters_plasmar2')
sl.declare_arg('spawn_pose', default_value = "5.0 0.0 0.0 0.0 0.0 0.5")

sl.declare_gazebo_axes(x=0.4, y=0.4, z=0., roll=0.,pitch=0., yaw=0.5)
# sl.declare_gazebo_axes(x=0., y=0., z=0., roll=0.,pitch=0., yaw=0.5)
def launch_setup():
    
    ns = sl.arg('namespace')
    thr = sl.arg('thr')

    if sl.arg('gazebo_world_name') != 'none':
        GazeboBridge.set_world_name(sl.arg('gazebo_world_name'))
    
    # robot state publisher
    sl.include('amarsmer_description', 'state_publisher_launch.py',
               launch_arguments={'namespace': ns, 'use_sim_time': sl.sim_time, 'jsp': False,
                                 'thr': thr})
               
    with sl.group(ns=ns):
                    
        # URDF spawner to Gazebo, defaults to relative robot_description topic
        sl.spawn_gz_model(ns, spawn_args = sl.gazebo_axes_args())
            
        # ROS-Gz bridges
        bridges = []
        gz_js_topic = GazeboBridge.model_prefix(ns) + '/joint_state'
        bridges.append(GazeboBridge(gz_js_topic, 'joint_states', 'sensor_msgs/JointState', GazeboBridge.gz2ros))
        
        # pose ground truth
        bridges.append(GazeboBridge(f'/model/{ns}/pose',
                                     'pose_gt', 'geometry_msgs/Pose', GazeboBridge.gz2ros))
        
        # ground truth if requested
        if sl.arg('ground_truth'):
            bridges.append(GazeboBridge(f'/model/{ns}/odometry',
                                     'odom', 'nav_msgs/Odometry', GazeboBridge.gz2ros,
                                     'gz.msgs.Odometry'))
            sl.node('pose_to_tf',parameters={'child_frame': ns + '/base_link'})
        else:
            # otherwise publish ground truth as another link to get, well, ground truth
            sl.node('pose_to_tf',parameters={'child_frame': ns+'/base_link_gt'})
        
        # thrusters and steering

        if thr == 'thrusters_plasmar2':

            for t in range(1, 5):
                thruster = f'thruster{t}'
                gz_thr_topic = f'/{ns}/{thruster}/cmd'
                bridges.append(GazeboBridge(gz_thr_topic, f'cmd_{thruster}', 'std_msgs/Float64', GazeboBridge.ros2gz))

                steering = f'/model/{ns}/joint/thruster{t}_steering/0/cmd_pos'
                bridges.append(GazeboBridge(steering, f'cmd_{thruster}_steering', 'std_msgs/Float64', GazeboBridge.ros2gz))

        elif thr == 'thrusters_plasmar1':
            pass

        sl.create_gz_bridge(bridges)

        if sl.arg('sliders'):
            # TODO create .yaml file for each thruster config
            sl.node('slider_publisher', arguments=[sl.find('amarsmer_description', thr +'.yaml')])
    
    return sl.launch_description()


generate_launch_description = sl.launch_description(launch_setup)
