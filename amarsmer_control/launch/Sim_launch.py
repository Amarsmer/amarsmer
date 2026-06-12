from simple_launch import SimpleLauncher

sl = SimpleLauncher(use_sim_time = True)

# Simulation parameters
sl.declare_arg('thrusters', default_value='ur')
sl_spawn = sl.declare_arg('spawn_pose', default_value='0.0 0.0 0.0 0.0 0.0 0.0')

# Control parameters
sl_trajectory = sl.declare_arg('trajectory', default_value = 'station_keeping')  # Compute selected trajectory
sl_controller = sl.declare_arg('controller_type', default_value = 'MPC')         # Uses selected controller
sl_comment = sl.declare_arg('comment', default_value='')                         # Add specified comment to the recorded data filename
sl_dt = sl.declare_arg('dt', default_value = 0.05)                               # Make sure different systems use the same time period

# AI specific parameters
sl_network_name = sl.declare_arg('network_name', default_value = '')             # Load saved network (name only, no ".json")
sl_train = sl.declare_arg('train', default_value = True)                         # Whether to train or test the network
sl_automate = sl.declare_arg('automate', default_value = False)                  # Will teleport the robot if the criteria is small enough for a set duration, mostly used with the station_keeping task

architecture_param = {'ur': {
                                'xacro': 'thrusters_plasmar_ur',
                                'thrusters': 2
                        },
                      'uvr': {
                                'xacro': 'thrusters_plasmar_uvr',
                                'thrusters': 3
                        },
                      'plasmar2': {
                                'xacro': 'thrusters_plasmar2',
                                'thrusters': 4
                        }
                }
                        

def launch_setup():
        archi = architecture_param.get(sl.arg('thrusters'), {})
        thr_file = archi.get('xacro')
        thr_nb = archi.get('thrusters')

        controllers = {'PID': {
                                'node': 'PID.py',
                                'params': {'nb_thrusters' : thr_nb,
                                           'dt' : sl_dt}
                        },
                       'MPC': {
                                'node': 'planar_mpc_control.py',
                                'params': {'nb_thrusters' : thr_nb,
                                           'dt' : sl_dt}
                        },
                        'AI': {
                                'node': 'AI_run.py',
                                'params': {'network_name' : sl_network_name,
                                           'train': sl_train,
                                           'automate': sl_automate,
                                           'dt' : sl_dt}
                        }
                }
                    
        sl.include('amarsmer_description', 
                   'world_launch.py',  
                   launch_arguments={'sliders': False,  
                                     'thr': thr_file,
                                     'nb_thr' : thr_nb})

        sl.node('amarsmer_control', 
                'path_generation.py', 
                parameters={'trajectory' : sl_trajectory})

        sl.node('amarsmer_control',
                'path_publisher.py')

        sl.node('amarsmer_control', 
                'simulation_interface.py',
                parameters={'nb_thrusters' : thr_nb,
                            'dt' : sl_dt,
                            'spawn_pose': sl_spawn})

        sl.node('amarsmer_control',
                'control_manager.py',
                parameters={'controller_type' : sl_controller,
                            'comment' : sl_comment,
                            'simulation' : True,
                            'use_sim_time': True})
    
        cfg = controllers.get(sl.arg('controller_type'), {})
        sl.node('amarsmer_control',
                cfg.get('node'),
                parameters=cfg.get('params', {})
        )

        return sl.launch_description()

generate_launch_description = sl.launch_description(opaque_function = launch_setup)