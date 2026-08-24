from simple_launch import SimpleLauncher

sl = SimpleLauncher(use_sim_time = False)

# Simulation parameters
sl.declare_arg('thrusters', default_value='plasmar2')
sl_spawn = sl.declare_arg('spawn_pose', default_value='0.0 0.0 0.0 0.0 0.0 0.0')
sl_IP = sl.declare_arg('IP', default_value="192.168.1.255")
sl_Port = sl.declare_arg('Port', default_value=61022)

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
        thr_file = 'thrusters_plasmar2'
        thr_nb = 4
                    
        sl.include('amarsmer_description', 
                   'world_launch.py',  
                   launch_arguments={'sliders': True,  
                                     'thr': thr_file,
                                     'nb_thr' : thr_nb})

        sl.node('amarsmer_control', 
                'input_computation.py', 
                parameters={'nb_thr' : thr_nb})

        sl.node('amarsmer_control', 
                'input_computation.py', 
                parameters={'nb_thr' : thr_nb})
        
        sl.node('amarsmer_control',
                'comm.py',
                parameters={'IP': sl_IP,
                            'Port': sl_Port})

        return sl.launch_description()

generate_launch_description = sl.launch_description(opaque_function = launch_setup)