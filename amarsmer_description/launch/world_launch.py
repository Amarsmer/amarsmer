from simple_launch import SimpleLauncher, GazeboBridge


def generate_launch_description():
    
    sl = SimpleLauncher()
    sl.declare_arg('gui', default_value=True)
    sl.declare_arg('spawn', default_value=True)

    with sl.group(if_arg='gui'):
        sl.gz_launch(sl.find('amarsmer_description', 'world.sdf'), "-r")
        
    with sl.group(unless_arg='gui'):
        sl.gz_launch(sl.find('amarsmer_description', 'world.sdf'), "-r -s")

    bridges = [GazeboBridge.clock(),
               GazeboBridge('/ocean_current', '/current', 'geometry_msgs/Vector3',
                            GazeboBridge.ros2gz)]
        
    sl.create_gz_bridge(bridges)

    with sl.group(if_arg='spawn'):
        sl.include('amarsmer_description', 'upload_rov_launch.py')
        
    return sl.launch_description()
