# AMARSMER ROS2

This repository contains the robot description and necessary launch files to describe and simulate the Plasmar2 (unmanned underwater vehicle) with Gazebo and its hydrodynamics plugins under ROS2.

Additionnal steps are included to make sure this can be used starting from a fresh Ubuntu install.

NOTE: This package is a modified version of the original [BlueROV2](https://github.com/CentraleNantesROV/bluerov2/tree/main)

# Requirements

## ROS2
The current recommended ROS2 version is Jazzy. All the related info can be found [here](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)

## Gazebo
- The recommended gazebo version is GZ Harmonic (LTS). More info and step-by-step installation guide are found [here](https://gazebosim.org/docs/latest/ros_installation/)
- [pose_to_tf](https://github.com/oKermorgant/pose_to_tf), to get the ground truth from Gazebo if needed.

## For the description

- [Xacro](https://github.com/ros/xacro/tree/ros2) , installable through `apt install ros-${ROS_DISTRO}-xacro`
- [simple_launch](https://github.com/oKermorgant/simple_launch), installable through `apt install ros-${ROS_DISTRO}-simple-launch`

## For the control part

- [slider_publisher](https://github.com/oKermorgant/slider_publisher), installable through `apt install ros-${ROS_DISTRO}-slider-publisher`
- [auv_control](https://github.com/CentraleNantesROV/auv_control) for basic control laws
- [urdf_parser](https://github.com/ros/urdf_parser_py) intended to have the controller work with any robot description

# Installation

- Clone the package and its dependencies (if from source) in your ROS 2 workspace `src` and compile with `colcon build`, make sure you are in the parent folder of `src` when compiling.

# Running 
- Make sure to source the terminal if you did not modify the bashrc file.

    `source /opt/ros/jazzy/setup.bash`

    `source install/setup.bash`

- The rviz, which is useful to quickly test if the architecture behaves correctly, can be displayed by using: 

    `ros2 launch amarsmer_description state_publisher_launch.py`

- To run a demonstration with the vehicle, you can run a Gazebo scenario, and spawn the robot with a GUI to control the thrusters:

    `ros2 launch amarsmer_description world_launch.py sliders:=true`

# Input / output

Gazebo will:

- Subscribe to /amarsmer/cmd_thruster[i] and /amarsmer/cmd_thruster[i]_steering, and expect std_msgs/Float64 messages (for both), respectively being the thrust in Newton and the angle in radians.
- NOT YET IMPLEMENTED: ~~Publish sensor data to various topics (image, mpu+lsm for IMU, cloud for the sonar, odom)~~
- Publish the ground truth on /amarsmer/pose_gt. This pose is forwarded to /tf if pose_to_tf is used.

# High-level control

Basic control is available in the [auv_control package](https://github.com/CentraleNantesROV/auv_control)

The simple node running an open-loop control (use the world launch first) can be ran as:

`ros2 run amarsmer_control control.py`

The full launch file for MPC control (including world launch) is ran with:

`ros2 launch amarsmer_control MPC_launch.py`


# License
Amarsmer package is open-sourced under the MIT License. See the LICENSE file for details.
