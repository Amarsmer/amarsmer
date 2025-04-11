# AMARSMER ROS2

This repository contains the robot description and necessary launch files to describe and simulate the Plasmar2 (unmanned underwater vehicle) with Gazebo and its hydrodynamics plugins under ROS2.

Additionnal steps are included to make sure this can be used starting from a fresh Ubuntu install.

NOTE: Some of this document is takenbuild
 from the original [BlueROV2](https://github.com/CentraleNantesROV/bluerov2/tree/main) readme and does not currently apply

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
- [auv_control](https://github.com/CentraleNantesROV/auv_control) for basic control laws, from source

# Installation

- Clone the package and its dependencies (if from source) in your ROS 2 workspace `src` and compile with `colcon build`, make sure you are in the parent folder of `src` when compiling.

# Running 
- Make sure to source the terminal if you did not modify the bashrc file.

    `source /opt/ros/jazzy/setup.bash`

    `source install/setup.bash`

- The rviz, which is useful to quickly test if the architecture behaves correctly, can be displayed by using: 

    `ros2 launch amarsmer_description state_publisher_launch.py`

- To run a demonstration with the vehicle, you can run a Gazebo scenario, sand spawn the robot with a GUI to control the thrusters:

    `ros2 launch amarsmer_description world_launch.py sliders:=true`

# Input / output

Gazebo will:

Subscribe to /bluerov2/cmd_thruster[1..6] and expect std_msgs/Float64 messages, being the thrust in Newton
Publish sensor data to various topics (image, mpu+lsm for IMU, cloud for the sonar, odom)
Publish the ground truth on /bluerov2/pose_gt. This pose is forwarded to /tf if pose_to_tf is used.

# High-level control (TODO)

Basic control is available in the [auv_control package](https://github.com/CentraleNantesROV/auv_control)

First run world_launch, then:

`ros2 launch amarsmer_control control_launch.py`

~~`ros2 launch bluerov2_control cascaded_pids_launch.py sliders:=true`~~

# License
Amarsmer package is open-sourced under the Apache-2.0 license. See the LICENSE file for details.
