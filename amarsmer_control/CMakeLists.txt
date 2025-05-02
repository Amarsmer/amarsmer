cmake_minimum_required(VERSION 3.5)
project(amarsmer_control)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(PROGRAMS src/control.py
        DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS src/mpc_control.py
        DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS src/pose_to_rpy.py
        DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS src/hydrodynamic_model.py
        DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS src/path_publisher.py
        DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS src/path_generation.py
        DESTINATION lib/${PROJECT_NAME})

ament_python_install_package(${PROJECT_NAME})

ament_package()
