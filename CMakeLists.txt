cmake_minimum_required(VERSION 3.0.2)
project(craic_task_template)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/scene1_patrol.py
  scripts/scene2_sorting.py
  scripts/basic_motion.py
  scripts/obstacle_avoidance.py
  scripts/button_presser.py
  scripts/arm_controller.py
  scripts/controller_manager.py
  scripts/localization.py
  scripts/state_machine.py
  scripts/terrain_traverser.py
  scripts/waypoints.py
  scripts/utils/math_utils.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY scripts/utils/
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}/utils
  FILES_MATCHING PATTERN "*.py")
