# Copyright (C) 2012-2023 by Jan Philipp Thiele and Uwe Koecher
# 
# This file is part of pu-dwr-combustion.
# 
# pu-dwr-combustion is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# 
# pu-dwr-combustion is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with pu-dwr-combustion. If not, see <http://www.gnu.org/licenses/>.

## ---------------------------------------------------------------------
##
## Copyright (C) 2012 - 2014 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE at
## the top level of the deal.II distribution.
##
## ---------------------------------------------------------------------

# Usage:
#       INVOKE_AUTOPILOT()
#
# Options:
#       TARGET         - a string used for the project and target name
#       TARGET_SRC     - a list of source file to compile for target
#                         ${TARGET}
#       TARGET_RUN     - (optional) the command line that should be
#                         invoked by "make run", will be set to default
#                         values if undefined. If no run target should be
#                         created, set it to an empty string.
#       CLEAN_UP_FILES - (optional) a list of files (globs) that will be
#                         removed with "make runclean" and "make
#                         distclean", will be set to default values if
#                         empty
#


macro(INVOKE_AUTOPILOT)

# CMake GENERATOR specific values:
if(CMAKE_GENERATOR MATCHES "Ninja")
	SET(_make_command "$ ninja")
	
else()
	SET(_make_command " $ make")
	
endif()

# DEFINE a make target:
add_executable(${TARGET} ${TARGET_SRC})

# SETUP a make target
DEAL_II_SETUP_TARGET(${TARGET})

message(STATUS "Autopilot invoked")

# DEFINE TARGET_RUN: a custom make target to run the application.
if(NOT DEFINED TARGET_RUN)
	SET(TARGET_RUN ${TARGET} ./input/default.prm)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "(CYGWIN|Windows)")
	#
	# Hack for Cygwin and Windows targets: Export PATH to point to the
	# dynamic library.
	#
	set(_delim ":")
	
	if(CMAKE_SYSTEM_NAME MATCHES "Windows")
		SET(_delim ";")
	endif()
	
	file(
		WRITE ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/run_target.cmake
		"SET(ENV{PATH} \"${CMAKE_CURRENT_BINARY_DIR}${_delim}${DEAL_II_PATH}/${DEAL_II_EXECUTABLE_RELDIR}${_delim}\$ENV{PATH}\")\n"
		"EXECUTE_PROCESS(COMMAND ${TARGET_RUN}\n"
		"  RESULT_VARIABLE _return_value\n"
		"  )\n"
		"IF(NOT \"\${_return_value}\" STREQUAL \"0\")\n"
		"  MESSAGE(SEND_ERROR \"\nProgram terminated with exit code: \${_return_value}\")\n"
		"ENDIF()\n"
	)
	
	set(
		_command
		${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/run_target.cmake
	)
	
else()
	set(_command ${TARGET_RUN})
	
endif()

if(NOT "${TARGET_RUN}" STREQUAL "")
	add_custom_target(
		run
		COMMAND ${_command}
		DEPENDS ${TARGET}
		COMMENT "Run ${TARGET} with ${CMAKE_BUILD_TYPE} configuration"
	)
	
	set(
		_run_targets
		"#      ${_make_command} run            - to (compile, link and) run the program\n"
	)
endif()

################################################################################
# PROVIDE a target to SIGN the generated executable on Mac OSX with a
# developer key. This avoids problems with an enabled firewall and MPI
# tasks that need networking.
#
IF(CMAKE_SYSTEM_NAME MATCHES "Darwin")
	IF(DEFINED OSX_CERTIFICATE_NAME)
		ADD_CUSTOM_COMMAND(
			OUTPUT ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${TARGET}.signed
			COMMAND codesign -f -s ${OSX_CERTIFICATE_NAME} ${TARGET}
			COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${TARGET}.signed
			COMMENT "Digitally signing ${TARGET}"
			DEPENDS ${TARGET}
			VERBATIM
		)
		
		ADD_CUSTOM_TARGET(
			sign ALL
			DEPENDS ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${TARGET}.signed
		)
		
		ADD_DEPENDENCIES(run sign)
		
	ELSE()
		ADD_CUSTOM_TARGET(
			sign
			COMMAND
			${CMAKE_COMMAND} -E echo ''
			&& ${CMAKE_COMMAND} -E echo '***************************************************************************'
			&& ${CMAKE_COMMAND} -E echo '**  Error: No Mac OSX developer certificate specified'
			&& ${CMAKE_COMMAND} -E echo '**  Please reconfigure with -DOSX_CERTIFICATE_NAME="<...>"'
			&& ${CMAKE_COMMAND} -E echo '***************************************************************************'
			&& ${CMAKE_COMMAND} -E echo ''
			COMMENT "Digitally signing ${TARGET}"
		)
		
	ENDIF()

	SET(
		_run_targets
		"${_run_targets}#\n#      ${_make_command} sign           - to sign the executable with the supplied OSX developer key\n"
	)
ENDIF()
################################################################################

################################################################################
# DEFINE custom targets to easily create and clean the documentation:

# DOC: doc target to generate the documentation via doxygen
add_custom_target(
	doc
	COMMAND doxygen ./doc/Doxyfile
	COMMENT "documentation creation invoked"
)

# DOCCLEAN: docclean target to clean the documentation from doxygen
add_custom_target(
	docclean
	COMMAND rm -rf ./doc/doxygen
	COMMENT "documentation clean invoked"
)

#
################################################################################

################################################################################
# DEFINE custom targets to easily switch the build type:
add_custom_target(
	debug
	COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
	COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
	COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
)

add_custom_target(
	release
	COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
	COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
	COMMENT "Switch CMAKE_BUILD_TYPE to Release"
)

# Only mention release and debug targets if it is actuallay possible to
# switch between them:
if(${DEAL_II_BUILD_TYPE} MATCHES "DebugRelease")
	set(
		_switch_targets
"#      ${_make_command} debug          - to switch the build type to 'Debug'
#      ${_make_command} release        - to switch the build type to 'Release'\n"
	)
endif()
################################################################################

################################################################################

# CLEAN_UP_FILES: custom target to clean up all files generated by the application:
IF("${CLEAN_UP_FILES}" STREQUAL "")
	SET(CLEAN_UP_FILES *.log *.tex *.h5 *.xdmf *.gmv *.gnuplot *.gpl *.eps *.pov *.vtk *.ucd *.d2)
ENDIF()

# RUNCLEAN: custom target
ADD_CUSTOM_TARGET(
	runclean
	COMMAND ${CMAKE_COMMAND} -E remove ${CLEAN_UP_FILES}
	COMMENT "runclean invoked"
)

# DISTCLEAN: distclean target to remove every generated file:
ADD_CUSTOM_TARGET(
	distclean
	COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
	COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target runclean
	COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target docclean
	COMMAND ${CMAKE_COMMAND} -E remove_directory CMakeFiles
	COMMAND ${CMAKE_COMMAND} -E remove CMakeCache.txt cmake_install.cmake Makefile
	COMMENT "distclean invoked"
)

################################################################################

################################################################################
# PRINT out some usage information to file:
FILE(
	WRITE ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/print_usage.cmake
"MESSAGE(
\"###
#
#  Project  ${TARGET}  set up with  ${DEAL_II_PACKAGE_NAME}-${DEAL_II_PACKAGE_VERSION}  found at
#       ${DEAL_II_PATH}
#
#  CMAKE_BUILD_TYPE:          ${CMAKE_BUILD_TYPE}
#
#  You can now run
#      ${_make_command}                - to compile and link the program
${_run_targets}#
${_switch_targets}#
")

IF(NOT CMAKE_GENERATOR MATCHES "Ninja")
	FILE(
		APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/print_usage.cmake
"#      ${_make_command} edit_cache     - to change (cached) configuration variables
#                               and reruns CMake
#
"
	)
ENDIF()

FILE(
	APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/print_usage.cmake
"#      ${_make_command} clean          - to remove the generated executable as well as
#                               all intermediate compilation files
#      ${_make_command} runclean       - to remove all output generated by the program
#      ${_make_command} distclean      - to clean the directory from _all_ generated
#                               files (includes clean, runclean and the removal
#                               of the generated build system)
#
#      ${_make_command} kdev_include_paths - to set .kdev_include_paths files
#                                   appropriately for using KDevelop without a
#                                   setting up a project file
#
#
#      ${_make_command} doc            - to build the documentation via doxygen
#
#      ${_make_command} info           - to view this message again
#
###\")"
)

################################################################################

add_custom_target(
	info
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/print_usage.cmake
)

# PRINT this message ONCE:
if(NOT USAGE_PRINTED)
	include(${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/print_usage.cmake)
	
	set(USAGE_PRINTED TRUE CACHE INTERNAL "")

else()
	message(STATUS "Run  ${_make_command} info  to print a detailed help message")
	
endif()

endmacro()

