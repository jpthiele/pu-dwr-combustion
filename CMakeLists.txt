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


MESSAGE("================================================================================")
MESSAGE("=== Configuring PU-DWR-Combustion ==============================================")
MESSAGE("================================================================================")

# SETs the NAME of the TARGET and PROJECT
SET(TARGET "pu-dwr-combustion")

FILE(GLOB_RECURSE TARGET_SRC "source/*.cc")
INCLUDE_DIRECTORIES(include)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

# Need at least deal.II version 9.3 as support to copy 
# the refinement information from a p4est object to a 
# new Triangulation with the same coarse mesh
# was added there

FIND_PACKAGE(deal.II 9.3 QUIET HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
IF(NOT ${deal.II_FOUND})
	MESSAGE(FATAL_ERROR "\n*** ERROR: Could not locate a suitably recent version of deal.II. ***\n"
		"SOLUTION A: EITHER you pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake.\n"
		"SOLUTION B: OR, set an environment variable \"DEAL_II_DIR\".\n"
	)
ENDIF()

MESSAGE(STATUS "Found deal.II version ${DEAL_II_PACKAGE_VERSION} at '${deal.II_DIR}'")

SET(_DEALII_GOOD ON)

# IF(NOT DEAL_II_WITH_P4EST)
# 	MESSAGE(SEND_ERROR "\n-- deal.II was build without support for p4est!\n")
# 	SET(_DEALII_GOOD OFF)
# ENDIF()

# IF(NOT DEAL_II_WITH_TRILINOS)
# 	MESSAGE(SEND_ERROR "\n-- deal.II was build without support for Trilinos!\n")
# 	SET(_DEALII_GOOD OFF)
# ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})

MESSAGE("================================================================================")

INCLUDE(cmake/InvokeAutopilot.cmake)
INCLUDE(cmake/AddAdditionalCompilerWarnings.cmake)
INVOKE_AUTOPILOT()

MESSAGE(STATUS "NOTE: Writing configuration into ./detailed.log")
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

INCLUDE(cmake/CreateDetailedLog.cmake)
