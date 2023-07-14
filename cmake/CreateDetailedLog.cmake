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


SET(_log_detailed "${CMAKE_BINARY_DIR}/detailed.log")
FILE(REMOVE ${_log_detailed})

MACRO(_detailed)
	FILE(APPEND ${_log_detailed} "${ARGN}")
ENDMACRO()

_detailed(
"###
# 
# MEAT Configuration:
#   PATH:                   $ENV{PATH}
#   
#   DEAL_II_DIR:            ${deal.II_DIR}
#   DEAL_II_VERSION:        ${DEAL_II_PACKAGE_VERSION}
#   
#   CMAKE_BUILD_TYPE:       ${CMAKE_BUILD_TYPE}
#   CMAKE_INSTALL_PREFIX:   ${CMAKE_INSTALL_PREFIX}
#   CMAKE_SOURCE_DIR:       ${CMAKE_SOURCE_DIR} 
#   CMAKE_BINARY_DIR:       ${CMAKE_BINARY_DIR}
#   CMAKE_CXX_COMPILER:     ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} on platform ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}
#                           ${CMAKE_CXX_COMPILER}
")

IF(CMAKE_C_COMPILER_WORKS)
  _detailed("#   CMAKE_C_COMPILER:       ${CMAKE_C_COMPILER}\n")
ENDIF()


IF(DEAL_II_STATIC_EXECUTABLE)
_detailed(
"#
#   LINKAGE:                STATIC
")
ELSE()
_detailed(
"#
#   LINKAGE:                DYNAMIC
")
ENDIF()

_detailed("#\n###\n")

