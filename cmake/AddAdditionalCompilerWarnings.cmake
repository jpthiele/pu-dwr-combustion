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


# CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

################################################################################
# Add additional compiler options:
# 
ADD_DEFINITIONS(-std=c++17)

ADD_DEFINITIONS(-Wall)
ADD_DEFINITIONS(-Wextra)
ADD_DEFINITIONS(-Wpedantic)

# Unset specific warnings (that are warnings coming from the deal.II lib)
#ADD_DEFINITIONS(-Wno-unneeded-internal-declaration)
#ADD_DEFINITIONS(-Wno-unused-parameter)
#ADD_DEFINITIONS(-Wno-unused-variable)

# Make WARNINGS to ERRORS
#ADD_DEFINITIONS(-Werror)        # Compilation breaks, if warnings are present
ADD_DEFINITIONS(-Wfatal-errors) # Compilation breaks directly after first error
################################################################################

