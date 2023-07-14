/**
 * @file LogStream.hh
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 */

/*  Copyright (C) 2012-2023 by Jan Philipp Thiele and Uwe Koecher             */
/*                                                                            */
/*  pu-dwr-combustion is free software: you can redistribute it and/or modify */
/*  it under the terms of the GNU Lesser General Public License as            */
/*  published by the Free Software Foundation, either                         */
/*  version 3 of the License, or (at your option) any later version.          */
/*                                                                            */
/*  pu-dwr-combustion is distributed in the hope that it will be useful,      */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/*  GNU Lesser General Public License for more details.                       */
/*                                                                            */
/*  You should have received a copy of the GNU Lesser General Public License  */
/*  along with pu-dwr-combustion. If not, see <http://www.gnu.org/licenses/>. */

#ifndef __LogStream_hh
#define __LogStream_hh

// MPI includes

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

// C++ includes
#include <iostream>
#include <fstream>
#include <memory>

namespace DTM {

class LogStream : public std::ofstream {
public:
    LogStream() = default;
    ~LogStream() = default;

    virtual void open();
	
};

extern DTM::LogStream pout;

} // namespace

#endif
