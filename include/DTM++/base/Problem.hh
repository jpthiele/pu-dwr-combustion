/**
 * @file Problem.hh
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 *
 * @brief Abstract problem class
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

#ifndef __DTM_Problem_hh
#define __DTM_Problem_hh

// MPI includes

// DEAL.II includes
#include <deal.II/base/parameter_handler.h>

// C++ includes
#include <memory>

namespace DTM {

class Problem {
public:
    Problem() = default;

    virtual ~Problem() = default;

    virtual void set_input_parameters(
        std::shared_ptr< dealii::ParameterHandler > parameter_handler
    );

    virtual void run();
};


} // namespace
#endif
