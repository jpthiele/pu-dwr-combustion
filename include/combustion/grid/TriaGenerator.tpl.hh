/**
 * @file TriaGenerator.tpl.hh
 *
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

#ifndef __TriaGenerator_tpl_hh
#define __TriaGenerator_tpl_hh

// Project includes
#include <deal.II/grid/tria.h>

// C++ includes
#include <string>
#include <vector>

namespace combustion {

template<int dim>
class TriaGenerator {
public:
    TriaGenerator() = default;
    virtual ~TriaGenerator() = default;

    virtual void generate(
        const std::string &TriaGenerator_Type,
        const std::string &TriaGenerator_Options,
        std::shared_ptr< dealii::Triangulation<dim> > tria
    );
};

}

#endif
