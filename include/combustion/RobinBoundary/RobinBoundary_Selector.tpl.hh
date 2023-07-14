/**
 * @file RobinBoundary_Selector.tpl.hh
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhäuser (MPB)
 */

/*  Copyright (C) 2012-2023 by Jan Philipp Thiele, Uwe Koecher                */
/*                          and Marius Paul Bruchhaeuser                      */
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


#ifndef __RobinBoundary_Selector_tpl_hh
#define __RobinBoundary_Selector_tpl_hh

// DEAL.II includes
#include <deal.II/base/function.h>

// C++ includes
#include <memory>
#include <string>

namespace combustion {
namespace robin_boundary {

template<int dim>
class Selector {
public:
    Selector() = default;
    virtual ~Selector() = default;

    virtual void create_function(
        const std::string &type,
        const std::string &options,
        std::shared_ptr< dealii::Function<dim> > &function
    ) const;
};

}}

#endif
