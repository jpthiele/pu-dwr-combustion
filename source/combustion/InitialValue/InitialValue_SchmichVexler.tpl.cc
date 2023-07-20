/**
 * @file InitialValue_SchmichVexler.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 */

/*  Copyright (C) 2012-2023 by Jan Philipp Thiele, Uwe Koecher                */
/*                          and Marius Paul Bruchhaeuser                      */
/*                                                                            */
/*  This file is part of pu-dwr-combustion                                    */
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

#include <combustion/InitialValue/InitialValue_SchmichVexler.tpl.hh>

namespace combustion {
namespace initial_value {

template<int dim>
double
SchmichVexler<dim>::
value(
    const dealii::Point<dim> &x,
    const unsigned int component
) const {
  
    Assert(dim==2, dealii::ExcNotImplemented());
    Assert(this->get_time() >= 0., dealii::ExcNotImplemented());

    if ( component == 0 ) //theta
    {
      if ( x[0] <= x_tilde )
        return 1.0;
      else
        return std::exp(x_tilde-x[0]);
    }
    else //Y
    {
        if ( x[0] <= x_tilde )
          return 0.0;
        else
          return 1.0-std::exp(Le*(x_tilde-x[0]));
    }
}

}} //namespaces

#include "InitialValue_SchmichVexler.inst.in"
