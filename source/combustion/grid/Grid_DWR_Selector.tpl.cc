/**
 * @file Grid_DWR_Selector.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 */

/*  Copyright (C) 2012-2023 by Jan Philipp Thiele and Uwe Koecher             */
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

#include <DTM++/base/LogStream.hh>

#include <combustion/grid/Grid_DWR_Selector.tpl.hh>
#include <combustion/grid/Grids.hh>

// MPI includes

// DEAL.II includes

// C++ includes

namespace combustion {
namespace grid {

template<int dim>
void
Selector<dim>::
create_grid(
    const std::string &Grid_Class,
    const std::string &Grid_Class_Options,
    const std::string &TriaGenerator,
    const std::string &TriaGenerator_Options,
    std::shared_ptr< combustion::Grid_DWR<dim,1> > &grid,
    const MPI_Comm &mpi_comm
) const {
    ////////////////////////////////////////////////////////////////////////////
    //
    DTM::pout << "grid selector: creating Grid Class = " << Grid_Class << std::endl;
	
    ////////////////////////////////////////////////////////////////////////////
    //

    if ( !Grid_Class.compare("Grid_DWR_Dirichlet_Neumann_and_Robin") )
    {
        grid = std::make_shared< combustion::grid::Grid_DWR_Dirichlet_Neumann_and_Robin<dim,1> > (
            Grid_Class_Options,
            TriaGenerator,
            TriaGenerator_Options,
            mpi_comm
        );
        return;
    }
    ////////////////////////////////////////////////////////////////////////////
    //
    AssertThrow(
        false,
        dealii::ExcMessage("Grid Class unknown, please check your input file data.")
    );
}

}} //namespaces

#include "Grid_DWR_Selector.inst.in"
