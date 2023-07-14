/**
 * @file Grid_DWR_Dirichlet_Neumann_and_Robin.tpl.hh
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * 
 * @date 2018-07-26, UK
 * @date 2018-03-06, UK
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

#ifndef __Grid_DWR_Dirichlet_Neumann_and_Robin_tpl_hh
#define __Grid_DWR_Dirichlet_Neumann_and_Robin_tpl_hh

// PROJECT includes
#include <combustion/grid/Grid_DWR.tpl.hh>

// DEAL.II includes

// C++ includes

namespace combustion {
namespace grid {

/**
 * Colorises the boundary \f$ \Gamma_N \cup \Gamma_D = \partial \Omega \f$
 * for the application of a Neumann type boundary for faces having
 * \f$ x_1 = 0 \f$ and a Dirichlet type boundary elsewhere,
 * independently of the geometry of \f$ \Omega \f$.
 */
template<int dim, int spacedim>
class Grid_DWR_Dirichlet_Neumann_and_Robin : public combustion::Grid_DWR<dim,spacedim> {
public:
    Grid_DWR_Dirichlet_Neumann_and_Robin(
        const std::string &Grid_Class_Options,
        const std::string &TriaGenerator,
        const std::string &TriaGenerator_Options,
        const MPI_Comm &mpi_comm) :
        combustion::Grid_DWR<dim,spacedim> (TriaGenerator, TriaGenerator_Options,mpi_comm),
        Grid_Class_Options(Grid_Class_Options) { };

    virtual ~Grid_DWR_Dirichlet_Neumann_and_Robin() = default;

    virtual void set_boundary_indicators();

private:
    const std::string Grid_Class_Options;
};

}} // namespace

#endif
