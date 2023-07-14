/**
 * @file Grid_DWR.tpl.hh
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

#ifndef __Grid_DWR_tpl_hh
#define __Grid_DWR_tpl_hh

// PROJECT includes
#include <DTM++/types/slabs.tpl.hh>
#include <combustion/types/boundary_id.hh>

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/vector_tools.h>

// C++ includes
#include <memory>
#include <algorithm>
#include <list>
#include <iterator>

namespace combustion {

template<int dim, int spacedim>
class Grid_DWR {
public:
    Grid_DWR(
        const std::string &TriaGenerator,
        const std::string &TriaGenerator_Options,
        const MPI_Comm &mpi_comm) :
        TriaGenerator(TriaGenerator),
        TriaGenerator_Options(TriaGenerator_Options),
        mpi_comm(mpi_comm),
        coarse_tria(){ };

    virtual ~Grid_DWR();

    virtual void initialize_slabs(
        const unsigned int &p_low,
        const unsigned int &q_high,
        const double &t0,
        const double &T,
        const double &tau_n
    );

    virtual void refine_slab_in_time(
        typename DTM::types::spacetime::dwr::slabs<dim>::iterator slab
    );

    virtual void generate();
    virtual void refine_global(const unsigned int n = 1);
    virtual void set_boundary_indicators();

    virtual void distribute_dofs();

    DTM::types::spacetime::dwr::slabs<dim> slabs;

    dealii::GridIn<dim>            grid_in;
    dealii::GridOut                grid_out;
	
protected:
    const std::string TriaGenerator;
    const std::string TriaGenerator_Options;
    const MPI_Comm    &mpi_comm;
    std::shared_ptr<dealii::Triangulation<dim> > coarse_tria;
};

} // namespace

#endif
