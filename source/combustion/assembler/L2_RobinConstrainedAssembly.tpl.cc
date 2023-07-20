/**
 * @file L2_RobinConstrainedAssembly.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
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

// PROJECT includes
#include <combustion/assembler/L2_RobinConstrainedAssembly.tpl.hh>
#include <combustion/types/boundary_id.hh>

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/filtered_iterator.h>

// C++ includes
#include <functional>
#include <iterator>

namespace combustion {
namespace Assemble {
namespace L2 {
namespace RobinConstrained {

namespace Assembly {
namespace Scratch {

/// (Struct-) Constructor.
template<int dim>
RobinConstrainedAssembly<dim>::RobinConstrainedAssembly(
    const dealii::FiniteElement<dim> &fe,
    const dealii::Quadrature<dim-1> &quad_face,
    const dealii::UpdateFlags &uflags_face) :
    fe_face_values(fe, quad_face, uflags_face),
    phi(fe.dofs_per_cell),
    JxW(0),
    u_R(0),
    face_no(0),
    q(0),
    k(0),
    i(0){
}


/// (Struct-) Copy constructor.
template<int dim>
RobinConstrainedAssembly<dim>::RobinConstrainedAssembly(
    const RobinConstrainedAssembly &scratch) :
    fe_face_values(
        scratch.fe_face_values.get_fe(),
        scratch.fe_face_values.get_quadrature(),
        scratch.fe_face_values.get_update_flags()),
    phi(scratch.phi),
    JxW(scratch.JxW),
    u_R(scratch.u_R),
    face_no(scratch.face_no),
    q(scratch.q),
    k(scratch.k),
    i(scratch.i) {
}

}

namespace CopyData {

/// (Struct-) Constructor.
template<int dim>
RobinConstrainedAssembly<dim>::RobinConstrainedAssembly(
    const dealii::FiniteElement<dim> &fe) :
    fi_vi_vector(fe.dofs_per_cell),
    local_dof_indices(fe.dofs_per_cell) {
}


/// (Struct-) Copy constructor.
template<int dim>
RobinConstrainedAssembly<dim>::RobinConstrainedAssembly(
    const RobinConstrainedAssembly& copydata) :
    fi_vi_vector(copydata.fi_vi_vector),
    local_dof_indices(copydata.local_dof_indices) {
}

}}
////////////////////////////////////////////////////////////////////////////////


template<int dim>
Assembler<dim>::
Assembler(
    std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > u_R,
    std::shared_ptr< dealii::DoFHandler<dim> > dof,
    std::shared_ptr< dealii::FiniteElement<dim> > fe,
    std::shared_ptr< dealii::AffineConstraints<double> > constraints) :
    u_R(u_R),
    dof(dof),
    fe(fe),
    constraints(constraints) {
    uflags_face =
        dealii::update_quadrature_points |
        dealii::update_values |
        dealii::update_JxW_values;
}


template<int dim>
void
Assembler<dim>::set_function(
    std::shared_ptr< dealii::Function<dim> > fun) {
    function.u_R = fun;
}


template<int dim>
void Assembler<dim>::assemble(
    const double time,
    const unsigned int q) {
    // init. global vector
    Assert(u_R.use_count(), dealii::ExcNotInitialized());
    *u_R = 0;

    // check
    Assert(function.u_R.use_count(), dealii::ExcNotInitialized());
    function.u_R->set_time(time);

    // setup quadrature; return if q==0
    const dealii::QGauss<dim-1> quad_face{q};
    if (!quad_face.size())
        return;
	
    typedef dealii::FilteredIterator<
        const typename dealii::DoFHandler<dim>::active_cell_iterator>
        CellFilter;
	
    // Using WorkStream to assemble.
    dealii::WorkStream::
    run(
        CellFilter(
            dealii::IteratorFilters::LocallyOwnedCell(), dof->begin_active()
        ),
        CellFilter(
            dealii::IteratorFilters::LocallyOwnedCell(), dof->end()
        ),
        std::bind (
            &Assembler<dim>::local_assemble_cell,
            this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3
        ),
        std::bind (
            &Assembler<dim>::copy_local_to_global_cell,
            this,
            std::placeholders::_1
        ),
        Assembly::Scratch::RobinConstrainedAssembly<dim> (
            *fe,
            quad_face,
            uflags_face
        ),
        Assembly::CopyData::RobinConstrainedAssembly<dim> (*fe)
    );
}


template<int dim>
void Assembler<dim>::local_assemble_cell(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::RobinConstrainedAssembly<dim> &scratch,
    Assembly::CopyData::RobinConstrainedAssembly<dim> &copydata) {
    // get global indices
    cell->get_dof_indices(copydata.local_dof_indices);

    // initialize local vector
    copydata.fi_vi_vector = 0;
	
    if (cell->at_boundary())
    for (scratch.face_no=0;
         scratch.face_no < dealii::GeometryInfo<dim>::faces_per_cell;
         ++scratch.face_no)
    if ((cell->face(scratch.face_no)->at_boundary()) &&
        (cell->face(scratch.face_no)->boundary_id() ==
        static_cast<dealii::types::boundary_id> (
        combustion::types::boundary_id::Robin)))
    {
        ////////////////////////////////////////////////////////////////////////////
        // initialize
        //
		
        // reinit scratch and data to current cell
        scratch.fe_face_values.reinit(cell,scratch.face_no);

        ////////////////////////////////////////////////////////////////////////////
        // assemble terms
        //

        for (scratch.q=0;
             scratch.q < scratch.fe_face_values.n_quadrature_points;
             ++scratch.q)
        {
            ////////////////////////////////////////////////////////////////////////
            // prefetch data on the current quadrature point

            scratch.JxW = scratch.fe_face_values.JxW(scratch.q);

            scratch.u_R = function.u_R->value(
                scratch.fe_face_values.quadrature_point(scratch.q),
                0 // component
            );

            // prefetch data
            for (scratch.k=0;
                scratch.k < scratch.fe_face_values.get_fe().dofs_per_cell;
                ++scratch.k) {
                scratch.phi[scratch.k] =
                    scratch.fe_face_values.shape_value_component(
                        scratch.k,
                        scratch.q,
                        0 // component
                    );
            }

            ////////////////////////////////////////////////////////////////////////
            // assemble
            //

            // u_R (Robin boundary function)
            for (scratch.i=0;
                scratch.i < scratch.fe_face_values.get_fe().dofs_per_cell;
                ++scratch.i) {
                copydata.fi_vi_vector[scratch.i] += (
                    scratch.phi[scratch.i] *
                    scratch.u_R *
                    scratch.JxW
                );
            }
        }
    }
}


template<int dim>
void Assembler<dim>::copy_local_to_global_cell(
    const Assembly::CopyData::RobinConstrainedAssembly<dim> &copydata) {
    constraints->distribute_local_to_global(
        copydata.fi_vi_vector, copydata.local_dof_indices, *u_R
    );
}


}}}}

#include "L2_RobinConstrainedAssembly.inst.in"
