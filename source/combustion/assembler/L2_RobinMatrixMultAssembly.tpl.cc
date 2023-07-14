/**
 * @file L2_RobinMatrixMultAssembly.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 */

/*  Copyright (C) 2012-2023 by Jan Philipp Thiele and Uwe Koecher             */
/*                                                                            */
/*  This file is part of pu-dwr-diffusion                                     */
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
#include <combustion/assembler/L2_RobinMatrixMultAssembly.tpl.hh>
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
namespace RobinMatrixMult {

namespace Assembly {
namespace Scratch {

/// (Struct-) Constructor.
template<int dim>
RobinMatrixMultAssembly<dim>::RobinMatrixMultAssembly(
	const dealii::FiniteElement<dim> &fe,
	const dealii::Quadrature<dim-1> &quad_face,
	const dealii::UpdateFlags &uflags_face) :
	fe_face_values(fe, quad_face, uflags_face),
	phi(fe.dofs_per_cell),
	JxW(0),
	dofs_per_cell(0),
	old_solution_values(
            fe_face_values.n_quadrature_points,
            dealii::Vector<double>(2)
        ),
	face_no(0),
	q(0),
	k(0),
	i(0){
}


/// (Struct-) Copy constructor.
template<int dim>
RobinMatrixMultAssembly<dim>::RobinMatrixMultAssembly(
	const RobinMatrixMultAssembly &scratch) :
	fe_face_values(
            scratch.fe_face_values.get_fe(),
            scratch.fe_face_values.get_quadrature(),
            scratch.fe_face_values.get_update_flags()
        ),
	phi(scratch.phi),
	JxW(scratch.JxW),
	dofs_per_cell(scratch.dofs_per_cell),
	old_solution_values(scratch.old_solution_values),
	face_no(scratch.face_no),
	q(scratch.q),
	k(scratch.k),
	i(scratch.i){
}

}

namespace CopyData {

/// (Struct-) Constructor.
template<int dim>
RobinMatrixMultAssembly<dim>::RobinMatrixMultAssembly(
	const dealii::FiniteElement<dim> &fe) :
	ui_vi_vector(fe.dofs_per_cell),
	local_dof_indices(fe.dofs_per_cell) {
}


/// (Struct-) Copy constructor.
template<int dim>
RobinMatrixMultAssembly<dim>::RobinMatrixMultAssembly(
	const RobinMatrixMultAssembly& copydata) :
	ui_vi_vector(copydata.ui_vi_vector),
	local_dof_indices(copydata.local_dof_indices) {
}

}}
////////////////////////////////////////////////////////////////////////////////


template<int dim>
Assembler<dim>::
Assembler(
    std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > Ru,
    std::shared_ptr< dealii::DoFHandler<dim> > dof,
    std::shared_ptr< dealii::FiniteElement<dim> > fe,
    std::shared_ptr< dealii::AffineConstraints<double> > constraints) :
    Ru(Ru),
    dof(dof),
    fe(fe),
    constraints(constraints),
    factor(0){
    uflags_face =
        dealii::update_quadrature_points |
        dealii::update_values |
        dealii::update_JxW_values;
}

template<int dim>
void Assembler<dim>::
set_parameters (
    double theta_factor,
    double Y_factor
){
    factors.push_back(theta_factor);
    factors.push_back(Y_factor);
}


template<int dim>
void Assembler<dim>::assemble(
    double _factor,
    std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > _u,
    const unsigned int q)
{
	
    factor = _factor;

    AssertThrow( _u.use_count(), dealii::ExcNotInitialized());
    primal.u = _u;

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
        Assembly::Scratch::RobinMatrixMultAssembly<dim> (
            *fe,
            quad_face,
            uflags_face
        ),
        Assembly::CopyData::RobinMatrixMultAssembly<dim> (*fe)
    );

    Ru->compress(dealii::VectorOperation::add);
}


template<int dim>
void Assembler<dim>::local_assemble_cell(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::RobinMatrixMultAssembly<dim> &scratch,
    Assembly::CopyData::RobinMatrixMultAssembly<dim> &copydata) {
    // get global indices
    scratch.dofs_per_cell = scratch.fe_face_values.get_fe().dofs_per_cell;
    cell->get_dof_indices(copydata.local_dof_indices);

    // initialize local matrix with zeros
    copydata.ui_vi_vector = 0;

    if (cell->at_boundary())
    {
        for (scratch.face_no=0;
            scratch.face_no < dealii::GeometryInfo<dim>::faces_per_cell;
            ++scratch.face_no)
        {

            if ((cell->face(scratch.face_no)->at_boundary()) &&
                (cell->face(scratch.face_no)->boundary_id() ==
                static_cast<dealii::types::boundary_id> (
                combustion::types::boundary_id::Robin)
                )
            ) {
                ////////////////////////////////////////////////////////////////////////////
                // initialize
                //
                // reinit scratch and data to current cell
                scratch.fe_face_values.reinit(cell,scratch.face_no);

                ////////////////////////////////////////////////////////////////////////////
                // assemble terms
                //

                scratch.fe_face_values.get_function_values(*primal.u,scratch.old_solution_values);

                for (scratch.q=0;
                     scratch.q < scratch.fe_face_values.n_quadrature_points;
                     ++scratch.q)
                {
                    ////////////////////////////////////////////////////////////////////////
                    // prefetch data on the current quadrature point

                    scratch.JxW = scratch.fe_face_values.JxW(scratch.q)*factor;


                    // prefetch data
                    for (scratch.k=0;
                        scratch.k < scratch.dofs_per_cell;
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

                    // R (Robin boundary matrix)
                    for (scratch.i=0; scratch.i < scratch.dofs_per_cell; ++scratch.i) {

                        const unsigned int comp_i =scratch.fe_face_values.get_fe().
                            system_to_component_index(scratch.i).first;

                        copydata.ui_vi_vector(scratch.i) += (
                            (scratch.phi[scratch.i] *
                                scratch.old_solution_values[scratch.q](comp_i)) *
                                            scratch.JxW*factors[comp_i]
                        );

                    } // for ij
                } // for quad points
            } // if face boundary_id
        } // for face      
    } //boundary cell
}


template<int dim>
void Assembler<dim>::copy_local_to_global_cell(
    const Assembly::CopyData::RobinMatrixMultAssembly<dim> &copydata) {
    constraints->distribute_local_to_global(
        copydata.ui_vi_vector,
        copydata.local_dof_indices,// copydata.local_dof_indices,
        *Ru
    );
}


}}}}

#include "L2_RobinMatrixMultAssembly.inst.in"
