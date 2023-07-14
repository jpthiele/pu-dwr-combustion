/**
 * @file L2_LaplaceAssembly.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 *
 * @brief Purpose: Assemble weak Laplace operator @f$ (\nabla v, \epsilon \nabla u) @f$
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
#include <combustion/assembler/L2_LaplaceAssembly.tpl.hh>

// deal.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/base/quadrature_lib.h>

// C++ includes
// #include <iterator>
#include <functional>


namespace combustion {
namespace Assemble {
namespace L2 {
namespace Laplace {

namespace Assembly {
namespace Scratch {

/// (Struct-) Constructor.
template<int dim>
LaplaceAssembly<dim>::LaplaceAssembly(
	const dealii::FiniteElement<dim> &fe,
	const dealii::Quadrature<dim> &quad,
	const dealii::UpdateFlags &uflags_cell) :
	fe_values( fe, quad, uflags_cell),
	grad_phi(fe.dofs_per_cell),
	dofs_per_cell(0),
	JxW(0.),
	q(0),
	k(0),
	i(0),
	j(0){
}


/// (Struct-) Copy constructor.
template<int dim>
LaplaceAssembly<dim>::LaplaceAssembly(const LaplaceAssembly &scratch) :
    fe_values(
        scratch.fe_values.get_fe(),
        scratch.fe_values.get_quadrature(),
        scratch.fe_values.get_update_flags()),
    grad_phi(scratch.grad_phi),
    dofs_per_cell(scratch.dofs_per_cell),
    JxW(scratch.JxW),
    q(scratch.q),
    k(scratch.k),
    i(scratch.i),
    j(scratch.j) {
}

}

namespace CopyData {

/// (Struct-) Constructor.
template<int dim>
LaplaceAssembly<dim>::LaplaceAssembly(const dealii::FiniteElement<dim> &fe) :
    vi_ui_matrix(fe.dofs_per_cell, fe.dofs_per_cell),
    local_dof_indices(fe.dofs_per_cell) {
}


/// (Struct-) Copy constructor.
template<int dim>
LaplaceAssembly<dim>::LaplaceAssembly(const LaplaceAssembly &copydata) :
    vi_ui_matrix(copydata.vi_ui_matrix),
    local_dof_indices(copydata.local_dof_indices) {
}

}}
////////////////////////////////////////////////////////////////////////////////


/// Constructor.
template<int dim>
Assembler<dim>::
Assembler(
    std::shared_ptr< dealii::TrilinosWrappers::SparseMatrix > A,
    std::shared_ptr< dealii::DoFHandler<dim> > dof,
    std::shared_ptr< dealii::FiniteElement<dim> > fe,
    std::shared_ptr< dealii::AffineConstraints<double> > constraints) :
    A(A),
    dof(dof),
    fe(fe),
    constraints(constraints),
    factor(0.){
    // init UpdateFlags
    uflags =
        dealii::update_quadrature_points |
        dealii::update_gradients |
        dealii::update_JxW_values;
}


template<int dim>
void
Assembler<dim>::
assemble(
    double _factor,
    const unsigned int q) {

    ////////////////////////////////////////////////////////////////////////////
    // check
    AssertThrow( A.use_count(), dealii::ExcNotInitialized() );
    AssertThrow( dof.use_count(), dealii::ExcNotInitialized() );
    AssertThrow( fe.use_count(), dealii::ExcNotInitialized() );
    AssertThrow( constraints.use_count(), dealii::ExcNotInitialized() );

    factor = _factor;

    ////////////////////////////////////////////////////////////////////////////
    // assemble matrix

    // create quadrature on cells
    const dealii::QGauss<dim> quad( (q ? q : (fe->tensor_degree()+5)) );

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
        Assembly::Scratch::LaplaceAssembly<dim> (*fe, quad, uflags),
        Assembly::CopyData::LaplaceAssembly<dim> (*fe)
    );

    A->compress(dealii::VectorOperation::add);
}


/// Local assemble on cell.
template<int dim>
void Assembler<dim>::local_assemble_cell(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::LaplaceAssembly<dim> &scratch,
    Assembly::CopyData::LaplaceAssembly<dim> &copydata) {

    const dealii::FEValuesExtractors::Scalar theta(0);

    // reinit scratch and data to current cell
    scratch.fe_values.reinit(cell);
    scratch.dofs_per_cell = scratch.fe_values.get_fe().dofs_per_cell;
    cell->get_dof_indices(copydata.local_dof_indices);

    // initialize local matrix with zeros
    copydata.vi_ui_matrix = 0;

    // assemble cell terms
    for (scratch.q=0; scratch.q < scratch.fe_values.n_quadrature_points;
         ++scratch.q) {

        scratch.JxW = scratch.fe_values.JxW(scratch.q)*factor;

        // loop over all basis functions to get the shape gradient
        for (scratch.k=0; scratch.k < scratch.dofs_per_cell; ++scratch.k) {
            scratch.grad_phi[scratch.k] = scratch.fe_values.shape_grad(
                scratch.k,
                scratch.q
            );
        }
        
        // loop over all test & trial function combinitions to get the assembly
        for (scratch.i=0; scratch.i < scratch.dofs_per_cell; ++scratch.i) {
        for (scratch.j=0; scratch.j < scratch.dofs_per_cell; ++scratch.j) {
            const unsigned int comp_j =scratch.fe_values.get_fe().
                system_to_component_index(scratch.j).first;

            const unsigned int comp_i =scratch.fe_values.get_fe().
                system_to_component_index(scratch.i).first;
            if ( comp_i == comp_j )
            {
                copydata.vi_ui_matrix(scratch.i,scratch.j) +=
                    scratch.grad_phi[scratch.i] *
                    scratch.grad_phi[scratch.j] *
                    scratch.JxW;
            }
        }}
    } // for q
}


/// Copy local assembly to global matrix.
template<int dim>
void Assembler<dim>::copy_local_to_global_cell(
    const Assembly::CopyData::LaplaceAssembly<dim> &copydata) {
    constraints->distribute_local_to_global(
        copydata.vi_ui_matrix,
        copydata.local_dof_indices,// copydata.local_dof_indices,
        *A
    );
}


}}}}

#include "L2_LaplaceAssembly.inst.in"
