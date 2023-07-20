/**
 * @file L2_MassAssembly.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 *
 * @brief Purpose: Assemble Mass Matrix (L2-integrals)
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
#include <combustion/assembler/L2_MassAssembly.tpl.hh>
#include <deal.II/fe/fe_values_extractors.h>

namespace combustion {
namespace Assemble {
namespace L2 {
namespace Mass {

namespace Assembly {
namespace Scratch {

/// (Struct-) Constructor.
template<int dim>
MassAssembly<dim>::MassAssembly(
    const dealii::FiniteElement<dim> &fe,
    const dealii::Quadrature<dim> &quad,
    const dealii::UpdateFlags &uflags_cell) :
    fe_values(fe, quad, uflags_cell),
    phi(fe_values.get_fe().dofs_per_cell),
    JxW(0),
    dofs_per_cell(0),
    q(0),
    k(0),
    i(0),
    j(0){
}


/// (Struct-) Copy constructor.
template<int dim>
MassAssembly<dim>::MassAssembly(const MassAssembly &scratch) :
    fe_values(
        scratch.fe_values.get_fe(),
        scratch.fe_values.get_quadrature(),
        scratch.fe_values.get_update_flags()),
    phi(scratch.phi),
    JxW(scratch.JxW),
    dofs_per_cell(scratch.dofs_per_cell),
    q(scratch.q),
    k(scratch.k),
    i(scratch.i),
    j(scratch.j) {
}

} // namespace Scratch
namespace CopyData {

/// (Struct-) Constructor.
template<int dim>
MassAssembly<dim>::MassAssembly(
    const dealii::FiniteElement<dim> &fe) :
    ui_vi_matrix(fe.dofs_per_cell, fe.dofs_per_cell),
    local_dof_indices(fe.dofs_per_cell) {
}


/// (Struct-) Copy constructor.
template<int dim>
MassAssembly<dim>::MassAssembly(const MassAssembly &copydata) :
    ui_vi_matrix(copydata.ui_vi_matrix),
    local_dof_indices(copydata.local_dof_indices) {
}

} // namespace CopyData
} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////


/// Constructor.
template<int dim>
Assembler<dim>::
Assembler(
    std::shared_ptr< dealii::TrilinosWrappers::SparseMatrix > M,
    std::shared_ptr< dealii::DoFHandler<dim> > dof,
    std::shared_ptr< dealii::FiniteElement<dim> > fe,
    std::shared_ptr< dealii::AffineConstraints<double> > constraints) :
    M(M),
    dof(dof),
    fe(fe),
    constraints(constraints) {
    // init UpdateFlags
    uflags =
            dealii::update_quadrature_points |
            dealii::update_values |
            dealii::update_JxW_values;
    factor = 0;
}



template<int dim>
void Assembler<dim>::assemble(
    double _factor,
    const unsigned int q)
{
    factor = _factor;
    // assemble matrix
    const dealii::QGauss<dim> quad{ (q ? q : (fe->tensor_degree()+5)) };

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
        Assembly::Scratch::MassAssembly<dim> (*fe, quad, uflags),
        Assembly::CopyData::MassAssembly<dim> (*fe)
    );

    M->compress(dealii::VectorOperation::add);
}


/// Local assemble on cell.
template<int dim>
void Assembler<dim>::local_assemble_cell(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::MassAssembly<dim> &scratch,
    Assembly::CopyData::MassAssembly<dim> &copydata)
{
	
    const dealii::FEValuesExtractors::Scalar theta(0);
    const dealii::FEValuesExtractors::Scalar Y(1);
    // reinit scratch and data to current cell
    scratch.fe_values.reinit(cell);
    scratch.dofs_per_cell = scratch.fe_values.get_fe().dofs_per_cell;
    cell->get_dof_indices(copydata.local_dof_indices);
	
    // initialize local matrix with zeros
    copydata.ui_vi_matrix = 0;

    // assemble cell terms
    for (scratch.q=0; scratch.q < scratch.fe_values.n_quadrature_points;
         ++scratch.q) {
                
        scratch.JxW = scratch.fe_values.JxW(scratch.q)*factor;
                        
        // loop over all basis functions to get the shape values
        for (scratch.k=0; scratch.k < scratch.dofs_per_cell; ++scratch.k) {
            scratch.phi[scratch.k] =
                scratch.fe_values.shape_value(
                    scratch.k,
                    scratch.q
                );
        }
		
        // loop over all test & trial function combinitions to get the assembly
        for (scratch.i=0; scratch.i < scratch.dofs_per_cell; ++scratch.i)
        for (scratch.j=0; scratch.j < scratch.dofs_per_cell; ++scratch.j) {
            const unsigned int comp_j =scratch.fe_values.get_fe().
                system_to_component_index(scratch.j).first;

            const unsigned int comp_i =scratch.fe_values.get_fe().
                system_to_component_index(scratch.i).first;
            if ( comp_i == comp_j )
            {
                copydata.ui_vi_matrix(scratch.i,scratch.j) += (
                    (scratch.phi[scratch.i] * scratch.phi[scratch.j]) *
                    scratch.JxW
                );
            }
        } // for ij
		
    } // for q
	
}


/// Copy local assembly to global matrix.
template<int dim>
void Assembler<dim>::copy_local_to_global_cell(
	const Assembly::CopyData::MassAssembly<dim> &copydata) {
	// copy mass matrix
	constraints->distribute_local_to_global(
		copydata.ui_vi_matrix,
		copydata.local_dof_indices,// copydata.local_dof_indices,
		*M
	);
}

}}}} // namespaces

#include "L2_MassAssembly.inst.in"
