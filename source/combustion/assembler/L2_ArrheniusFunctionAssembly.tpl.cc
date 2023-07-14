/**
 * @file L2_FunctionAssembly.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 *
 * @brief Purpose: Assemble weak Function operator @f$ (\nabla v, \epsilon \nabla u) @f$
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
#include <combustion/assembler/L2_ArrheniusFunctionAssembly.tpl.hh>

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
namespace Arrhenius{
namespace Function{
        
namespace Assembly {
namespace Scratch {

/// (Struct-) Constructor.
template<int dim>
FunctionAssembly<dim>::FunctionAssembly(
    const dealii::FiniteElement<dim> &fe,
    const dealii::Quadrature<dim> &quad,
    const dealii::UpdateFlags &uflags_cell) :
    fe_values(fe, quad, uflags_cell),
    phi(fe.dofs_per_cell),
    dofs_per_cell(0),
    JxW(0.),
    old_solution_values(fe_values.n_quadrature_points, dealii::Vector<double>(2)),
    theta(0.),
    Y(0.),
    tm1(0.),
    q(0),
    k(0),
    i(0){
}


/// (Struct-) Copy constructor.
template<int dim>
FunctionAssembly<dim>::FunctionAssembly(const FunctionAssembly &scratch) :
    fe_values(
        scratch.fe_values.get_fe(),
        scratch.fe_values.get_quadrature(),
        scratch.fe_values.get_update_flags()),
    phi(scratch.phi),
    dofs_per_cell(scratch.dofs_per_cell),
    JxW(scratch.JxW),
    old_solution_values(scratch.old_solution_values),
    theta(scratch.theta),
    Y(scratch.Y),
    tm1(scratch.tm1),
    q(scratch.q),
    k(scratch.k),
    i(scratch.i) {
}

}

namespace CopyData {

/// (Struct-) Constructor.
template<int dim>
FunctionAssembly<dim>::FunctionAssembly(const dealii::FiniteElement<dim> &fe) :
    fi_vi_vector(fe.dofs_per_cell),
    local_dof_indices(fe.dofs_per_cell) {
}


/// (Struct-) Copy constructor.
template<int dim>
FunctionAssembly<dim>::FunctionAssembly(const FunctionAssembly &copydata) :
    fi_vi_vector(copydata.fi_vi_vector),
    local_dof_indices(copydata.local_dof_indices) {
}

}}
////////////////////////////////////////////////////////////////////////////////


/// Constructor.
template<int dim>
Assembler<dim>::
Assembler(
    std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > f,
    std::shared_ptr< dealii::DoFHandler<dim> > dof,
    std::shared_ptr< dealii::FiniteElement<dim> > fe,
    std::shared_ptr< dealii::AffineConstraints<double> > constraints) :
    f(f),
    dof(dof),
    fe(fe),
    constraints(constraints) {
    // init UpdateFlags
    uflags =
        dealii::update_quadrature_points |
        dealii::update_values |
        dealii::update_JxW_values;
}

template<int dim>
void 
Assembler<dim>::set_parameters(
    double alpha,
    double beta,
    double Le){
       param.alpha = alpha;
       param.beta = beta;
       param.Le = Le;
       param.c = beta*beta/(2*Le);
}


template<int dim>
void
Assembler<dim>::
assemble(
    std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > _u,
    const unsigned int q) {

    ////////////////////////////////////////////////////////////////////////////
    // check
    AssertThrow( f.use_count(), dealii::ExcNotInitialized());
    AssertThrow( dof.use_count(), dealii::ExcNotInitialized() );
    AssertThrow( fe.use_count(), dealii::ExcNotInitialized() );
    AssertThrow( constraints.use_count(), dealii::ExcNotInitialized() );

    AssertThrow( _u.use_count(), dealii::ExcNotInitialized());
    primal.u = _u;
    ////////////////////////////////////////////////////////////////////////////
    // assemble matrix

    // create quadrature on cells
    const dealii::QGauss<dim> quad( (q ? q : (fe->tensor_degree()+5)) );

    typedef
    dealii::
    FilteredIterator<const typename dealii::DoFHandler<dim>::active_cell_iterator>
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
        Assembly::Scratch::FunctionAssembly<dim> (*fe, quad, uflags),
        Assembly::CopyData::FunctionAssembly<dim> (*fe)
    );
    f->compress(dealii::VectorOperation::add);
}


/// Local assemble on cell.
template<int dim>
void Assembler<dim>::local_assemble_cell(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::FunctionAssembly<dim> &scratch,
    Assembly::CopyData::FunctionAssembly<dim> &copydata)
{
	
    // reinit scratch and data to current cell
    scratch.fe_values.reinit(cell);
    scratch.dofs_per_cell = scratch.fe_values.get_fe().dofs_per_cell;
    cell->get_dof_indices(copydata.local_dof_indices);

    // initialize local vector with zeros
    copydata.fi_vi_vector = 0;

    scratch.fe_values.get_function_values (*primal.u
                                          ,scratch.old_solution_values);

    // assemble cell terms
    for (scratch.q=0; scratch.q < scratch.fe_values.n_quadrature_points;
        ++scratch.q) {
        scratch.theta = scratch.old_solution_values[scratch.q](0);
        scratch.Y     = scratch.old_solution_values[scratch.q](1);

        scratch.tm1 = /*theta-1*/ scratch.theta-1;
        scratch.JxW = scratch.fe_values.JxW(scratch.q);
        
        // loop over all basis functions to get the shape gradient
        for (scratch.k=0; scratch.k < scratch.dofs_per_cell; ++scratch.k) {
            scratch.phi[scratch.k] =
                scratch.fe_values.shape_value(
                    scratch.k,
                    scratch.q
            );
        }

        // loop over all test & trial function combinitions to get the assembly
        for (scratch.i=0; scratch.i < scratch.dofs_per_cell; ++scratch.i) {

            copydata.fi_vi_vector[scratch.i] +=
                /*   phi vs. -psi  */
                std::pow(-1,scratch.fe_values.get_fe().
                     system_to_component_index(scratch.i).first)
                *
                param.c*scratch.Y*
                /* exp(f(theta)) */
                std::exp(
                param.beta*scratch.tm1/(
                1+param.alpha*scratch.tm1
                )
                )*
                scratch.phi[scratch.i] *
                scratch.JxW;
        }
    } // for q
}


/// Copy local assembly to global matrix.
template<int dim>
void Assembler<dim>::copy_local_to_global_cell(
    const Assembly::CopyData::FunctionAssembly<dim> &copydata) {
    constraints->distribute_local_to_global(
        copydata.fi_vi_vector, copydata.local_dof_indices, *f
    );
}


}}}}}

#include "L2_ArrheniusFunctionAssembly.inst.in"
