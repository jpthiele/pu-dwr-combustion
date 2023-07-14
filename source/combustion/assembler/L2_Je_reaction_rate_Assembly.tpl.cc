/**
 * @file L2_Je_reaction_rate_Assembly.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhäuser (MPB)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 */

/*  Copyright (C) 2012-2023 by Jan Philipp Thiele, Uwe Koecher                */
/*                          and Marius Paul Bruchhaeuser                      */
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
#include <combustion/assembler/L2_Je_reaction_rate_Assembly.tpl.hh>

namespace combustion {
namespace Assemble {
namespace L2 {
namespace Je_reaction_rate {

namespace Assembly {
namespace Scratch {

/// (Struct-) Constructor.
template<int dim>
Je_reaction_rateAssembly<dim>::Je_reaction_rateAssembly(
    const dealii::FiniteElement<dim> &fe,
    const dealii::Quadrature<dim> &quad,
    const dealii::UpdateFlags &uflags) :
    fe_values(fe, quad, uflags),
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
Je_reaction_rateAssembly<dim>::Je_reaction_rateAssembly(
    const Je_reaction_rateAssembly &scratch) :
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
Je_reaction_rateAssembly<dim>::Je_reaction_rateAssembly(
    const dealii::FiniteElement<dim> &fe) :
    vi_Jei_vector(fe.dofs_per_cell),
    local_dof_indices(fe.dofs_per_cell) {
}


/// (Struct-) Copy constructor.
template<int dim>
Je_reaction_rateAssembly<dim>::Je_reaction_rateAssembly(
    const Je_reaction_rateAssembly& copydata) :
    vi_Jei_vector(copydata.vi_Jei_vector),
    local_dof_indices(copydata.local_dof_indices) {
}

}}
////////////////////////////////////////////////////////////////////////////////


template<int dim>
Assembler<dim>::
Assembler(
    std::shared_ptr< dealii::DoFHandler<dim> > dof,
    std::shared_ptr< dealii::FiniteElement<dim> > fe,
    std::shared_ptr< dealii::AffineConstraints<double> > constraints) :
    dof(dof),
    fe(fe),
    constraints(constraints) {
    // init update flags
    uflags =
        dealii::update_quadrature_points |
        dealii::update_values |
        dealii::update_JxW_values;
    domain_area = 0;
}

template<int dim>
void
Assembler<dim>::set_parameters(
    double alpha,
    double beta,
    double Le,
    double area) {
        param.alpha = alpha;
        param.beta = beta;
        param.c = beta*beta/(2*Le);
        domain_area = area;
}

template<int dim>
void Assembler<dim>::assemble(
    std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > _Je,
    std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > _u_h,
    const unsigned int q,
    const bool quadrature_points_auto_mode) {
    // init
    Je = _Je;
    Assert( Je.use_count(), dealii::ExcNotInitialized() );

    Assert( _u_h.use_count(), dealii::ExcNotInitialized() );
    primal.u = _u_h;
	
        
    const dealii::QGauss<dim> quad{
        (quadrature_points_auto_mode ? (fe->tensor_degree()+5) : q)
    };

    if (!quad.size()) {
        return;
    }
	
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
        Assembly::Scratch::Je_reaction_rateAssembly<dim> (*fe, quad, uflags),
        Assembly::CopyData::Je_reaction_rateAssembly<dim> (*fe)
    );
    Je->compress(dealii::VectorOperation::add);
}


template<int dim>
void Assembler<dim>::local_assemble_cell(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::Je_reaction_rateAssembly<dim> &scratch,
    Assembly::CopyData::Je_reaction_rateAssembly<dim> &copydata) {
    // reinit fe_values and init local to global dof mapping
    scratch.fe_values.reinit(cell);
    scratch.dofs_per_cell = scratch.fe_values.get_fe().dofs_per_cell;
    cell->get_dof_indices(copydata.local_dof_indices);

    // initialize local vector
    copydata.vi_Jei_vector = 0;

    scratch.fe_values.get_function_values(*primal.u,
                                          scratch.old_solution_values);

    // assemble cell terms
    for (scratch.q=0; scratch.q < scratch.fe_values.n_quadrature_points;
        ++scratch.q) {
        scratch.theta = scratch.old_solution_values[scratch.q](0);
        scratch.Y     = scratch.old_solution_values[scratch.q](1);

        scratch.tm1 = scratch.theta-1;
        scratch.JxW = scratch.fe_values.JxW(scratch.q);
        
        for ( scratch.k=0; scratch.k < scratch.dofs_per_cell; ++scratch.k ){
              scratch.phi[scratch.k] =
            scratch.fe_values.shape_value(
                scratch.k,
                scratch.q
            );
        }
        
        // assemble
        for (scratch.i=0; scratch.i < scratch.fe_values.get_fe().dofs_per_cell;
            ++scratch.i) {

            if(scratch.fe_values.get_fe().
               system_to_component_index(scratch.i).first == 0 )
            {
                //first component (omega' w.r.t. theta)
                copydata.vi_Jei_vector[scratch.i] +=
                    param.c*
                    param.beta/((1+param.alpha*scratch.tm1)*
                                (1+param.alpha*scratch.tm1))*
                    scratch.Y*
                    /*exp(f(theta)) */
                    std::exp(
                        param.beta*scratch.tm1/(
                        1+param.alpha*scratch.tm1)
                    )*
                    scratch.phi[scratch.i] *
                    scratch.JxW/domain_area;
            }
            else {
                //second component (omega' w.r.t. Y)
                copydata.vi_Jei_vector[scratch.i] +=
                    param.c*
                    /*exp(f(theta)) */
                    std::exp(
                        param.beta*scratch.tm1/(
                        1+param.alpha*scratch.tm1)
                    )*
                    scratch.phi[scratch.i] *
                    scratch.JxW/domain_area;
            }
                
        } // for i
    } // for q
}


template<int dim>
void Assembler<dim>::copy_local_to_global_cell(
    const Assembly::CopyData::Je_reaction_rateAssembly<dim> &copydata) {
    constraints->distribute_local_to_global(
        copydata.vi_Jei_vector, copydata.local_dof_indices, *Je
    );
}


}}}}

#include "L2_Je_reaction_rate_Assembly.inst.in"
