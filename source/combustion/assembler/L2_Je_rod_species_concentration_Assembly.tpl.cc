/**
 * @file L2_Je_rod_species_concentration_Assembly.tpl.cc
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
#include <combustion/assembler/L2_Je_rod_species_concentration_Assembly.tpl.hh>

#include <combustion/types/boundary_id.hh>
namespace combustion {
namespace Assemble {
namespace L2 {
namespace Je_rod_species_concentration {

namespace Assembly {
namespace Scratch {

/// (Struct-) Constructor.
template<int dim>
Je_rod_species_concentrationAssembly<dim>::Je_rod_species_concentrationAssembly(
	const dealii::FiniteElement<dim> &fe,
	const dealii::Quadrature<dim-1> &quad_face,
	const dealii::UpdateFlags &uflags) :
	fe_face_values(fe, quad_face, uflags),
	phi(fe.dofs_per_cell),
	dofs_per_cell(0),
	JxW(0.),
	q(0),
	face(0),
	i(0){
}


/// (Struct-) Copy constructor.
template<int dim>
Je_rod_species_concentrationAssembly<dim>::Je_rod_species_concentrationAssembly(
    const Je_rod_species_concentrationAssembly &scratch) :
    fe_face_values(
        scratch.fe_face_values.get_fe(),
        scratch.fe_face_values.get_quadrature(),
        scratch.fe_face_values.get_update_flags()),
    phi(scratch.phi),
    dofs_per_cell(scratch.dofs_per_cell),
    JxW(scratch.JxW),
    q(scratch.q),
    face(scratch.face),
    i(scratch.i) {
}

}

namespace CopyData {

/// (Struct-) Constructor.
template<int dim>
Je_rod_species_concentrationAssembly<dim>::Je_rod_species_concentrationAssembly(
    const dealii::FiniteElement<dim> &fe) :
    vi_Jei_vector(fe.dofs_per_cell),
    local_dof_indices(fe.dofs_per_cell) {
}


/// (Struct-) Copy constructor.
template<int dim>
Je_rod_species_concentrationAssembly<dim>::Je_rod_species_concentrationAssembly(
    const Je_rod_species_concentrationAssembly& copydata) :
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
    constraints(constraints),
    rod_surface_area(0.){
    // init update flags
    uflags_face =
        dealii::update_quadrature_points |
        dealii::update_values |
        dealii::update_JxW_values;
}

template<int dim>
void
Assembler<dim>::set_parameters(double area) {
    rod_surface_area = area;
}

template<int dim>
void Assembler<dim>::assemble(
    std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > _Je,
    const unsigned int q,
    const bool quadrature_points_auto_mode) {
    // init
    Je = _Je;
    Assert( Je.use_count(), dealii::ExcNotInitialized() );

    const dealii::QGauss<dim-1> face_quad{
        (quadrature_points_auto_mode ? (fe->tensor_degree()+5) : q)
    };

    if (!face_quad.size()) {
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
        Assembly::Scratch::Je_rod_species_concentrationAssembly<dim> (*fe, face_quad, uflags_face),
        Assembly::CopyData::Je_rod_species_concentrationAssembly<dim> (*fe)
    );
    Je->compress(dealii::VectorOperation::add);
}


template<int dim>
void Assembler<dim>::local_assemble_cell(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::Je_rod_species_concentrationAssembly<dim> &scratch,
    Assembly::CopyData::Je_rod_species_concentrationAssembly<dim> &copydata)
{

    scratch.dofs_per_cell = scratch.fe_face_values.get_fe().dofs_per_cell;
    cell->get_dof_indices(copydata.local_dof_indices);
	
    // initialize local vector
    copydata.vi_Jei_vector = 0;

    for ( scratch.face = 0 ;
          scratch.face < dealii::GeometryInfo<dim>::faces_per_cell ;
          scratch.face++ )
    if ( cell -> face(scratch.face)->at_boundary() &&
         (cell->face(scratch.face)->boundary_id() ==
             static_cast<dealii::types::boundary_id> (
             combustion::types::boundary_id::Robin))
    ){
        scratch.fe_face_values.reinit(cell,scratch.face);

        // assemble face terms
        for (scratch.q=0; scratch.q < scratch.fe_face_values.n_quadrature_points;
             ++scratch.q) {

            scratch.JxW = scratch.fe_face_values.JxW(scratch.q);

            for ( scratch.i=0; scratch.i < scratch.dofs_per_cell; ++scratch.i ){
                scratch.phi[scratch.i] =
                    scratch.fe_face_values.shape_value(
                        scratch.i,
                        scratch.q
                );
            }
            for ( scratch.i=0; scratch.i < scratch.dofs_per_cell; ++scratch.i ){
                if(scratch.fe_face_values.get_fe().
                    system_to_component_index(scratch.i).first == 1 )
                {
                    copydata.vi_Jei_vector[scratch.i]+=
                        scratch.phi[scratch.i]*
                        scratch.JxW/rod_surface_area;
                }
            }
        }
    }
}


template<int dim>
void Assembler<dim>::copy_local_to_global_cell(
    const Assembly::CopyData::Je_rod_species_concentrationAssembly<dim> &copydata) {
    constraints->distribute_local_to_global(
        copydata.vi_Jei_vector, copydata.local_dof_indices, *Je
    );
}


}}}}

#include "L2_Je_rod_species_concentration_Assembly.inst.in"
