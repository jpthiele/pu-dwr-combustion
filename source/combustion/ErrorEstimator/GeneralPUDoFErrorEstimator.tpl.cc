/**
 * @file GeneralPUDoFErrorEstimator.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 *
 */

/*  Copyright (C) 2012-2023 by Jan Philipp Thiele and Uwe Koecher             */
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
#include <combustion/ErrorEstimator/GeneralPUDoFErrorEstimator.tpl.hh>

// DTM++ includes
#include <DTM++/base/LogStream.hh>

// DEAL.II includes
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/filtered_iterator.h>

// C++ includes

namespace combustion {
namespace dwr {

namespace estimator{

namespace Assembly {

namespace Scratch {

/// (Struct-) Constructor.
template<int dim>
PUDoFErrorEstimateOnCell<dim>::PUDoFErrorEstimateOnCell(
    const dealii::DoFHandler<dim> &dof_dual,
    const dealii::DoFHandler<dim> &dof_pu,
    const dealii::FiniteElement<dim> &fe_dual,
    const dealii::FiniteElement<dim> &fe_pu,
    const dealii::Quadrature<dim> &quad,
    const dealii::UpdateFlags &uflags) :
    fe_values_dual(fe_dual, quad, uflags),
    fe_values_pu(fe_pu,quad,uflags),
    dof_dual(dof_dual),
    dof_pu(dof_pu),
    local_dof_indices_dual(fe_dual.dofs_per_cell),
    phi(fe_dual.dofs_per_cell),
    grad_phi(fe_dual.dofs_per_cell),
    chi(fe_pu.dofs_per_cell),
    grad_chi(fe_pu.dofs_per_cell),
    local_u_kh_m(fe_dual.dofs_per_cell),
    local_u_kh_n(fe_dual.dofs_per_cell),
    local_u_k_n(fe_dual.dofs_per_cell),
    local_u_k_np1(fe_dual.dofs_per_cell),
    local_z_kh_n(fe_dual.dofs_per_cell),
    local_z_kh_np1(fe_dual.dofs_per_cell),
    local_z_k_n(fe_dual.dofs_per_cell),
    local_z_k_m(fe_dual.dofs_per_cell),
    val_dw_theta_j(0),
    val_dw_Y_j(0),
    val_pw_theta_j(0),
    val_pw_Y_j(0),
    val_theta_kh_j(0),
    val_Y_kh_j(0),
    val_z_theta_kh_j(0),
    val_z_Y_kh_j(0),
    val_theta_kh_jump_j(0),
    val_Y_kh_jump_j(0),
    val_z_theta_kh_jump_j(0),
    val_z_Y_kh_jump_j(0),
    omega(0.),
    omega_y(0.),
    omega_theta(0.),
    JxW(0.),
    q(0),
    d(0),
    j(0)
    {
}


/// (Struct-) Copy constructor.
template<int dim>
PUDoFErrorEstimateOnCell<dim>::PUDoFErrorEstimateOnCell(const PUDoFErrorEstimateOnCell &scratch) :
    fe_values_dual(
        scratch.fe_values_dual.get_fe(),
        scratch.fe_values_dual.get_quadrature(),
        scratch.fe_values_dual.get_update_flags()),
    fe_values_pu(
        scratch.fe_values_pu.get_fe(),
        scratch.fe_values_pu.get_quadrature(),
        scratch.fe_values_pu.get_update_flags()),
    dof_dual(scratch.dof_dual),
    dof_pu(scratch.dof_pu),
    local_dof_indices_dual(scratch.local_dof_indices_dual),
    phi(scratch.phi),
    grad_phi(scratch.grad_phi),
    chi(scratch.chi),
    grad_chi(scratch.grad_chi),
    local_u_kh_m(scratch.local_u_kh_m),
    local_u_kh_n(scratch.local_u_kh_n),
    local_u_k_n(scratch.local_u_k_n),
    local_u_k_np1(scratch.local_u_k_np1),
    local_z_kh_n(scratch.local_z_kh_n),
    local_z_kh_np1(scratch.local_z_kh_np1),
    local_z_k_n(scratch.local_z_k_n),
    local_z_k_m(scratch.local_z_k_m),

    val_dw_theta_j(scratch.val_dw_theta_j),
    val_dw_Y_j(scratch.val_dw_Y_j),
    val_pw_theta_j(scratch.val_pw_theta_j),
    val_pw_Y_j(scratch.val_pw_Y_j),

    val_theta_kh_j(scratch.val_theta_kh_j),
    val_Y_kh_j(scratch.val_Y_kh_j),
    val_z_theta_kh_j(scratch.val_z_theta_kh_j),
    val_z_Y_kh_j(scratch.val_z_Y_kh_j),

    val_theta_kh_jump_j(scratch.val_theta_kh_jump_j),
    val_Y_kh_jump_j(scratch.val_Y_kh_jump_j),
    val_z_theta_kh_jump_j(scratch.val_z_theta_kh_jump_j),
    val_z_Y_kh_jump_j(scratch.val_z_Y_kh_jump_j),

    grad_dw_theta_j(scratch.grad_dw_theta_j),
    grad_dw_Y_j(scratch.grad_dw_Y_j),
    grad_pw_theta_j(scratch.grad_pw_theta_j),
    grad_pw_Y_j(scratch.grad_pw_Y_j),
    grad_theta_kh_j(scratch.grad_theta_kh_j),
    grad_Y_kh_j(scratch.grad_Y_kh_j),
    grad_z_theta_kh_j(scratch.grad_z_theta_kh_j),
    grad_z_Y_kh_j(scratch.grad_z_Y_kh_j),
    omega(scratch.omega),
    omega_y(scratch.omega_y),
    omega_theta(scratch.omega_theta),
    JxW(scratch.JxW),
    q(scratch.q),
    d(scratch.d),
    j(scratch.j) {
}

template<int dim>
PUDoFErrorEstimateOnFace<dim>::PUDoFErrorEstimateOnFace(
    const dealii::DoFHandler<dim> &dof_dual,
    const dealii::DoFHandler<dim> &dof_pu,
    const dealii::FiniteElement<dim> &fe_dual,
    const dealii::FiniteElement<dim> &fe_pu,
    const dealii::Quadrature<dim-1> &quad,
    const dealii::UpdateFlags &uflags) :
    fe_face_values_dual(fe_dual, quad, uflags),
    fe_face_values_pu(fe_pu,quad,uflags),
    dof_dual(dof_dual),
    dof_pu(dof_pu),
    local_dof_indices_dual(fe_dual.dofs_per_cell),
    phi(fe_dual.dofs_per_cell),
    chi(fe_pu.dofs_per_cell),
    local_u_kh_m(fe_dual.dofs_per_cell),
    local_u_kh_n(fe_dual.dofs_per_cell),
    local_u_k_n(fe_dual.dofs_per_cell),
    local_u_k_np1(fe_dual.dofs_per_cell),
    local_z_kh_n(fe_dual.dofs_per_cell),
    local_z_k_m(fe_dual.dofs_per_cell),
    local_z_k_n(fe_dual.dofs_per_cell),
    val_dw_theta_j(0.),
    val_dw_Y_j(0.),
    val_pw_theta_j(0.),
    val_pw_Y_j(0.),
    val_theta_kh_j(0.),
    val_Y_kh_j(0.),
    val_z_theta_kh_j(0.),
    val_z_Y_kh_j(0.),
    value_u_N_t(0.),
    value_u_N_y(0.),
    value_u_R_t(0.),
    value_u_R_y(0.),
    JxW(0.),
    q(0),
    j(0)
{}


/// (Struct-) Copy constructor.
template<int dim>
PUDoFErrorEstimateOnFace<dim>::PUDoFErrorEstimateOnFace(const PUDoFErrorEstimateOnFace &scratch) :
    fe_face_values_dual(
        scratch.fe_face_values_dual.get_fe(),
        scratch.fe_face_values_dual.get_quadrature(),
        scratch.fe_face_values_dual.get_update_flags()),
    fe_face_values_pu(
        scratch.fe_face_values_pu.get_fe(),
        scratch.fe_face_values_pu.get_quadrature(),
        scratch.fe_face_values_pu.get_update_flags()),
    dof_dual(scratch.dof_dual),
    dof_pu(scratch.dof_pu),
    local_dof_indices_dual(scratch.local_dof_indices_dual),
    phi(scratch.phi),
    chi(scratch.chi),
    local_u_kh_m(scratch.local_u_kh_m),
    local_u_kh_n(scratch.local_u_kh_n),
    local_u_k_n(scratch.local_u_k_n),
    local_u_k_np1(scratch.local_u_k_np1),
    local_z_kh_m(scratch.local_z_kh_m),
    local_z_kh_n(scratch.local_z_kh_n),
    local_z_k_m(scratch.local_z_k_m),
    local_z_k_n(scratch.local_z_k_n),
    val_dw_theta_j(scratch.val_dw_theta_j),
    val_dw_Y_j(scratch.val_dw_Y_j),
    val_pw_theta_j(scratch.val_pw_theta_j),
    val_pw_Y_j(scratch.val_pw_Y_j),
    val_theta_kh_j(scratch.val_theta_kh_j),
    val_Y_kh_j(scratch.val_Y_kh_j),
    val_z_theta_kh_j(scratch.val_z_theta_kh_j),
    val_z_Y_kh_j(scratch.val_z_Y_kh_j),
    value_u_N_t(scratch.value_u_N_t),
    value_u_N_y(scratch.value_u_N_y),
    value_u_R_t(scratch.value_u_R_t),
    value_u_R_y(scratch.value_u_R_y),
    JxW(scratch.JxW),
    q(scratch.q),
    j(scratch.j) {
}

/// (Struct-) Constructor,
template<int dim>
PUDoFErrorEstimates<dim>::PUDoFErrorEstimates(
    const dealii::DoFHandler<dim>    &dof_dual,
    const dealii::DoFHandler<dim>    &dof_pu,
    const dealii::FiniteElement<dim> &fe_dual,
    const dealii::FiniteElement<dim> &fe_pu,
    const dealii::Quadrature<dim>    &quad_cell,
    const dealii::Quadrature<dim-1>  &quad_face,
    const dealii::UpdateFlags        &uflags_cell,
    const dealii::UpdateFlags        &uflags_face):
    cell(dof_dual, dof_pu, fe_dual, fe_pu, quad_cell, uflags_cell),
    face(dof_dual, dof_pu, fe_dual, fe_pu, quad_face, uflags_face),
    face_no(0){
}

/// (Struct-) Copy constructor.
template<int dim>
PUDoFErrorEstimates<dim>::PUDoFErrorEstimates(const PUDoFErrorEstimates &scratch) :
    cell(scratch.cell),
    face(scratch.face),
    face_no(scratch.face_no){
}

}

namespace CopyData {
/// (Struct-) constructor.
template<int dim>
PUDoFErrorEstimateOnCell<dim>::PUDoFErrorEstimateOnCell(const dealii::FiniteElement<dim> &fe) :
    local_eta_h_vector(fe.dofs_per_cell),
    local_eta_k_vector(fe.dofs_per_cell),
    local_dof_indices_pu(fe.dofs_per_cell){
}

/// (Struct-) Copy constructor.
template<int dim>
PUDoFErrorEstimateOnCell<dim>::PUDoFErrorEstimateOnCell(const PUDoFErrorEstimateOnCell &copydata) :
    local_eta_h_vector(copydata.local_eta_h_vector),
    local_eta_k_vector(copydata.local_eta_k_vector),
    local_dof_indices_pu(copydata.local_dof_indices_pu){
}


template<int dim>
PUDoFErrorEstimateOnFace<dim>::PUDoFErrorEstimateOnFace(const dealii::FiniteElement<dim> &fe) :
    local_eta_h_vector(fe.dofs_per_cell),
    local_eta_k_vector(fe.dofs_per_cell),
    local_dof_indices_pu(fe.dofs_per_cell){
}

template<int dim>
PUDoFErrorEstimateOnFace<dim>::PUDoFErrorEstimateOnFace(const PUDoFErrorEstimateOnFace &copydata) :
    local_eta_h_vector(copydata.local_eta_h_vector),
    local_eta_k_vector(copydata.local_eta_k_vector),
    local_dof_indices_pu(copydata.local_dof_indices_pu){
}
template<int dim>
PUDoFErrorEstimates<dim>::PUDoFErrorEstimates(const dealii::FiniteElement<dim> &fe) :
    cell(fe),
    face(fe) {
}
template<int dim>
PUDoFErrorEstimates<dim>::PUDoFErrorEstimates(const PUDoFErrorEstimates &copydata) :
    cell(copydata.cell),
    face(copydata.face) {
}

} // namespace CopyData

} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////

template<int dim>
void
PUDoFErrorEstimator<dim>::
set_functions(
    std::shared_ptr< dealii::Function<dim> > _u_0,
    std::shared_ptr< dealii::Function<dim> > _u_N,
    std::shared_ptr< dealii::Function<dim> > _u_R
) {
    Assert(_u_0.use_count(), dealii::ExcNotInitialized());
    function.u_0 = _u_0;

    Assert(_u_N.use_count(), dealii::ExcNotInitialized());
    function.u_N = _u_N;

    Assert(_u_R.use_count(), dealii::ExcNotInitialized());
    function.u_R = _u_R;
}


template<int dim>
void
PUDoFErrorEstimator<dim>::set_parameters(
    double alpha,
    double beta,
    double Le,
    double robin_factor_theta,
    double robin_factor_Y,
    double T,
    double domain_area,
    double rod_area,
    std::string goal_type){
    param.alpha = alpha;
    param.beta  = beta;
    param.Le    = Le;
    param.c     = beta*beta/(2*Le);
    param.robin.theta = robin_factor_theta;
    param.robin.Y     = robin_factor_Y;
    param.T  		  = T;
    param.area.domain = domain_area;
    param.area.rod    = rod_area;
    param.goal_type = goal_type;
}

template<int dim>
void
PUDoFErrorEstimator<dim>::
estimate_primal(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    std::shared_ptr< Arguments > _args,
    std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector > _eta_h,
    std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector > _eta_k
) {
    Assert(_args.use_count(), dealii::ExcNotInitialized());
    args =_args;

    ////////////////////////////////////////////////////////////////////////////
    // do estimate errors over \Omega x (tm,tn) loop
    //

    // local time variables
    tm = slab->t_m;
    t0 = tm + slab->tau_n()/2.;
    tn = slab->t_n;

    DTM::pout << "evaluating error on ( " << tm << ", " << tn << ")"
                    << std::endl;

    pu.constraints = slab->pu->constraints;
    // local tau_n (used in the local assembly functions internally)
    tau_n = slab->tau_n();

    error_estimator.x_h = _eta_h;
    error_estimator.x_k = _eta_k;

    // assemble slab problem
    dealii::QGauss<dim>   quad_cell(slab->high->fe->tensor_degree()+3);
    dealii::QGauss<dim-1> quad_face(slab->high->fe->tensor_degree()+3);

    // set time variable for boundary functions
    function.u_N->set_time(t0);
    function.u_R->set_time(t0);

    cell_number = 0;

    typedef
        dealii::FilteredIterator<const typename dealii::Triangulation<dim>::active_cell_iterator>
        CellFilter;

    dealii::WorkStream::run(
        CellFilter(
            dealii::IteratorFilters::LocallyOwnedCell(), slab->tria->begin_active()
        ),
        CellFilter(
            dealii::IteratorFilters::LocallyOwnedCell(), slab->tria->end()
        ),
        std::bind (
            &PUDoFErrorEstimator<dim>::assemble_local_primal_error,
            this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3
        ),
        std::bind (
            &PUDoFErrorEstimator<dim>::copy_local_error,
            this,
            std::placeholders::_1
        ),
        Assembly::Scratch::PUDoFErrorEstimates<dim> (
            *slab->high->dof,
            *slab->pu->dof,
            *slab->high->fe,
            *slab->pu->fe,
            quad_cell,
            quad_face,
            //
            dealii::update_values |
            dealii::update_gradients |
            dealii::update_quadrature_points |
            dealii::update_JxW_values,
            //
            dealii::update_values |
            dealii::update_quadrature_points |
            dealii::update_JxW_values
        ),
        Assembly::CopyData::PUDoFErrorEstimates<dim> (*slab->pu->fe)
    );

    error_estimator.x_h->compress(dealii::VectorOperation::add);
    error_estimator.x_k->compress(dealii::VectorOperation::add);

}


template<int dim>
void
PUDoFErrorEstimator<dim>::
estimate_dual(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    std::shared_ptr< Arguments > _args,
    std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector > _eta_h,
    std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector > _eta_k
) {
    Assert(_args.use_count(), dealii::ExcNotInitialized());
    args =_args;
    ////////////////////////////////////////////////////////////////////////////
    // do estimate errors over \Omega x (tm,tn) loop
    //


    // local time variables
    const double tm = slab->t_m;
    const double t0 = tm + slab->tau_n()/2.;
    const double tn = slab->t_n;

    DTM::pout << "evaluating adjoint error on ( " << tm << ", " << tn << ")"
              << std::endl;

    pu.constraints = slab->pu->constraints;

    // local tau_n (used in the local assembly functions internally)
    tau_n = slab->tau_n();

    error_estimator.x_h = _eta_h;
    error_estimator.x_k = _eta_k;

    // assemble slab problem
    dealii::QGauss<dim>   quad_cell(slab->high->fe->tensor_degree()+3);
    dealii::QGauss<dim-1> quad_face(slab->high->fe->tensor_degree()+3);

    // set time variable for boundary functions
    function.u_N->set_time(t0);
    function.u_R->set_time(t0);

    cell_number = 0;

    typedef
        dealii::FilteredIterator<const typename dealii::Triangulation<dim>::active_cell_iterator>
    	CellFilter;

    dealii::WorkStream::run(
        CellFilter(
            dealii::IteratorFilters::LocallyOwnedCell(), slab->tria->begin_active()
        ),
        CellFilter(
            dealii::IteratorFilters::LocallyOwnedCell(), slab->tria->end()
        ),
        std::bind (
            &PUDoFErrorEstimator<dim>::assemble_local_dual_error,
            this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3
        ),
        std::bind (
            &PUDoFErrorEstimator<dim>::copy_local_error,
            this,
            std::placeholders::_1
        ),
        Assembly::Scratch::PUDoFErrorEstimates<dim> (
            *slab->high->dof,
            *slab->pu->dof,
            *slab->high->fe,
            *slab->pu->fe,
            quad_cell,
            quad_face,
            //
            dealii::update_values |
            dealii::update_gradients |
            dealii::update_quadrature_points |
            dealii::update_JxW_values,
            //
            dealii::update_values |
            dealii::update_quadrature_points |
            dealii::update_JxW_values
        ),
        Assembly::CopyData::PUDoFErrorEstimates<dim> (*slab->pu->fe)
    );

    error_estimator.x_h->compress(dealii::VectorOperation::add);
    error_estimator.x_k->compress(dealii::VectorOperation::add);

}
////////////////////////////////////////////////////////////////////////////////
//
//
template<int dim>
void
PUDoFErrorEstimator<dim>::
assemble_local_primal_error(
    const typename dealii::Triangulation<dim>::active_cell_iterator &tria_cell,
    Assembly::Scratch::PUDoFErrorEstimates<dim> &scratch,
    Assembly::CopyData::PUDoFErrorEstimates<dim> &copydata) {

    ///////////////////////////////////////////////////////////////////////////
    // get dual and pu cell iterators:
    //

    typename dealii::DoFHandler<dim>::active_cell_iterator cell_dual(&tria_cell->get_triangulation(),
                                                                     tria_cell->level(),
                                                                     tria_cell->index(),
                                                                     &scratch.cell.dof_dual);

    typename dealii::DoFHandler<dim>::active_cell_iterator cell_pu(&tria_cell->get_triangulation(),
                                                                   tria_cell->level(),
                                                                   tria_cell->index(),
                                                                   &scratch.cell.dof_pu);
        
    ///////////////////////////////////////////////////////////////////////////
    // cell integrals:
    //
    assemble_primal_error_on_cell(cell_dual, cell_pu, scratch.cell, copydata.cell);
        
    ///////////////////////////////////////////////////////////////////////////
    // face integrals:
    //

    // initialize copydata
    copydata.face.local_eta_h_vector = 0.;
    copydata.face.local_eta_k_vector = 0.;
    cell_pu -> get_dof_indices(copydata.face.local_dof_indices_pu);

    for ( scratch.face_no = 0;
          scratch.face_no < dealii::GeometryInfo<dim>::faces_per_cell;
          ++scratch.face_no ){
        ///////////////////////////////////////////////////////////////////
        // handle boundary faces
        if ( cell_dual->face(scratch.face_no)->at_boundary()){
            if ( cell_dual->face(scratch.face_no)->boundary_id() ==
                 static_cast<dealii::types::boundary_id> (
                     combustion::types::boundary_id::Dirichlet ) ){
                // only on Dirchlet type boundary face
                continue; //nothing to do
            }
            else if ( cell_dual->face(scratch.face_no)->boundary_id() ==
                      static_cast<dealii::types::boundary_id> (
                          combustion::types::boundary_id::Neumann ) ){
                    // only on Neumann type boundary face
                    continue; // nothing to do for hom. Neumann BC
            }
            else if ( cell_dual->face(scratch.face_no)->boundary_id() ==
                      static_cast<dealii::types::boundary_id> (
                          combustion::types::boundary_id::Robin ) ){
                // only on Robin type boundary face
                assemble_primal_error_on_robin_boundary_face(
                    cell_dual,
                    cell_pu,
                    scratch.face_no,
                    scratch.face,
                    copydata.face
                );
            }

            continue;
        }
                
        ////////////////////////////////////////////////////////////////////////
        // interior faces only:
        Assert(!cell_dual->face(scratch.face_no)->at_boundary(), dealii::ExcInvalidState());

        // skip face with same refinement level where the neighbor cell index
        // is smaller than this ones
        if ((cell_dual->index() > cell_dual->neighbor(scratch.face_no)->index()) &&
            (cell_dual->neighbor(scratch.face_no)->has_children() == false) &&
            (cell_dual->level() == cell_dual->neighbor(scratch.face_no)->level())) {
            // integrate face value from the neighbor cell
            continue;
        }

        // integrate from coarser cell
        if (cell_dual->level() > cell_dual->neighbor(scratch.face_no)->level()) {
            continue;
        }

        if (cell_dual->face(scratch.face_no)->has_children() == false) {
            continue;
        }
        else {
            continue;
        }
    }
        
}


template<int dim>
void
PUDoFErrorEstimator<dim>::
assemble_local_dual_error(
    const typename dealii::Triangulation<dim>::active_cell_iterator &tria_cell,
    Assembly::Scratch::PUDoFErrorEstimates<dim> &scratch,
    Assembly::CopyData::PUDoFErrorEstimates<dim> &copydata) {

    ///////////////////////////////////////////////////////////////////////////
    // get dual and pu cell iterators:
    //

    typename dealii::DoFHandler<dim>::active_cell_iterator cell_dual(&tria_cell->get_triangulation(),
                                                                     tria_cell->level(),
                                                                     tria_cell->index(),
                                                                     &scratch.cell.dof_dual);

    typename dealii::DoFHandler<dim>::active_cell_iterator cell_pu(&tria_cell->get_triangulation(),
                                                                   tria_cell->level(),
                                                                   tria_cell->index(),
                                                                   &scratch.cell.dof_pu);

    ///////////////////////////////////////////////////////////////////////////
    // cell integrals:
    //
    assemble_dual_error_on_cell(cell_dual, cell_pu, scratch.cell, copydata.cell);

    ///////////////////////////////////////////////////////////////////////////
    // face integrals:
    //

    // initialize copydata
    copydata.face.local_eta_h_vector = 0.;
    copydata.face.local_eta_k_vector = 0.;
    cell_pu -> get_dof_indices(copydata.face.local_dof_indices_pu);

    for ( scratch.face_no = 0;
          scratch.face_no < dealii::GeometryInfo<dim>::faces_per_cell;
          ++scratch.face_no ){
        ///////////////////////////////////////////////////////////////////
        // handle boundary faces
        if ( cell_dual->face(scratch.face_no)->at_boundary()){
            if ( cell_dual->face(scratch.face_no)->boundary_id() ==
                static_cast<dealii::types::boundary_id> (
                        combustion::types::boundary_id::Dirichlet ) ){
                // only on Dirchlet type boundary face
                continue; //nothing to do
            }
            else if ( cell_dual->face(scratch.face_no)->boundary_id() ==
                      static_cast<dealii::types::boundary_id> (
                        combustion::types::boundary_id::Neumann ) ){
                // only on Neumann type boundary face
                continue; // nothing to do for hom. Neumann BC
            }
            else if ( cell_dual->face(scratch.face_no)->boundary_id() ==
                      static_cast<dealii::types::boundary_id> (
                         combustion::types::boundary_id::Robin ) ){
                // only on Robin type boundary face
                assemble_dual_error_on_robin_boundary_face(
                    cell_dual,
                    cell_pu,
                    scratch.face_no,
                    scratch.face,
                    copydata.face
                );
            }

            continue;
        }

        ////////////////////////////////////////////////////////////////////////
        // interior faces only:
        Assert(!cell_dual->face(scratch.face_no)->at_boundary(), dealii::ExcInvalidState());

        // skip face with same refinement level where the neighbor cell index
        // is smaller than this ones
        if ((cell_dual->index() > cell_dual->neighbor(scratch.face_no)->index()) &&
            (cell_dual->neighbor(scratch.face_no)->has_children() == false) &&
            (cell_dual->level() == cell_dual->neighbor(scratch.face_no)->level())) {
            // integrate face value from the neighbor cell
            continue;
        }

        // integrate from coarser cell
        if (cell_dual->level() > cell_dual->neighbor(scratch.face_no)->level()) {
            continue;
        }

        if (cell_dual->face(scratch.face_no)->has_children() == false) {
            continue;
        }
        else {
            continue;
        }
    }

}

template<int dim>
void
PUDoFErrorEstimator<dim>::
assemble_primal_error_on_cell(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell_dual,
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell_pu,
    Assembly::Scratch::PUDoFErrorEstimateOnCell<dim> &scratch,
    Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> &copydata) {
        
        
    // reinit scratch and data to current cell
    scratch.fe_values_dual.reinit(cell_dual);
    scratch.fe_values_pu.reinit(cell_pu);
    // fetch local dof data
    cell_dual->get_dof_indices(scratch.local_dof_indices_dual);


    for (scratch.j=0; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell;
         ++scratch.j) {
        scratch.local_u_kh_n[scratch.j] =
        (*args->tn.u_kh)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_z_kh_n[scratch.j] =
        (*args->tn.z_kh)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_z_k_n[scratch.j] =
        (*args->tn.z_k)[ scratch.local_dof_indices_dual[scratch.j] ];
    }

    for (scratch.j=0; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell;
        ++scratch.j) {
        scratch.local_u_kh_m[scratch.j] =
        (*args->tm.u_kh)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_z_k_m[scratch.j] =
        (*args->tm.z_k)[ scratch.local_dof_indices_dual[scratch.j] ];

    }
        
    // initialize copydata
    copydata.local_eta_h_vector = 0.;
    copydata.local_eta_k_vector = 0.;
    cell_pu -> get_dof_indices(copydata.local_dof_indices_pu);
        
        
    //assemble PU
    for (scratch.q = 0; scratch.q < scratch.fe_values_pu.n_quadrature_points; ++scratch.q)
    {
        scratch.JxW = scratch.fe_values_pu.JxW(scratch.q);

        //shape values for dual basis
        for (scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell;
             ++scratch.j)
        {
            scratch.phi[scratch.j] =
                scratch.fe_values_dual.shape_value(scratch.j,scratch.q);

            scratch.grad_phi[scratch.j] =
                scratch.fe_values_dual.shape_grad(scratch.j,scratch.q);
        }
        //shape values for spatial partition of Unity
        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell;
              ++scratch.j )
        {
            scratch.chi[scratch.j] =
                scratch.fe_values_pu.shape_value(scratch.j,scratch.q);

            scratch.grad_chi[scratch.j] =
                scratch.fe_values_pu.shape_grad(scratch.j,scratch.q);
        }
          
        //everything except for dual weights is same for both error parts
        scratch.val_theta_kh_j  = 0.;
        scratch.grad_theta_kh_j = 0.;
        scratch.val_theta_kh_jump_j = 0.;

        scratch.val_Y_kh_j = 0.;
        scratch.grad_Y_kh_j = 0.;
        scratch.val_Y_kh_jump_j = 0.;

              
        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell ;
            ++scratch.j )
        {
            //component 0 theta
            if (scratch.fe_values_dual.get_fe().
                system_to_component_index(scratch.j).first == 0 )
            {

                scratch.val_theta_kh_j +=
                    scratch.local_u_kh_n[scratch.j] * scratch.phi[scratch.j];


                scratch.grad_theta_kh_j +=
                    scratch.local_u_kh_n[scratch.j] * scratch.grad_phi[scratch.j];


                scratch.val_theta_kh_jump_j +=
                    (scratch.local_u_kh_n[scratch.j] - scratch.local_u_kh_m[scratch.j])
                        * scratch.phi[scratch.j];

            }
            //component 1 Y
            else {
                scratch.val_Y_kh_j +=
                    scratch.local_u_kh_n[scratch.j] * scratch.phi[scratch.j];

                scratch.grad_Y_kh_j +=
                    scratch.local_u_kh_n[scratch.j] * scratch.grad_phi[scratch.j];

                scratch.val_Y_kh_jump_j +=
                    (scratch.local_u_kh_n[scratch.j] - scratch.local_u_kh_m[scratch.j])
                    * scratch.phi[scratch.j];

            }
        }

        scratch.omega = param.c*scratch.val_Y_kh_j*
            std::exp(
                (param.beta*(scratch.val_theta_kh_j-1.0))
                /
                (1+param.alpha*(scratch.val_theta_kh_j-1.0))
            );

        //calculating PU indicators for temporal error over I_n
        scratch.val_dw_theta_j = 0.;
        scratch.grad_dw_theta_j = 0.;

        scratch.val_dw_Y_j = 0.;
        scratch.grad_dw_Y_j = 0.;

        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell ;
            ++scratch.j )
        {
            //component 0 theta
            if (scratch.fe_values_dual.get_fe().
                system_to_component_index(scratch.j).first == 0 )
            {

                scratch.val_dw_theta_j +=
                    (scratch.local_z_k_m[scratch.j] - scratch.local_z_k_n[scratch.j] )
                    * scratch.phi[scratch.j];

                scratch.grad_dw_theta_j +=
                    (scratch.local_z_k_m[scratch.j] - scratch.local_z_k_n[scratch.j])
                    * scratch.grad_phi[scratch.j];

            }
            //component 1 Y
            else {

                scratch.val_dw_Y_j +=
                    (scratch.local_z_k_m[scratch.j] - scratch.local_z_k_n[scratch.j] )
                    * scratch.phi[scratch.j];

                scratch.grad_dw_Y_j +=
                    (scratch.local_z_k_m[scratch.j] - scratch.local_z_k_n[scratch.j])
                    * scratch.grad_phi[scratch.j];

            }
        }
              
            for ( scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell;
                ++ scratch.j )
            {
                // RHS/Arrhenius Function
                copydata.local_eta_k_vector[scratch.j] -=
                    0.5 *
                    //omega(theta,Y)
                    scratch.omega*
                    //zRz_theta
                    (scratch.val_dw_Y_j - scratch.val_dw_theta_j) *
                    //chi * JxW
                    scratch.chi[scratch.j] *
                    scratch.JxW*tau_n;

                //Lagrange Terms
                //theta
                copydata.local_eta_k_vector[scratch.j] -=
                    0.5 *
                    scratch.grad_theta_kh_j *
                        (scratch.grad_dw_theta_j*scratch.chi[scratch.j]
                        +scratch.val_dw_theta_j*scratch.grad_chi[scratch.j]
                        ) * scratch.JxW * tau_n;

                //Y
                copydata.local_eta_k_vector[scratch.j] -=
                    0.5 *
                    scratch.grad_Y_kh_j/param.Le *
                        (scratch.grad_dw_Y_j*scratch.chi[scratch.j]
                        +scratch.val_dw_Y_j*scratch.grad_chi[scratch.j]
                        )* scratch.JxW*tau_n;

                //Jump Terms
                copydata.local_eta_k_vector[scratch.j]  -=
                    scratch.val_theta_kh_jump_j * scratch.val_dw_theta_j * scratch.chi[scratch.j]
                    * scratch.JxW;

                copydata.local_eta_k_vector[scratch.j]  -=
                    scratch.val_Y_kh_jump_j * scratch.val_dw_Y_j * scratch.chi[scratch.j]
                    * scratch.JxW;
            }
                
            //calculating PU indicators for spatial error over I_n
            scratch.val_dw_theta_j = 0.;
            scratch.grad_dw_theta_j = 0.;

            scratch.val_dw_Y_j = 0.;
            scratch.grad_dw_Y_j = 0.;

            for ( scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell ;
                ++scratch.j )
            {
                //component 0 theta
                if (scratch.fe_values_dual.get_fe().
                                system_to_component_index(scratch.j).first == 0 )
                {
                    scratch.val_dw_theta_j +=
                        (scratch.local_z_k_n[scratch.j] - scratch.local_z_kh_n[scratch.j] )
                        * scratch.phi[scratch.j];

                    scratch.grad_dw_theta_j +=
                        (scratch.local_z_k_n[scratch.j] - scratch.local_z_kh_n[scratch.j])
                        * scratch.grad_phi[scratch.j];
                }
                //component 1 Y
                else {
                    scratch.val_dw_Y_j +=
                        (scratch.local_z_k_n[scratch.j] - scratch.local_z_kh_n[scratch.j] )
                        * scratch.phi[scratch.j];

                    scratch.grad_dw_Y_j +=
                        (scratch.local_z_k_n[scratch.j] - scratch.local_z_kh_n[scratch.j] )
                        * scratch.grad_phi[scratch.j];
                }
            }

            for ( scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell;
                                    ++ scratch.j )
            {
                // RHS/Arrhenius Function
                copydata.local_eta_h_vector[scratch.j] -=
                    //omega(theta,Y)
                    scratch.omega*
                    //zRz_theta
                    (scratch.val_dw_Y_j - scratch.val_dw_theta_j) *
                    //chi * JxW
                    scratch.chi[scratch.j] *
                    scratch.JxW*tau_n;

                //Lagrange Terms
                //theta
                copydata.local_eta_h_vector[scratch.j] -=
                    scratch.grad_theta_kh_j *
                    (scratch.grad_dw_theta_j*scratch.chi[scratch.j]
                         +scratch.val_dw_theta_j*scratch.grad_chi[scratch.j]
                    ) * scratch.JxW * tau_n;

                //Y
                copydata.local_eta_h_vector[scratch.j] -=
                    scratch.grad_Y_kh_j/param.Le *
                    (scratch.grad_dw_Y_j*scratch.chi[scratch.j]
                         +scratch.val_dw_Y_j*scratch.grad_chi[scratch.j]
                    )* scratch.JxW*tau_n;

                //Jump Terms
                copydata.local_eta_h_vector[scratch.j] -=
                    scratch.val_theta_kh_jump_j * scratch.val_dw_theta_j * scratch.chi[scratch.j]
                                                                                                                                                                                   * scratch.JxW;

                copydata.local_eta_h_vector[scratch.j] -=
                    scratch.val_Y_kh_jump_j * scratch.val_dw_Y_j * scratch.chi[scratch.j]
                                                                                                                                                                       * scratch.JxW;
            }
        }

}



template<int dim>
void
PUDoFErrorEstimator<dim>::
assemble_dual_error_on_cell(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell_dual,
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell_pu,
    Assembly::Scratch::PUDoFErrorEstimateOnCell<dim> &scratch,
    Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> &copydata) {


    // reinit scratch and data to current cell
    scratch.fe_values_dual.reinit(cell_dual);
    scratch.fe_values_pu.reinit(cell_pu);
    // fetch local dof data
    cell_dual->get_dof_indices(scratch.local_dof_indices_dual);


    for (scratch.j=0; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell;
        ++scratch.j) {
        scratch.local_u_kh_n[scratch.j] =
            (*args->tn.u_kh)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_u_k_n[scratch.j] =
            (*args->tn.u_k)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_z_kh_n[scratch.j] =
            (*args->tn.z_kh)[ scratch.local_dof_indices_dual[scratch.j] ];
    }

    for (scratch.j=0; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell;
        ++scratch.j) {
        scratch.local_u_k_np1[scratch.j] =
            (*args->tnp1.u_k)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_z_kh_np1[scratch.j] =
            (*args->tnp1.z_kh)[ scratch.local_dof_indices_dual[scratch.j] ];

    }

    // initialize copydata
    copydata.local_eta_h_vector = 0.;
    copydata.local_eta_k_vector = 0.;
    cell_pu -> get_dof_indices(copydata.local_dof_indices_pu);


    //assemble PU
    for (scratch.q = 0; scratch.q < scratch.fe_values_pu.n_quadrature_points; ++scratch.q)
    {
        scratch.JxW = scratch.fe_values_pu.JxW(scratch.q);

        //shape values for dual basis
        for (scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell;
             ++scratch.j)
        {
            scratch.phi[scratch.j] =
                scratch.fe_values_dual.shape_value(scratch.j,scratch.q);

            scratch.grad_phi[scratch.j] =
                scratch.fe_values_dual.shape_grad(scratch.j,scratch.q);
        }
        //shape values for spatial partition of Unity
        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell;
          ++scratch.j )
        {
            scratch.chi[scratch.j] =
                scratch.fe_values_pu.shape_value(scratch.j,scratch.q);

            scratch.grad_chi[scratch.j] =
                scratch.fe_values_pu.shape_grad(scratch.j,scratch.q);
        }


        //everything except for primal weights is same for both error parts
        scratch.val_theta_kh_j = 0.;
        scratch.val_z_theta_kh_j = 0.;
        scratch.grad_z_theta_kh_j = 0.;
        scratch.val_z_theta_kh_jump_j = 0.;

        scratch.val_Y_kh_j = 0.;
        scratch.val_z_Y_kh_j = 0.;
        scratch.grad_z_Y_kh_j = 0.;
        scratch.val_z_Y_kh_jump_j = 0.;

        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell ;
                        ++ scratch.j )
        {
            //component 0 theta
            if (scratch.fe_values_dual.get_fe().
                system_to_component_index(scratch.j).first == 0 )
            {
                scratch.val_theta_kh_j +=
                    scratch.local_u_kh_n[scratch.j] * scratch.phi[scratch.j];

                scratch.val_z_theta_kh_j +=
                    scratch.local_z_kh_n[scratch.j] * scratch.phi[scratch.j];


                scratch.grad_z_theta_kh_j +=
                    scratch.local_z_kh_n[scratch.j] * scratch.grad_phi[scratch.j];


                scratch.val_z_theta_kh_jump_j +=
                    (scratch.local_z_kh_n[scratch.j] - scratch.local_z_kh_np1[scratch.j])
                        * scratch.phi[scratch.j];

            }
            //component 1 Y
            else {
                scratch.val_Y_kh_j +=
                    scratch.local_u_kh_n[scratch.j] * scratch.phi[scratch.j];

                scratch.val_z_Y_kh_j +=
                    scratch.local_z_kh_n[scratch.j] * scratch.phi[scratch.j];


                scratch.grad_z_Y_kh_j +=
                    scratch.local_z_kh_n[scratch.j] * scratch.grad_phi[scratch.j];


                scratch.val_z_Y_kh_jump_j +=
                    (scratch.local_z_kh_n[scratch.j] - scratch.local_z_kh_np1[scratch.j])
                        * scratch.phi[scratch.j];
            }

        }

        scratch.omega_y = param.c*std::exp(
            (param.beta*(scratch.val_theta_kh_j-1.0))
            /
            (1+param.alpha*(scratch.val_theta_kh_j-1.0))
        );

        double denom = (1+param.alpha*(scratch.val_theta_kh_j-1.0));

        scratch.omega_theta = param.beta/
            (denom*denom)
            *scratch.val_Y_kh_j*scratch.omega_y;

        //Calculating PU indicators for temporal error over I_n
        scratch.val_pw_theta_j = 0.;
        scratch.grad_pw_theta_j = 0.;

        scratch.val_pw_Y_j = 0.;
        scratch.grad_pw_Y_j = 0.;


        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell ;
            ++scratch.j )
        {
            //component 0 theta
            if (scratch.fe_values_dual.get_fe().
                system_to_component_index(scratch.j).first == 0 )
            {
                scratch.val_pw_theta_j +=
                    (scratch.local_u_k_np1[scratch.j]-scratch.local_u_k_n[scratch.j]) *
                     scratch.phi[scratch.j];


                scratch.grad_pw_theta_j +=
                    (scratch.local_u_k_np1[scratch.j]-scratch.local_u_k_n[scratch.j]) *
                    scratch.grad_phi[scratch.j];

            }
            //component 1 Y
                else {
                    scratch.val_pw_Y_j +=
                    (scratch.local_u_k_np1[scratch.j]-scratch.local_u_k_n[scratch.j]) *
                    scratch.phi[scratch.j];


                    scratch.grad_pw_Y_j +=
                    (scratch.local_u_k_np1[scratch.j]-scratch.local_u_k_n[scratch.j]) *
                    scratch.grad_phi[scratch.j];

            }
        }
        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell;
            ++ scratch.j )
        {

            if ( param.goal_type.compare("reaction rate")==0 ){
                copydata.local_eta_k_vector[scratch.j] += 0.5*
                    1./(param.T*param.area.domain) *
                    ( scratch.omega_theta*scratch.val_pw_theta_j
                     +scratch.omega_y*scratch.val_pw_Y_j )
                    *scratch.chi[scratch.j]
                    *scratch.JxW*tau_n;
            }
            //Temporal derivatives = 0 as z_kh in dG(0)

            //Lagrange Terms
            //theta
            copydata.local_eta_k_vector[scratch.j] -=
                0.5 *
                scratch.grad_z_theta_kh_j *
                    (scratch.grad_pw_theta_j*scratch.chi[scratch.j]
                    +scratch.val_pw_theta_j*scratch.grad_chi[scratch.j]
                ) * scratch.JxW * tau_n;

            //Y
            copydata.local_eta_k_vector[scratch.j] -=
                0.5 *
                scratch.grad_z_Y_kh_j/param.Le *
                    (scratch.grad_pw_Y_j*scratch.chi[scratch.j]
                     +scratch.val_pw_Y_j*scratch.grad_chi[scratch.j]
                )* scratch.JxW*tau_n;

            //Arrhenius derivatives
            //w.r.t theta
            copydata.local_eta_k_vector[scratch.j] +=
                0.5*
                scratch.omega_theta*scratch.val_pw_theta_j*scratch.chi[scratch.j]*
                (scratch.val_z_theta_kh_j-scratch.val_z_Y_kh_j)
                *scratch.JxW*tau_n;
            //w.r.t. Y
            copydata.local_eta_k_vector[scratch.j] +=
                0.5*
                scratch.omega_y*scratch.val_pw_Y_j*scratch.chi[scratch.j]*
                (scratch.val_z_theta_kh_j-scratch.val_z_Y_kh_j)
                *scratch.JxW*tau_n;

            copydata.local_eta_k_vector[scratch.j] -=
                scratch.val_z_theta_kh_jump_j * scratch.val_pw_theta_j * scratch.chi[scratch.j] * scratch.JxW;

            copydata.local_eta_k_vector[scratch.j] -=
                scratch.val_z_Y_kh_jump_j * scratch.val_pw_Y_j * scratch.chi[scratch.j] * scratch.JxW;
        }

        //Calculating PU indicators for spatial error over I_n
        scratch.val_pw_theta_j = 0.;
        scratch.grad_pw_theta_j = 0.;

        scratch.val_pw_Y_j = 0.;
        scratch.grad_pw_Y_j = 0.;

        for ( scratch.j = 0 ; scratch.j < scratch.fe_values_dual.get_fe().dofs_per_cell ;
            ++scratch.j )
        {
            //component 0 theta
            if (scratch.fe_values_dual.get_fe().
                system_to_component_index(scratch.j).first == 0 )
            {

                scratch.val_pw_theta_j +=
                    (scratch.local_u_k_n[scratch.j]-scratch.local_u_kh_n[scratch.j]) *
                    scratch.phi[scratch.j];


                scratch.grad_pw_theta_j +=
                    (scratch.local_u_k_n[scratch.j]-scratch.local_u_kh_n[scratch.j]) *
                    scratch.grad_phi[scratch.j];

            }
            //component 1 Y
            else {
                scratch.val_pw_Y_j +=
                    (scratch.local_u_k_n[scratch.j]-scratch.local_u_kh_n[scratch.j]) *
                    scratch.phi[scratch.j];


                scratch.grad_pw_Y_j +=
                    (scratch.local_u_k_n[scratch.j]-scratch.local_u_kh_n[scratch.j]) *
                    scratch.grad_phi[scratch.j];

            }
        }


            for ( scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell;
                            ++ scratch.j )
            {

                if ( param.goal_type.compare("reaction rate")==0 ){
                    copydata.local_eta_h_vector[scratch.j] +=
                        1./(param.T*param.area.domain) *
                            ( scratch.omega_theta*scratch.val_pw_theta_j
                                +scratch.omega_y*scratch.val_pw_Y_j )
                            *scratch.chi[scratch.j]
                            *scratch.JxW*tau_n;
                }
                //Temporal derivatives = 0 as z_kh in dG(0)

                //Lagrange Terms
                //theta
                copydata.local_eta_h_vector[scratch.j] -=
                    scratch.grad_z_theta_kh_j *
                    (scratch.grad_pw_theta_j*scratch.chi[scratch.j]
                         +scratch.val_pw_theta_j*scratch.grad_chi[scratch.j]
                    ) * scratch.JxW * tau_n;

                //Y
                copydata.local_eta_h_vector[scratch.j] -=
                    scratch.grad_z_Y_kh_j/param.Le *
                    (scratch.grad_pw_Y_j*scratch.chi[scratch.j]
                         +scratch.val_pw_Y_j*scratch.grad_chi[scratch.j]
                    )* scratch.JxW*tau_n;

                //Arrhenius derivatives
                //w.r.t theta
                copydata.local_eta_h_vector[scratch.j] +=
                    scratch.omega_theta*
                    scratch.val_pw_theta_j*scratch.chi[scratch.j]*
                    (scratch.val_z_theta_kh_j-scratch.val_z_Y_kh_j)
                    *scratch.JxW*tau_n;
                //w.r.t. Y
                copydata.local_eta_h_vector[scratch.j] +=
                    scratch.omega_y*
                    scratch.val_pw_Y_j*scratch.chi[scratch.j]*
                    (scratch.val_z_theta_kh_j-scratch.val_z_Y_kh_j)
                    *scratch.JxW*tau_n;

                //Jump Terms
                copydata.local_eta_h_vector[scratch.j] -=
                    scratch.val_pw_theta_j*scratch.chi[scratch.j] *
                    scratch.val_z_theta_kh_jump_j*scratch.JxW;

                copydata.local_eta_h_vector[scratch.j] -=
                    scratch.val_pw_Y_j*scratch.chi[scratch.j] *
                    scratch.val_z_Y_kh_jump_j*scratch.JxW;

            }

    }

}

template<int dim>
void
PUDoFErrorEstimator<dim>::
assemble_primal_error_on_robin_boundary_face(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell_dual,
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell_pu,
    const unsigned int face_no,
    Assembly::Scratch::PUDoFErrorEstimateOnFace<dim>  &scratch,
    Assembly::CopyData::PUDoFErrorEstimateOnFace<dim> &copydata ) {

    Assert(
           (cell_dual->face(face_no).state() == dealii::IteratorState::valid),
           dealii::ExcInternalError()
    );
    Assert(
           (cell_pu->face(face_no).state() == dealii::IteratorState::valid),
           dealii::ExcInternalError()
    );

    scratch.fe_face_values_dual.reinit(cell_dual, face_no);
    scratch.fe_face_values_pu.reinit(cell_pu, face_no);

    cell_dual->get_dof_indices(scratch.local_dof_indices_dual);

    for (scratch.j=0; scratch.j < scratch.fe_face_values_dual.get_fe().dofs_per_cell;
        ++scratch.j) {
        scratch.local_u_kh_n[scratch.j] =
            (*args->tn.u_kh)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_z_kh_n[scratch.j] =
            (*args->tn.z_kh)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_z_k_n[scratch.j] =
            (*args->tn.z_k)[ scratch.local_dof_indices_dual[scratch.j] ];
    }

    for (scratch.j=0; scratch.j < scratch.fe_face_values_dual.get_fe().dofs_per_cell;
        ++scratch.j) {

        scratch.local_z_k_m[scratch.j] =
            (*args->tm.z_k)[ scratch.local_dof_indices_dual[scratch.j] ];

    }

    // initialize copydata
    copydata.local_eta_h_vector = 0.;
    copydata.local_eta_k_vector = 0.;
    cell_pu -> get_dof_indices(copydata.local_dof_indices_pu);


    // assemble face terms
    for ( scratch.q = 0 ; scratch.q < scratch.fe_face_values_pu.n_quadrature_points; ++ scratch.q)
    {
        scratch.JxW = scratch.fe_face_values_pu.JxW(scratch.q);

        // shape values for dual basis
        for ( scratch.j = 0 ; scratch.j < scratch.fe_face_values_dual.get_fe().dofs_per_cell;
          ++scratch.j )
        {
            scratch.phi[scratch.j] =
                scratch.fe_face_values_dual.shape_value(scratch.j,scratch.q);

        }
        // shape values for partition of Unity
        for ( scratch.j = 0 ; scratch.j < scratch.fe_face_values_pu.get_fe().dofs_per_cell;
          ++scratch.j )
        {
            scratch.chi[scratch.j] =
                scratch.fe_face_values_pu.shape_value(scratch.j,scratch.q);
        }

        //everything except for dual weights is same for both error parts
        scratch.val_theta_kh_j  = 0.;

        scratch.val_Y_kh_j = 0.;

        for ( scratch.j = 0 ; scratch.j < scratch.fe_face_values_dual.get_fe().dofs_per_cell;
                  ++scratch.j )
        {
            //component 0 theta
            if ( scratch.fe_face_values_dual.get_fe().
                system_to_component_index(scratch.j).first == 0)
            {
            scratch.val_theta_kh_j +=
                scratch.local_u_kh_n[scratch.j] * scratch.phi[scratch.j];
            }
            //component 1 Y
            else
            {
                scratch.val_Y_kh_j +=
                    scratch.local_u_kh_n[scratch.j] * scratch.phi[scratch.j];
            }
        }



        //calculating PU indicators for temporal error over I_n
        scratch.val_dw_theta_j = 0.;

        scratch.val_dw_Y_j = 0.;

        for ( scratch.j = 0 ; scratch.j < scratch.fe_face_values_dual.get_fe().dofs_per_cell ;
            ++scratch.j )
        {
            //component 0 theta
            if (scratch.fe_face_values_dual.get_fe().
                system_to_component_index(scratch.j).first == 0 )
            {

                scratch.val_dw_theta_j +=
                    (scratch.local_z_k_m[scratch.j] - scratch.local_z_k_n[scratch.j] )
                    * scratch.phi[scratch.j];
            }
            //component 1 Y
            else {
                scratch.val_dw_Y_j +=
                    (scratch.local_z_k_m[scratch.j] - scratch.local_z_k_n[scratch.j] )
                    * scratch.phi[scratch.j];
            }
        }

        for ( scratch.j = 0 ; scratch.j < scratch.fe_face_values_pu.get_fe().dofs_per_cell;
                ++ scratch.j )
        {
            //lhs part
            copydata.local_eta_k_vector[scratch.j] -=
                tau_n * 0.5 *
                param.robin.theta * scratch.val_theta_kh_j *
                scratch.val_dw_theta_j *scratch.chi[scratch.j]
                * scratch.JxW;

            copydata.local_eta_k_vector[scratch.j] -=
                tau_n * 0.5 *
                param.robin.Y     * scratch.val_Y_kh_j *
                scratch.val_dw_Y_j * scratch.chi[scratch.j] * scratch.JxW;
        }

        //calculating PU indicators for spatial error over I_n
        scratch.val_dw_theta_j = 0.;

        scratch.val_dw_Y_j = 0.;

        for ( scratch.j = 0 ; scratch.j < scratch.fe_face_values_dual.get_fe().dofs_per_cell;
            ++scratch.j )
        {
            //component 0 theta
            if ( scratch.fe_face_values_dual.get_fe().
                system_to_component_index(scratch.j).first == 0)
            {
                scratch.val_dw_theta_j +=
                    (scratch.local_z_k_n[scratch.j] - scratch.local_z_kh_n[scratch.j] )
                    * scratch.phi[scratch.j];
            }
            //component 1 Y
            else
            {
                scratch.val_dw_Y_j +=
                    (scratch.local_z_k_n[scratch.j] - scratch.local_z_kh_n[scratch.j] )
                    * scratch.phi[scratch.j];
            }
        }


        for ( scratch.j = 0 ; scratch.j < scratch.fe_face_values_pu.get_fe().dofs_per_cell;
                        ++ scratch.j )
        {
            //lhs part
            copydata.local_eta_h_vector[scratch.j] -=
                 tau_n *
                 param.robin.theta * scratch.val_theta_kh_j *
                 scratch.val_dw_theta_j*scratch.chi[scratch.j] * scratch.JxW;

            copydata.local_eta_h_vector[scratch.j] -=
                 tau_n *
                 param.robin.Y     * scratch.val_Y_kh_j *
                 scratch.val_dw_Y_j*scratch.chi[scratch.j] * scratch.JxW;
        }
    }
        
}




template<int dim>
void
PUDoFErrorEstimator<dim>::
assemble_dual_error_on_robin_boundary_face(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell_dual,
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell_pu,
    const unsigned int face_no,
    Assembly::Scratch::PUDoFErrorEstimateOnFace<dim>  &scratch,
    Assembly::CopyData::PUDoFErrorEstimateOnFace<dim> &copydata ) {

    Assert(
        (cell_dual->face(face_no).state() == dealii::IteratorState::valid),
        dealii::ExcInternalError()
    );
    Assert(
        (cell_pu->face(face_no).state() == dealii::IteratorState::valid),
        dealii::ExcInternalError()
    );


    scratch.fe_face_values_dual.reinit(cell_dual, face_no);
    scratch.fe_face_values_pu.reinit(cell_pu, face_no);

    cell_dual->get_dof_indices(scratch.local_dof_indices_dual);
    for (scratch.j=0; scratch.j < scratch.fe_face_values_dual.get_fe().dofs_per_cell;
       ++scratch.j) {
        scratch.local_u_kh_n[scratch.j] =
            (*args->tn.u_kh)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_u_k_n[scratch.j] =
            (*args->tn.u_k)[ scratch.local_dof_indices_dual[scratch.j] ];

        scratch.local_z_kh_n[scratch.j] =
            (*args->tn.z_kh)[ scratch.local_dof_indices_dual[scratch.j] ];
    }

    for (scratch.j=0; scratch.j < scratch.fe_face_values_dual.get_fe().dofs_per_cell;
       ++scratch.j) {
        scratch.local_u_k_np1[scratch.j] =
            (*args->tnp1.u_k)[ scratch.local_dof_indices_dual[scratch.j] ];

    }

    // initialize copydata
    copydata.local_eta_h_vector = 0.;
    copydata.local_eta_k_vector = 0.;
    cell_pu -> get_dof_indices(copydata.local_dof_indices_pu);

    // assemble face terms
    for ( scratch.q = 0 ; scratch.q < scratch.fe_face_values_pu.n_quadrature_points; ++ scratch.q)
    {
        scratch.JxW = scratch.fe_face_values_pu.JxW(scratch.q);

        // shape values for dual basis
        for ( scratch.j = 0 ; scratch.j < scratch.fe_face_values_dual.get_fe().dofs_per_cell;
            ++scratch.j )
        {
            scratch.phi[scratch.j] =
                scratch.fe_face_values_dual.shape_value(scratch.j,scratch.q);

        }
        // shape values for partition of Unity
        for ( scratch.j = 0 ; scratch.j < scratch.fe_face_values_pu.get_fe().dofs_per_cell;
            ++scratch.j )
        {
            scratch.chi[scratch.j] =
                scratch.fe_face_values_pu.shape_value(scratch.j,scratch.q);
        }

        //everything except for primal weights is same for both error parts
        scratch.val_z_theta_kh_j = 0.;

        scratch.val_z_Y_kh_j = 0.;

        for ( scratch.j = 0 ; scratch.j < scratch.fe_face_values_dual.get_fe().dofs_per_cell ;
            ++ scratch.j )
        {
            //component 0 theta
            if (scratch.fe_face_values_dual.get_fe().
                system_to_component_index(scratch.j).first == 0 )
            {
                scratch.val_z_theta_kh_j +=
                    scratch.local_z_kh_n[scratch.j] * scratch.phi[scratch.j];
            }
            //component 1 Y
            else {
                scratch.val_z_Y_kh_j +=
                    scratch.local_z_kh_n[scratch.j] * scratch.phi[scratch.j];
            }

        }

        //Calculating PU indicators for temporal error over I_n
        scratch.val_pw_theta_j = 0.;

        scratch.val_pw_Y_j = 0.;


        for ( scratch.j = 0 ; scratch.j < scratch.fe_face_values_dual.get_fe().dofs_per_cell;
            ++scratch.j )
        {
            //component 0 theta
            if ( scratch.fe_face_values_dual.get_fe().
                system_to_component_index(scratch.j).first == 0)
            {
                scratch.val_pw_theta_j +=
                    (scratch.local_u_k_np1[scratch.j]-scratch.local_u_k_n[scratch.j]) *
                    scratch.phi[scratch.j];
            }
            //component 1 Y
            else
            {
                scratch.val_pw_Y_j +=
                    (scratch.local_u_k_np1[scratch.j]-scratch.local_u_k_n[scratch.j]) *
                    scratch.phi[scratch.j];
            }
        }


        for ( scratch.j = 0 ; scratch.j < scratch.fe_face_values_pu.get_fe().dofs_per_cell;
                        ++ scratch.j )
        {
            if ( param.goal_type.compare("rod species concentration")==0 ){
                copydata.local_eta_k_vector[scratch.j] +=
                    tau_n * 0.5*
                    1./(param.T*param.area.rod) *
                    scratch.val_pw_Y_j *scratch.chi[scratch.j]*scratch.JxW;

            }
            //lhs part
            copydata.local_eta_k_vector[scratch.j] -=
                tau_n * 0.5 *
                param.robin.theta * scratch.val_z_theta_kh_j *
                scratch.val_pw_theta_j *scratch.chi[scratch.j]* scratch.JxW;

            copydata.local_eta_k_vector[scratch.j] -=
                tau_n * 0.5 *
                param.robin.Y     * scratch.val_z_Y_kh_j *
                scratch.val_pw_Y_j* scratch.chi[scratch.j] * scratch.JxW;
        }

        //Calculating PU indicators for spatial error over I_n
        scratch.val_pw_theta_j = 0.;

        scratch.val_pw_Y_j = 0.;

        for ( scratch.j = 0 ; scratch.j < scratch.fe_face_values_dual.get_fe().dofs_per_cell;
            ++scratch.j )
    {
            //component 0 theta
            if ( scratch.fe_face_values_dual.get_fe().
                 system_to_component_index(scratch.j).first == 0)
            {
                scratch.val_pw_theta_j +=
                    (scratch.local_u_k_n[scratch.j]-scratch.local_u_kh_n[scratch.j]) *
                    scratch.phi[scratch.j];

            }
            //component 1 Y
            else
            {
                scratch.val_pw_Y_j +=
                    (scratch.local_u_k_n[scratch.j]-scratch.local_u_kh_n[scratch.j]) *
                    scratch.phi[scratch.j];
            }
        }


        for ( scratch.j = 0 ; scratch.j < scratch.fe_face_values_pu.get_fe().dofs_per_cell;
                        ++ scratch.j )
        {
            if ( param.goal_type.compare("rod species concentration")==0 ){
                copydata.local_eta_h_vector[scratch.j] +=
                    tau_n *
                    1./(param.T*param.area.rod) *
                    scratch.val_pw_Y_j *scratch.chi[scratch.j]*scratch.JxW;

            }
            //lhs part
            copydata.local_eta_h_vector[scratch.j] -=
                tau_n *
                param.robin.theta * scratch.val_z_theta_kh_j *
                scratch.val_pw_theta_j*scratch.chi[scratch.j] * scratch.JxW;

            copydata.local_eta_h_vector[scratch.j] -=
                tau_n *
                param.robin.Y     * scratch.val_z_Y_kh_j *
                scratch.val_pw_Y_j*scratch.chi[scratch.j] * scratch.JxW;
        }

    }

}


template<int dim>
void
PUDoFErrorEstimator<dim>::copy_local_error(
    const Assembly::CopyData::PUDoFErrorEstimates<dim> & copydata) {
        pu.constraints->distribute_local_to_global(
            copydata.cell.local_eta_h_vector,
            copydata.cell.local_dof_indices_pu,
            *error_estimator.x_h
        );

        pu.constraints->distribute_local_to_global(
            copydata.cell.local_eta_k_vector,
            copydata.cell.local_dof_indices_pu,
            *error_estimator.x_k
        );

        pu.constraints->distribute_local_to_global(
            copydata.face.local_eta_h_vector,
            copydata.face.local_dof_indices_pu,
            *error_estimator.x_h
        );

        pu.constraints->distribute_local_to_global(
            copydata.face.local_eta_k_vector,
            copydata.face.local_dof_indices_pu,
            *error_estimator.x_k
        );

}

}}} // namespace

#include "GeneralPUDoFErrorEstimator.inst.in"
