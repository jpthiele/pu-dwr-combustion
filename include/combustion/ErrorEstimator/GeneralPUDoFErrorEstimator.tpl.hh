/**
 * @file GeneralPUDoFErrorEstimator.tpl.hh
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
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


#ifndef __GeneralPUDoFErrorEstimator_tpl_hh
#define __GeneralPUDoFErrorEstimator_tpl_hh

// PROJECT includes
#include <combustion/grid/Grid_DWR.tpl.hh>
#include <combustion/parameters/ParameterSet.hh>
#include <combustion/ErrorEstimator/EstimatorArgument.hh>

// DTM++ includes
#include <DTM++/types/storage_data_trilinos_vectors.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>

// C++ includes
#include <memory>
#include <vector>

namespace combustion {
namespace dwr {

namespace estimator{

namespace Assembly {

namespace Scratch {

/// Struct for scratch on local error estimate on cell
template<int dim>
struct PUDoFErrorEstimateOnCell {
    PUDoFErrorEstimateOnCell(
        const dealii::DoFHandler<dim>    &dof_dual,
        const dealii::DoFHandler<dim>    &dof_pu,
        const dealii::FiniteElement<dim> &fe_dual,
        const dealii::FiniteElement<dim> &fe_pu,
        const dealii::Quadrature<dim> &quad,
        const dealii::UpdateFlags &uflags
    );
        
    PUDoFErrorEstimateOnCell(const PUDoFErrorEstimateOnCell &scratch);
        
    PUDoFErrorEstimateOnCell& operator=(const PUDoFErrorEstimateOnCell &scratch) = default;
    // data structures of current cell
    dealii::FEValues<dim>         fe_values_dual;
    dealii::FEValues<dim>         fe_values_pu;
        
    const dealii::DoFHandler<dim> &dof_dual;
    const dealii::DoFHandler<dim> &dof_pu;
        
    std::vector< dealii::types::global_dof_index > local_dof_indices_dual;
        
    // shape fun scratch:
    std::vector<double>                 phi;
    std::vector<dealii::Tensor<1,dim> > grad_phi;
        
    //partition of unity shape functions
    std::vector<double>                 chi;
    std::vector<dealii::Tensor<1,dim> > grad_chi;

    // local dof scratch:
    std::vector<double> local_u_kh_m;
    std::vector<double> local_u_kh_n;

    std::vector<double> local_u_k_n;
    std::vector<double> local_u_k_np1;

    std::vector<double> local_z_kh_n;
    std::vector<double> local_z_kh_np1;

    std::vector<double> local_z_k_n;
    std::vector<double> local_z_k_m;
        
        
        
    double val_dw_theta_j;
    double val_dw_Y_j;

    double val_pw_theta_j;
    double val_pw_Y_j;

    double val_theta_kh_j;
    double val_Y_kh_j;

    double val_z_theta_kh_j;
    double val_z_Y_kh_j;

    double val_theta_kh_jump_j;
    double val_Y_kh_jump_j;

    double val_z_theta_kh_jump_j;
    double val_z_Y_kh_jump_j;
        
    dealii::Tensor<1,dim> grad_dw_theta_j;
    dealii::Tensor<1,dim> grad_dw_Y_j;

    dealii::Tensor<1,dim> grad_pw_theta_j;
    dealii::Tensor<1,dim> grad_pw_Y_j;

    dealii::Tensor<1,dim> grad_theta_kh_j;
    dealii::Tensor<1,dim> grad_Y_kh_j;

    dealii::Tensor<1,dim> grad_z_theta_kh_j;
    dealii::Tensor<1,dim> grad_z_Y_kh_j;

    double omega;
    double omega_y;
    double omega_theta;

    // other:
    double JxW;

    unsigned int q;
    unsigned int d;
    unsigned int j;
};

/// Struct for scratch on local error estimate on face
template<int dim>
struct PUDoFErrorEstimateOnFace {
    PUDoFErrorEstimateOnFace(
        const dealii::DoFHandler<dim> &dof_dual,
        const dealii::DoFHandler<dim> &dof_pu,
        const dealii::FiniteElement<dim> &fe_dual,
        const dealii::FiniteElement<dim> &fe_pu,
        const dealii::Quadrature<dim-1> &quad,
        const dealii::UpdateFlags &uflags
    );
        
    PUDoFErrorEstimateOnFace(const PUDoFErrorEstimateOnFace &scratch);

    PUDoFErrorEstimateOnFace& operator=(const PUDoFErrorEstimateOnFace &scratch) = default;
    //data structures for current face on cell (+)
    dealii::FEFaceValues<dim>   fe_face_values_dual;
    dealii::FEFaceValues<dim>   fe_face_values_pu;

    const dealii::DoFHandler<dim> &dof_dual;
    const dealii::DoFHandler<dim> &dof_pu;

    std::vector< dealii::types::global_dof_index > local_dof_indices_dual;

    // shape fun scratch
    std::vector<double>                     phi;

    // partition of unity shape functions
    std::vector<double>                     chi;


    // local dof scratch:
    std::vector<double> local_u_kh_m;
    std::vector<double> local_u_kh_n;

    std::vector<double> local_u_k_n;
    std::vector<double> local_u_k_np1;


    std::vector<double> local_z_kh_m;
    std::vector<double> local_z_kh_n;

    std::vector<double> local_z_k_m;
    std::vector<double> local_z_k_n;

    double val_dw_theta_j;
    double val_dw_Y_j;

    double val_pw_theta_j;
    double val_pw_Y_j;

    double val_theta_kh_j;
    double val_Y_kh_j;

    double val_z_theta_kh_j;
    double val_z_Y_kh_j;

    // function eval scratch
    double value_u_N_t;
    double value_u_N_y;
    double value_u_R_t;
    double value_u_R_y;

    // other
    double JxW;

    unsigned int q;
    unsigned int j;
        
};

template<int dim>
struct PUDoFErrorEstimates {
    PUDoFErrorEstimates(
        const dealii::DoFHandler<dim>    &dof_dual,
        const dealii::DoFHandler<dim>    &dof_pu,
        const dealii::FiniteElement<dim> &fe_dual,
        const dealii::FiniteElement<dim> &fe_pu,
        const dealii::Quadrature<dim>    &quad_cell,
        const dealii::Quadrature<dim-1>  &quad_face,
        const dealii::UpdateFlags        &uflags_cell,
        const dealii::UpdateFlags        &uflags_face
    );

    PUDoFErrorEstimates( const PUDoFErrorEstimates &scratch);

    PUDoFErrorEstimateOnCell<dim> cell;
    PUDoFErrorEstimateOnFace<dim> face;

    // other
    unsigned int face_no;
};

} // namespace Scratch

namespace CopyData {

/// Struct for copydata on local cell matrix.
template<int dim>
struct PUDoFErrorEstimateOnCell{
    PUDoFErrorEstimateOnCell(const dealii::FiniteElement<dim> &fe);
    PUDoFErrorEstimateOnCell(const PUDoFErrorEstimateOnCell &copydata);

    dealii::Vector<double> local_eta_h_vector;
    dealii::Vector<double> local_eta_k_vector;
    std::vector< dealii::types::global_dof_index > local_dof_indices_pu;
};

template<int dim>
struct PUDoFErrorEstimateOnFace{
    PUDoFErrorEstimateOnFace(const dealii::FiniteElement<dim> &fe);
    PUDoFErrorEstimateOnFace(const PUDoFErrorEstimateOnFace &copydata);

    dealii::Vector<double> local_eta_h_vector;
    dealii::Vector<double> local_eta_k_vector;
    std::vector< dealii::types::global_dof_index > local_dof_indices_pu;
};

template<int dim> 
struct PUDoFErrorEstimates {
    PUDoFErrorEstimates(const dealii::FiniteElement<dim> &fe);
    PUDoFErrorEstimates(const PUDoFErrorEstimates &copydata);

    PUDoFErrorEstimateOnCell<dim> cell;
    PUDoFErrorEstimateOnFace<dim> face;
};

} // namespace CopyData

} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////

/**
 * Implements the computation of the node-wise a posteriori error estimator \f$ \eta_i \f$
 * with a partition of unity localization.
 * This is achieved by multiplying the dual weights \f$ \bar{z} = z_{kh}-i_{kh}z_{kh} \f$ with 
 * a node wise partition of unity \f$ \chi_i\in V_{PU}, i=1,\dots,N_{PU} \f$. 
 * The simplest choice is \f$ V_{PU}= Q_1(\Omega\times (0,T)) \f$.
 * Given the variational formulation \f$ A(u,\phi )= f(\phi ) \f$ our primal error
 * estimator reads
 * \f[
 * \eta_i = F(\bar{z}\chi_i) - A(u_{kh},\bar{z}\chi_i) 
 * \f]
 * From this we obtain cell-wise estimators \f$ \eta_K \f$ by 
 * \f[
 * \eta_K = \sum\limits_{i\in K} \eta_i
 * \f]
 * Since we operate on tensor-product cells we can split the partition of
 * unity into temporal and spatial components \f$ \chi(x,t) = \tau(t)\xi(x) \f$. We will denote our underlying
 * spatial cell grid as \f$ K_x \f$ and our temporal interval by \f$ I_n \f$.
 * Then, we can expand our cell wise error as
 * \f[
 * \eta_K = \sum\limits_{j\in K_x}\sum\limits_{k\in I_m} 
 *             F(\bar{z}\xi_j\tau_k) - A(u_{kh},\bar{z}\xi_j\tau_j)
 * \f]
 * Adding suitable quadrature formulas in space and time we obtain four sums.
 * These can be reordered so that the two innermost sums are the spatial components, 
 * which can be seen as functions of the temporal quadrature point \f$ q_t \f$.
 * That way our space-time error contributions can be seen as a sum over spatial PU
 * estimators weighted by our temporal PU. 
 * Note: In contrast to the classical estimator we work in a variational setting.
 * Since we don't do partial integration to obtain a strong form of the estimator
 * we do not get any of the explicit face terms in our estimator.
 */
template<int dim>
class PUDoFErrorEstimator {
public:
    PUDoFErrorEstimator(MPI_Comm _mpi_comm):
    tau_n(0),
    tm(0),
    t0(0),
    tn(0),
    cell_number(0),
    mpi_comm(_mpi_comm)
    {};

    virtual ~PUDoFErrorEstimator() = default;

    virtual void estimate_primal(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        std::shared_ptr< Arguments > _args,
        std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector > _eta_h,
        std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector > _eta_k
    );

    virtual void estimate_dual(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        std::shared_ptr< Arguments > args,
        std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector > _eta_h,
        std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector > _eta_k
    );

    virtual void set_functions(
        std::shared_ptr< dealii::Function<dim> > _u_0,
        std::shared_ptr< dealii::Function<dim> > _u_N,
        std::shared_ptr< dealii::Function<dim> > _u_R
    );

    virtual void set_parameters (
        double alpha,
        double beta,
        double Le,
        double robin_factor_theta,
        double robin_factor_Y,
        double T,
        double domain_area,
        double rod_area,
        std::string goal_type
    );
        
protected:
        
    ////////////////////////////////////////////////////////////////////////////
    // assemble local functions:
    //

    virtual void assemble_local_primal_error(
        const typename dealii::Triangulation<dim>::active_cell_iterator &tria_cell,
        Assembly::Scratch::PUDoFErrorEstimates<dim> &scratch,
        Assembly::CopyData::PUDoFErrorEstimates<dim> &copydata
    );

    virtual void assemble_local_dual_error(
        const typename dealii::Triangulation<dim>::active_cell_iterator &tria_cell,
        Assembly::Scratch::PUDoFErrorEstimates<dim> &scratch,
        Assembly::CopyData::PUDoFErrorEstimates<dim> &copydata
    );

    virtual void assemble_primal_error_on_cell(
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell_dual,
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell_pu,
        Assembly::Scratch::PUDoFErrorEstimateOnCell<dim> &scratch,
        Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> &copydata
    );

    virtual void assemble_dual_error_on_cell(
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell_dual,
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell_pu,
        Assembly::Scratch::PUDoFErrorEstimateOnCell<dim> &scratch,
        Assembly::CopyData::PUDoFErrorEstimateOnCell<dim> &copydata
    );

    virtual void assemble_primal_error_on_robin_boundary_face(
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell_dual,
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell_pu,
        const unsigned int face_no,
        Assembly::Scratch::PUDoFErrorEstimateOnFace<dim>  &scratch,
        Assembly::CopyData::PUDoFErrorEstimateOnFace<dim> &copydata
    );

    virtual void assemble_dual_error_on_robin_boundary_face(
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell_dual,
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell_pu,
        const unsigned int face_no,
        Assembly::Scratch::PUDoFErrorEstimateOnFace<dim>  &scratch,
        Assembly::CopyData::PUDoFErrorEstimateOnFace<dim> &copydata
    );

    virtual void copy_local_error(
        const Assembly::CopyData::PUDoFErrorEstimates<dim> &copydata
    );

    ////////////////////////////////////////////////////////////////////////////
    // internal data structures:
    //
    std::shared_ptr< Arguments> args;

    struct {
        std::shared_ptr<dealii::AffineConstraints<double>> constraints;
    } pu;

    struct {
        std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector > x_h;
        std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector > x_k;
    } error_estimator;

    struct {
        std::shared_ptr< dealii::Function<dim> > u_D;
        std::shared_ptr< dealii::Function<dim> > u_N;
        std::shared_ptr< dealii::Function<dim> > u_R;
        std::shared_ptr< dealii::Function<dim> > u_0;
    } function;
        
    struct {
        double alpha;
        double beta;
        double Le;
        double c;
        double T;
        struct {
            double theta;
            double Y;
        } robin;
        
        struct {
            double domain;
            double rod;
        } area;

        std::string goal_type;
    } param;

    double tau_n;
    double tm;
    double t0;
    double tn;

    int cell_number;

    MPI_Comm mpi_comm;
};

}}} // namespace

#endif
