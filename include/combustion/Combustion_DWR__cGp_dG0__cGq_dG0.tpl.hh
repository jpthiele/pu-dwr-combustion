/**
 * @file Combustion_DWR__cGp_dG0__cGq_dG0.tpl.hh
 *
 * @author Philipp Thiele (PT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 *
 * @brief Combustion/DWR Problem with primal solver: cG(p)-dG(0) and dual solver: cG(q)-dG(0)
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

#ifndef __Combustion_DWR__cGp_dG0__cGq_dG0_tpl_hh
#define __Combustion_DWR__cGp_dG0__cGq_dG0_tpl_hh

// PROJECT includes
#include <combustion/parameters/ParameterSet.hh>
#include <combustion/grid/Grid_DWR.tpl.hh>
#include <combustion/ErrorEstimator/ErrorEstimators.hh>
#include <combustion/ErrorEstimator/EstimatorArgument.hh>

// DTM++ includes
#include <DTM++/base/LogStream.hh>
#include <DTM++/base/Problem.hh>
#include <DTM++/io/DataOutput.tpl.hh>
#include <DTM++/types/storage_data_trilinos_vectors.tpl.hh>

// DEAL.II includes
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

// C++ includes
#include <memory>
#include <algorithm>
#include <list>
#include <iterator>

#include "sys/types.h"
#include "sys/sysinfo.h"

namespace combustion {


template<int dim>
class Combustion_DWR__cGp_dG0__cGq_dG0 : public DTM::Problem {
public:
    Combustion_DWR__cGp_dG0__cGq_dG0():
    mpi_comm(MPI_COMM_WORLD){}

    virtual ~Combustion_DWR__cGp_dG0__cGq_dG0() = default;
	
    virtual void set_input_parameters(
        std::shared_ptr< dealii::ParameterHandler > parameter_handler
    );
	
    virtual void run();

protected:

    std::shared_ptr< combustion::dwr::ParameterSet > parameter_set;

    std::shared_ptr< combustion::Grid_DWR<dim,1> > grid;
    virtual void check_parameters();
    virtual void init_grid();

    struct {
       double Le;
       double alpha;
       double beta;
       double x_tilde;

       double robin_factor_theta;
       double robin_factor_Y;
    } parameter;

    virtual void init_parameters();

    struct {
        unsigned int max_steps;
        double       lower_bound;
        double       rebuild;
        unsigned int line_search_steps;
        double       line_search_damping;
    } newton;

    virtual void init_newton_parameters();

    struct {
        double rod;
        double domain;
    } cardinality;

    virtual void calculate_cardinalities(const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab);

    struct {
        std::shared_ptr< dealii::Function<dim> > omega_dtheta;
        std::shared_ptr< dealii::Function<dim> > omega_dY;
        
        std::shared_ptr< dealii::Function<dim> > f;
        
        std::shared_ptr< dealii::Function<dim> > u_D;
        std::shared_ptr< dealii::Function<dim> > u_N;
        std::shared_ptr< dealii::Function<dim> > u_R;
        std::shared_ptr< dealii::Function<dim> > u_0;
        
        std::shared_ptr< dealii::Function<dim> > u_E;
    } function;
	
    virtual void init_functions();
	
        
    ////////////////////////////////////////////////////////////////////////////
    // primal problem:
    //

    /// primal: data structures for forward time marching
    struct {
        // storage container
        struct {
            /// primal solution dof list
            std::shared_ptr< DTM::types::storage_data_trilinos_vectors<1> > u;
        } storage;

        struct {
            struct {
                double reaction_rate;
                double rod_species_concentration;
            } average;
        } functionals;


        /// temporary storage for primal solution u at \f$ t_m \f$
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > um; // dof on t_m

        /// temporary storage for primal solution u at \f$ t_n \f$
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > un; // dof on t_n

        /// temporary storage for previous solution rhs part i.e. mass matrix times um
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > Mum;

        /// temporary storage for primal right hand side assembly
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > f0;
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > u_N0;
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > u_R0;
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > f_Arr;
        /// temporary storage for primal system matrix
        std::shared_ptr< dealii::TrilinosWrappers::SparseMatrix > A;
        std::shared_ptr< dealii::TrilinosWrappers::SparseMatrix > A_lin;

        /// temporary storage for primal system right hand side
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > b;

        /// newton defect
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > du;

        /// temporaty storage for matrix vector results
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > b_tmp;


        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > u_tmp;


        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > owned_tmp;
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > relevant_tmp;


        // Data Output
        std::shared_ptr< DTM::DataOutput<dim> > data_output;
        int    data_output_dwr_loop;
        double data_output_time_value;
        double data_output_trigger;
        bool   data_output_trigger_type_fixed;

    } primal;
	
    virtual void primal_reinit_storage();


    virtual void primal_setup_slab_low_order(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u
    );
    virtual void primal_setup_slab_high_order(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u
    );

    virtual void primal_assemble_system_low_order(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u
    );

    virtual void primal_assemble_system_high_order(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u
    );

    virtual void primal_assemble_rhs_nonlin_part_high_order(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector> u,
        const double t0
    );
    virtual void primal_assemble_rhs_nonlin_part_low_order(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector> u,
        const double t0
    );

    virtual void primal_construct_rhs(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
    );

    virtual void primal_solve_slab_newton_problem(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
        const double t0
    );
        

    /// do the forward time marching process of the primal problem
    virtual void primal_do_forward_TMS();


    // post-processing functions for L2(L2) error
    double primal_error;
    std::shared_ptr< dealii::Function<dim> > primal_L2_L2_error_weight;
    virtual void primal_init_error_computations();

    // post-processing functions for data output
    virtual void primal_init_data_output();

    virtual void primal_do_data_output_on_slab(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
        const unsigned int dwr_loop,
        const bool dG_initial_value
    );

    virtual void primal_do_data_output(
        const unsigned int dwr_loop,
        bool last
    );

    ////////////////////////////////////////////////////////////////////////////
    // dual problem:
    //

    /// dual: data structures for backward time marching and error estimation
    struct {
        // storage container
        struct {
            /// dual solution dof list
            std::shared_ptr< DTM::types::storage_data_trilinos_vectors<1> > z;
            std::shared_ptr< DTM::types::storage_data_trilinos_vectors<1> > output;
        } storage;
        
        /// temporary storage for dual solution z at \f$ t_m \f$
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > zm;
        
        /// temporary storage for dual solution z at \f$ t_n \f$
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > zn;

        /// temporary storage for 'upcoming' solution rhs part i.e. mass matrix times zm
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > Mzm;

        /// temporary storage for dual solution z on \f$ \hat t_0 \f$
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > u0;


        /// temporary storage for dual solution z on \f$ \hat .5*(t_0+t_1) \f$
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > u2;

        /// temporary storage for dual solution z on \f$ \hat t_1 \f$
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > u1;

        /// temporary storage for dual right hand side assembly
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > Je0;

        /// temporary storage for dual right hand side assembly
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > Je1;

        /// temporary storage for dual system matrix
        std::shared_ptr< dealii::TrilinosWrappers::SparseMatrix > A;

        /// temporary storage for dual system right hand side
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > b;

        /// temporary storage for matrix vector mult results
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > owned_tmp;

        /// temporary storage for matrix vector mult results
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > relevant_tmp;
        // Data Output
        std::shared_ptr< DTM::DataOutput<dim> > data_output;
        int    data_output_dwr_loop;
        double data_output_time_value;
        double data_output_trigger;
        bool   data_output_trigger_type_fixed;
    } dual;
	
    virtual void dual_reinit_storage();

    virtual void dual_setup_slab_high_order(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &z,
        bool mesh_interpolate = false
    );

    virtual void dual_setup_slab_low_order(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &z,
        bool mesh_interpolate = false
    );

    virtual void dual_assemble_system_lin_part_high_order(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
    );

    virtual void dual_assemble_system_lin_part_low_order(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
    );

    virtual void dual_assemble_system_nonlin_part_high_order(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
        const unsigned int &n,
        const double &t0
    );

    virtual void dual_assemble_system_nonlin_part_low_order(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
        const unsigned int &n,
        const double &t0
    );

    virtual void dual_assemble_rhs_functionals_high_order(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
        const unsigned int &n,
        const double &t0,
        const double &t1
    );

    virtual void dual_assemble_rhs_functionals_low_order(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
        const unsigned int &n,
        const double &t0,
        const double &t1
    );


    virtual void dual_construct_rhs(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
    );

    virtual void dual_solve_slab_linear_problem(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &z
    );

    /// do the backward time marching process of the dual problem

    virtual void dual_do_backward_TMS();

    // post-processing functions for data output
    virtual void dual_init_data_output();

    virtual void dual_do_data_output_on_slab(
        const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
        const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &z,
        const unsigned int dwr_loop
    );

    virtual void dual_do_data_output(
        const unsigned int dwr_loop,
        bool last
);
    ////////////////////////////////////////////////////////////////////////////
    // error estimation and space-time grid adaption
    //

    struct {
        struct {
            /// error indicator \f$ \eta_{I_n} \f$  list
            struct{
                std::shared_ptr< DTM::types::storage_data_trilinos_vectors<1> > eta_h;
                std::shared_ptr< DTM::types::storage_data_trilinos_vectors<1> > eta_k;
            } primal;
            struct{
                std::shared_ptr< DTM::types::storage_data_trilinos_vectors<1> > eta_h;
                std::shared_ptr< DTM::types::storage_data_trilinos_vectors<1> > eta_k;
            } adjoint;
        } storage;

        /// error estimator
        std::shared_ptr< combustion::dwr::estimator::PUDoFErrorEstimator<dim> > pu_dof;

        // Data Output
        std::shared_ptr< DTM::DataOutput<dim> > data_output;
        int    data_output_dwr_loop;
        double data_output_time_value;
        double data_output_trigger;
        bool   data_output_trigger_type_fixed;
    } error_estimator;
	
	virtual void eta_reinit_storage();
	virtual void eta_interpolate_slab(
            const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
            const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
            const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &z,
            std::shared_ptr<combustion::dwr::estimator::Arguments>,
            bool initial_slab,
            bool last_slab
	);

	virtual void compute_pu_dof_error_indicators();

	virtual void eta_init_data_output();

	virtual void eta_do_data_output_on_slab(
            const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
            const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &eta_k,
            const unsigned int dwr_loop,
            std::string eta_type
	);

	virtual void eta_do_data_output(
            const unsigned int dwr_loop,
            bool last
	);

	virtual void compute_effectivity_index();
	
	virtual void refine_and_coarsen_space_time_grid();

	virtual void refine_and_coarsen_space_time_grid_global();
	virtual void refine_and_coarsen_space_time_grid_sv_dof();
	
	////////////////////////////////////////////////////////////////////////////
	// other
	//
	
	unsigned int setw_value_dwr_loops;
	
	// Convergence Table
	dealii::ConvergenceTable convergence_table;
	virtual void write_convergence_table_to_tex_file();

	//Timer
	std::shared_ptr<dealii::TimerOutput> timer;

	//MPI Communicator
	MPI_Comm mpi_comm;

};

} // namespace

#endif
