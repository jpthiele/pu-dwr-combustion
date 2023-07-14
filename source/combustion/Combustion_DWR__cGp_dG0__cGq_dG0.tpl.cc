/**
 * @file Combustion_DWR__cGp_dG0__cGq_cG1.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 *
 * @brief Combustion/DWR Problem with primal solver: cG(p)-dG(0) and dual solver: cG(q)-dG(0)
 */

/*  Copyright (C) 2012-2023 by Jan Philipp Thiele, Uwe Koecher                */
/*                          and Marius Paul Bruchhaeuser               s       */
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
#include <combustion/Combustion_DWR__cGp_dG0__cGq_dG0.tpl.hh>

#include <combustion/grid/Grid_DWR_Selector.tpl.hh>

#include <combustion/InitialValue/InitialValue_Selector.tpl.hh>
#include <combustion/DirichletBoundary/DirichletBoundary_Selector.tpl.hh>
#include <combustion/NeumannBoundary/NeumannBoundary_Selector.tpl.hh>
#include <combustion/RobinBoundary/RobinBoundary_Selector.tpl.hh>

#include <combustion/types/boundary_id.hh>

#include <combustion/assembler/L2_MassAssembly.tpl.hh>
#include <combustion/assembler/L2_LaplaceAssembly.tpl.hh>
#include <combustion/assembler/L2_RobinMatrixAssembly.tpl.hh>
#include <combustion/assembler/L2_ArrheniusDerivAssembly.tpl.hh>

#include <combustion/assembler/L2_MassMultAssembly.tpl.hh>
#include <combustion/assembler/L2_LaplaceMultAssembly.tpl.hh>
#include <combustion/assembler/L2_RobinMatrixMultAssembly.tpl.hh>
#include <combustion/assembler/L2_ArrheniusFunctionAssembly.tpl.hh>

#include <combustion/assembler/L2_NeumannConstrainedAssembly.tpl.hh>
template <int dim>
using NeumannAssembler = combustion::Assemble::L2::NeumannConstrained::Assembler<dim>;

#include <combustion/assembler/L2_RobinConstrainedAssembly.tpl.hh>
template <int dim>
using RobinAssembler = combustion::Assemble::L2::RobinConstrained::Assembler<dim>;

#include <combustion/assembler/L2_Je_reaction_rate_Assembly.tpl.hh>
#include <combustion/assembler/L2_Je_rod_species_concentration_Assembly.tpl.hh>

// DEAL.II includes
#include <deal.II/fe/fe_tools.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
// #include <deal.II/numerics/dof_tools.h>


#include "stdlib.h"
#include "stdio.h"
#include "string.h"

namespace combustion {

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
set_input_parameters(
    std::shared_ptr< dealii::ParameterHandler > parameter_handler) {
    Assert(parameter_handler.use_count(), dealii::ExcNotInitialized());

    parameter_set = std::make_shared< combustion::dwr::ParameterSet > (
            parameter_handler
    );
}


template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
run() {
    timer = std::make_shared<dealii::TimerOutput>(DTM::pout,dealii::TimerOutput::summary,dealii::TimerOutput::wall_times);
    timer->enter_subsection("check parameters");
    check_parameters();
    // determine setw value for dwr loop number of data output filename
    setw_value_dwr_loops = static_cast<unsigned int>(
        std::floor(std::log10(parameter_set->dwr.loops))+1
    );

    timer->leave_subsection();
    timer->enter_subsection("initialization");
    init_grid();
    init_parameters();
    init_newton_parameters();
    init_functions();

    ////////////////////////////////////////////////////////////////////////////
    // DWR loop
    //
    DTM::pout
        << std::endl
        << "*******************************************************************"
        << "*************" << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // setup solver/reduction control for outer dwr loop
    std::shared_ptr< dealii::ReductionControl > solver_control_dwr;
    solver_control_dwr = std::make_shared< dealii::ReductionControl >();

    if (!parameter_set->dwr.solver_control.in_use) {
        solver_control_dwr->set_max_steps(parameter_set->dwr.loops);
        solver_control_dwr->set_tolerance(0.);
        solver_control_dwr->set_reduction(0.);

        DTM::pout
            << std::endl
            << "dwr loops (fixed number) = " << solver_control_dwr->max_steps()
            << std::endl << std::endl;
    }
    else {
        solver_control_dwr->set_max_steps(
            parameter_set->dwr.solver_control.max_iterations
        );

        solver_control_dwr->set_tolerance(
            parameter_set->dwr.solver_control.tolerance
        );

        solver_control_dwr->set_reduction(
            parameter_set->dwr.solver_control.reduction_mode ?
            parameter_set->dwr.solver_control.reduction :
            parameter_set->dwr.solver_control.tolerance
        );
    }

    DTM::pout
        << std::endl
        << "dwr tolerance = " << solver_control_dwr->tolerance() << std::endl
        << "dwr reduction = " << solver_control_dwr->reduction() << std::endl
        << "dwr max. iterations = " << solver_control_dwr->max_steps() << std::endl
        << std::endl;

    dealii::SolverControl::State dwr_loop_state{dealii::SolverControl::State::iterate};
    solver_control_dwr->set_max_steps(solver_control_dwr->max_steps()-1);

    timer->leave_subsection();
    unsigned int dwr_loop{solver_control_dwr->last_step()+1};
    do {
        if (dwr_loop > 0) {
            // do space-time mesh refinements and coarsenings
            {
                timer->enter_subsection("grid refinement");
                refine_and_coarsen_space_time_grid();
                timer->leave_subsection();
            }
        }

        DTM::pout
            << "***************************************************************"
            << "*****************" << std::endl
            << "dwr loop = " << dwr_loop << std::endl;

        convergence_table.add_value("DWR-loop", dwr_loop+1);

        timer->enter_subsection("grid distribute");

        grid->set_boundary_indicators();
        grid->distribute_dofs();
        // primal problem:

        timer->leave_subsection();
        timer->enter_subsection("storage primal");

        primal_reinit_storage();
        primal_init_data_output();

        if ( dwr_loop == 0 ){
            calculate_cardinalities(grid->slabs.begin());
        }
        timer->leave_subsection();

        primal_do_forward_TMS();


        timer->enter_subsection("data output primal");
        primal_do_data_output(dwr_loop,false);
        timer->leave_subsection();

        // check if dwr has converged
        dwr_loop_state = solver_control_dwr->check(
            dwr_loop,
            primal_error // convergence criterium here
        );
//
        if (dwr_loop_state == dealii::SolverControl::State::iterate) {
            DTM::pout << "state iterate = true" << std::endl;
        }
        else {
            DTM::pout << "state iterate = false" << std::endl;
        }
        // only solve dual problem and error estimators when doing
        // adaptive refinement
        if( parameter_set->dwr.refine_and_coarsen.adaptive) {

            // dual problem dG0
            timer->enter_subsection("storage dual");
            dual_reinit_storage();
            dual_init_data_output();
            timer->leave_subsection();

            dual_do_backward_TMS();

            timer->enter_subsection("data output dual");
            dual_do_data_output(dwr_loop,false);
            timer->leave_subsection();

            // error estimation
            timer->enter_subsection("storage eta");
            eta_reinit_storage();
            eta_init_data_output();
            timer->leave_subsection();

            DTM::pout << "estimating with DoF-wise partition of unity" << std::endl;
            timer->enter_subsection("pu estimator");

            compute_pu_dof_error_indicators();
            timer->leave_subsection();
            eta_do_data_output(dwr_loop,false);

            timer->enter_subsection("I_eff");
            compute_effectivity_index();
            timer->leave_subsection();

        }
        if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0){
            std::cout << "dwr loop = " << dwr_loop << " ... (done)" << std::endl;
        }
    } while ((dwr_loop_state == dealii::SolverControl::State::iterate) && ++dwr_loop);

    // data output of the last (final) dwr loop solution
    if (dwr_loop_state == dealii::SolverControl::State::success) {
        timer->enter_subsection("data output primal");
        primal_do_data_output(dwr_loop,true);
        timer->leave_subsection();
        timer->enter_subsection("data output dual");
        dual_do_data_output(dwr_loop,true);
        timer->leave_subsection();

    }
    std::cout << "finished!" << std::endl;
    timer->enter_subsection("convergence table");
    if( parameter_set->dwr.refine_and_coarsen.adaptive) {
        write_convergence_table_to_tex_file();
    }
    timer->leave_subsection();

}
template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
check_parameters() {
    Assert(parameter_set.use_count(), dealii::ExcNotInitialized());

    if ( parameter_set->fe.primal.time_type_support_points.compare("Gauss") == 0 ||
         parameter_set->fe.dual.time_type_support_points.compare("Gauss") == 0
    )
    {
        DTM::pout << "sorry, Gauss integration not implemented out yet"
                  << "\naborting, bye!"
                  << std::endl;

        exit(EXIT_FAILURE);
    }

    int primal_space_order,dual_space_order;
    if ( parameter_set->fe.primal.high_order) {
        primal_space_order = parameter_set->fe.dual.q;
    }else {
        primal_space_order =  parameter_set->fe.primal.p;
    }


    if ( parameter_set->fe.dual.high_order){
        dual_space_order = parameter_set->fe.dual.q;
    }else {
        dual_space_order =  parameter_set->fe.primal.p;
    }

    // check primal time discretisation
    if ((parameter_set->fe.primal.time_type.compare("dG") == 0) &&
        (parameter_set->fe.primal.r == 0)) {
        DTM::pout
            << "primal time discretisation = dG(0)-Q_"
            << primal_space_order
            << std::endl;
    }
    else {
        AssertThrow(
            false,
            dealii::ExcMessage(
                "primal time discretisation unknown"
            )
        );
    }

    // check dual time discretisation
    if ((parameter_set->fe.dual.time_type.compare("dG") == 0) &&
        (parameter_set->fe.dual.s == 0)) {
        DTM::pout
            << "dual time discretisation = dG(0)-Q_"
            << dual_space_order
            << std::endl;
    }
    else {
        AssertThrow(
            false,
            dealii::ExcMessage(
                "dual time discretisation unknown"
            )
        );
    }

}

////////////////////////////////////////////////////////////////////////////////
// protected member functions (internal use only)
//

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
init_grid() {
    Assert(parameter_set.use_count(), dealii::ExcNotInitialized());

    ////////////////////////////////////////////////////////////////////////////
    // init grid from input parameter file spec.
    //
    {
        combustion::grid::Selector<dim> selector;
        selector.create_grid(
            parameter_set->Grid_Class,
            parameter_set->Grid_Class_Options,
            parameter_set->TriaGenerator,
            parameter_set->TriaGenerator_Options,
            grid,
            mpi_comm
        );

        Assert(grid.use_count(), dealii::ExcNotInitialized());
    }

    ////////////////////////////////////////////////////////////////////////////
    // initialize slabs of grid
    //

    Assert((parameter_set->fe.primal.p), dealii::ExcInvalidState());
    Assert(
        (parameter_set->fe.primal.p < parameter_set->fe.dual.q),
        dealii::ExcInvalidState()
    );

    Assert((parameter_set->t0 >= 0), dealii::ExcInvalidState());
    Assert((parameter_set->t0 < parameter_set->T), dealii::ExcInvalidState());
    Assert((parameter_set->tau_n > 0), dealii::ExcInvalidState());

    Assert(grid.use_count(), dealii::ExcNotInitialized());
    grid->initialize_slabs(
        parameter_set->fe.primal.p,
        parameter_set->fe.dual.q,
        parameter_set->t0,
        parameter_set->T,
        parameter_set->tau_n
    );

    grid->generate();

    grid->refine_global(
        parameter_set->global_refinement
    );

    DTM::pout
        << "grid: number of slabs = " << grid->slabs.size()
        << std::endl;
}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
init_parameters() {
    Assert(parameter_set.use_count(), dealii::ExcNotInitialized());

    parameter.Le      = parameter_set->lewis_number;
    parameter.alpha   = parameter_set->arrhenius_alpha;
    parameter.beta    = parameter_set->arrhenius_beta;
    parameter.x_tilde = parameter_set->x_tilde;
    DTM::pout << "* equation parameters found:"
              << "\n\tLewis Number Le = " << parameter.Le
              << "\n\tArrhenius Law:"
              << "\n\t\tgas expansion alpha = " << parameter.alpha
              << "\n\t\tactivation energy beta = " << parameter.beta
              << "\n\tinitial flame front position x = " << parameter.x_tilde
              << std::endl;

    std::string argument;
    std::vector< std::string > options;
    for (auto &character : parameter_set->robin_boundary_u_R_factors) {
        if (!std::isspace(character) && (character!='\"') ) {
            argument += character;
        }
        else {
            if (argument.size()) {
                options.push_back(argument);
                argument.clear();
            }
        }
    }
    if (argument.size()) {
        options.push_back(argument);
        argument.clear();
    }
    AssertThrow(options.size() == 2,
        dealii::ExcMessage(
                "robin_boundary factors invalid, "
            "please check your input file data."
        )
    );

    parameter.robin_factor_theta = std::stod(options.at(0));
    parameter.robin_factor_Y     = std::stod(options.at(1));

    DTM::pout << "*robin boundary factors found "
              << "\n\t dn theta = g_1 - " << parameter.robin_factor_theta
              << "theta"
              << "\n\t dn Y = g_2 - " << parameter.robin_factor_Y
              << "Y"
              << std::endl;

}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
init_newton_parameters() {
    Assert(parameter_set.use_count(), dealii::ExcNotInitialized());

    newton.max_steps = parameter_set->newton.max_steps;
    newton.lower_bound = parameter_set->newton.lower_bound;
    newton.rebuild = parameter_set->newton.rebuild;
    newton.line_search_steps = parameter_set->newton.line_search_steps;
    newton.line_search_damping = parameter_set->newton.line_search_damping;


    DTM::pout << "\n* Newton parameters found:"
              << "\n\tmaximum # of steps = " << newton.max_steps
              << "\n\twanted tolerance = " << newton.lower_bound
              << "\n\trebuild matrix if residual quotient > " << newton.rebuild
              << "\n\tLine Search parameters:"
              << "\n\t\tmaximum # of steps = " << newton.line_search_steps
              << "\n\t\tdamping factor = " << newton.line_search_damping
              << std::endl << std::endl;

}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
init_functions() {


    {
        combustion::initial_value::Selector<dim> selector;
        selector.create_functions(
            parameter_set->initial_value_u0_function,
            parameter_set->initial_value_u0_options,
            function.u_0
        );

        Assert(function.u_0.use_count(), dealii::ExcNotInitialized());
    }

    // dirichlet boundary function u_D:
    {
        combustion::dirichlet_boundary::Selector<dim> selector;
        selector.create_function(
            parameter_set->dirichlet_boundary_u_D_function,
            parameter_set->dirichlet_boundary_u_D_options,
            function.u_D
        );

        Assert(function.u_D.use_count(), dealii::ExcNotInitialized());
    }

    //neumann boundary function u_N = \epsilon \partial_n u(x,t):
    {
        combustion::neumann_boundary::Selector<dim> selector;
        selector.create_function(
            parameter_set->neumann_boundary_u_N_function,
            parameter_set->neumann_boundary_u_N_options,
            function.u_N
        );

        Assert(function.u_N.use_count(), dealii::ExcNotInitialized());
    }

    //robin boundary function u_R - a u = \partial_n u(x,t);
    {
        combustion::robin_boundary::Selector<dim> selector;
        selector.create_function(
            parameter_set->robin_boundary_u_R_function,
            parameter_set->robin_boundary_u_R_options,
            function.u_R
        );

        Assert(function.u_R.use_count(), dealii::ExcNotInitialized());
    }

}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
calculate_cardinalities(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
) {


    dealii::QGauss<dim> quadrature_formula(3);

    dealii::FEValues<dim> fe_values(*slab->low->fe,
                                    quadrature_formula,
                                    dealii::update_JxW_values );

    dealii::QGauss<dim-1> face_quadrature_formula(3);

    dealii::FEFaceValues<dim> fe_face_values(*slab->low->fe,
                                             face_quadrature_formula,
                                             dealii::update_JxW_values );

    cardinality.domain = 0;
    cardinality.rod = 0;

    typename dealii::DoFHandler<dim>::active_cell_iterator
    cell = slab->low->dof->begin_active(),
    endc = slab->low->dof->end();

    for ( ; cell != endc ; ++cell )
    {
        fe_values.reinit(cell);

        for (unsigned int q = 0 ; q < quadrature_formula.size() ; q++ )
        {
            cardinality.domain+= fe_values.JxW(q);
        }

            if ( cell -> at_boundary() )
            {
                for ( unsigned int face_no = 0;
                    face_no < dealii::GeometryInfo<dim>::faces_per_cell;
                    face_no++ )
                {
                    if(( cell->face(face_no)->at_boundary() ) &&
                       (  cell->face(face_no)->boundary_id() ==
                           static_cast<dealii::types::boundary_id> (
                               combustion::types::boundary_id::Robin)))
                    {

                        fe_face_values.reinit(cell,face_no);

                        for (unsigned int q=0 ; q < face_quadrature_formula.size(); q++ )
                        {
                            cardinality.rod += fe_face_values.JxW(q);
                        }
                    }
                }

            }

    }

    DTM::pout << "\n* cardinalities calculated:"
              << "\n\tdomain area = " << cardinality.domain
              << "\n\trod surface area = " << cardinality.rod
              << std::endl << std::endl;
}


////////////////////////////////////////////////////////////////////////////////
// primal problem
//

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
primal_reinit_storage() {
    ////////////////////////////////////////////////////////////////////////////
    // init storage containers for vector data:
    // NOTE: * primal space: time dG(0) method (having 1 independent solution)
    //       * primal solution dof vectors: u
    //

    Assert(grid.use_count(), dealii::ExcNotInitialized());
    // get number of time steps N
    const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};

    primal.storage.u = std::make_shared< DTM::types::storage_data_trilinos_vectors<1> > ();
    primal.storage.u->resize(N);


    {
        auto slab = grid->slabs.begin();
        for (auto &element : *primal.storage.u) {
            for (unsigned int j{0}; j < element.x.size(); ++j) {
                element.x[j] = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

                Assert(slab != grid->slabs.end(), dealii::ExcInternalError());

                if (parameter_set->fe.primal.high_order){
                    Assert(slab->high->dof.use_count(), dealii::ExcNotInitialized());
                    Assert(slab->high->locally_owned_dofs.use_count(), dealii::ExcNotInitialized());

                    element.x[j]->reinit(*slab->high->locally_owned_dofs,
                                         *slab->high->locally_relevant_dofs,
                                         mpi_comm);

                } else {
                    Assert(slab->low->dof.use_count(), dealii::ExcNotInitialized());
                    Assert(slab->low->locally_owned_dofs.use_count(), dealii::ExcNotInitialized());

                    element.x[j]->reinit(*slab->low->locally_owned_dofs,
                                         *slab->low->locally_relevant_dofs,
                                         mpi_comm);
                }
            }
        ++slab;
        }
    }
}


template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
primal_setup_slab_low_order(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u
){
    Assert(slab->low->dof.use_count(), dealii::ExcNotInitialized());
    Assert((slab != grid->slabs.end()), dealii::ExcInternalError());


    //Setup vectors
    primal.um = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    primal.um->reinit(*slab->low->locally_owned_dofs,*slab->low->locally_relevant_dofs,mpi_comm);
    *primal.um = 0;

    primal.relevant_tmp = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    primal.relevant_tmp->reinit(*slab->low->locally_owned_dofs,*slab->low->locally_relevant_dofs,mpi_comm);

    primal.Mum = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    primal.Mum->reinit(*slab->low->locally_owned_dofs,mpi_comm);

    primal.un = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    primal.un->reinit(*slab->low->locally_owned_dofs,mpi_comm);

    primal.b = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    primal.b->reinit(*slab->low->locally_owned_dofs,mpi_comm);

    primal.b_tmp = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    primal.b_tmp->reinit(*slab->low->locally_owned_dofs,mpi_comm);

    primal.du = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    primal.du -> reinit(*slab->low->locally_owned_dofs,mpi_comm);

    primal.owned_tmp = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    primal.owned_tmp->reinit(*slab->low->locally_owned_dofs,mpi_comm);

    //Setup sparsity pattern and matrix
    {
        dealii::Table<2,dealii::DoFTools::Coupling> coupling(2,2);
        coupling[0][0] = dealii::DoFTools::always;
        coupling[1][1] = dealii::DoFTools::always;
        coupling[1][0] = dealii::DoFTools::always;
        coupling[0][1] = dealii::DoFTools::always;

        dealii::DynamicSparsityPattern dsp(slab->low->dof->n_dofs());

        dealii::DoFTools::make_sparsity_pattern(
            *slab->low->dof,
            coupling,
            dsp,
            *slab->low->constraints,
            false // keep constrained dofs?
            ,dealii::Utilities::MPI::this_mpi_process(mpi_comm)
        );
        dealii::SparsityTools::distribute_sparsity_pattern(
            dsp,
            *slab->low->locally_owned_dofs,
            mpi_comm,
            *slab->low->locally_relevant_dofs);

        primal.A = std::make_shared< dealii::TrilinosWrappers::SparseMatrix >();
        primal.A->reinit(*slab->low->locally_owned_dofs,dsp);
    }

    //interpolate initial condition or previous solution
    //Using un here as it has no ghost entries
    if (slab == grid->slabs.begin()){
        dealii::VectorTools::interpolate(
            *slab->low->dof,
            *function.u_0,
            *primal.un
        );
        slab->low->constraints->distribute(*primal.un);
    } else{
        // for n > 1 interpolate between two (different) spatial meshes
        // the solution u(t_n)|_{I_{n-1}}  to  u(t_m)|_{I_n}
        dealii::VectorTools::interpolate_to_different_mesh(
            // solution on I_{n-1}:
            *std::prev(slab)->low->dof,
            *std::prev(u)->x[0],
            // solution on I_n:
            *slab->low->dof,
            *slab->low->constraints,
            *primal.un
        );
    }
    *primal.um = *primal.un;


    //// ASSEMBLY MASS MULT TIMES OLD SOLUTION ////////////////////////////////////
    {
        *primal.Mum =0;
        combustion::Assemble::L2::MassMult::
        Assembler<dim> assemble_mass(
            primal.Mum,
            slab->low->dof,
            slab->low->fe,
            slab->low->constraints
        );

        DTM::pout << "pu-dwr-combustion: assemble mass matrix mult...";
        assemble_mass.assemble(1.0,primal.um);
        DTM::pout << " (done)" << std::endl;
    }
}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
primal_setup_slab_high_order(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u
){
    Assert(slab->high->dof.use_count(), dealii::ExcNotInitialized());
    Assert((slab != grid->slabs.end()), dealii::ExcInternalError());


    primal.um = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    primal.um->reinit(*slab->high->locally_owned_dofs,*slab->high->locally_relevant_dofs,mpi_comm);
    *primal.um = 0;

    primal.relevant_tmp = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    primal.relevant_tmp->reinit(*slab->high->locally_owned_dofs,*slab->high->locally_relevant_dofs,mpi_comm);

    primal.Mum = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    primal.Mum->reinit(*slab->high->locally_owned_dofs,mpi_comm);

    primal.un = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    primal.un->reinit(*slab->high->locally_owned_dofs,mpi_comm);

    primal.b = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    primal.b->reinit(*slab->high->locally_owned_dofs,mpi_comm);


    primal.b_tmp = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    primal.b_tmp->reinit(*slab->high->locally_owned_dofs,mpi_comm);

    primal.owned_tmp = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    primal.owned_tmp->reinit(*slab->high->locally_owned_dofs,mpi_comm);

    primal.du = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    primal.du -> reinit(*slab->high->locally_owned_dofs,mpi_comm);

    //Setup sparsity pattern and matrix
    {
        dealii::Table<2,dealii::DoFTools::Coupling> coupling(2,2);
        coupling[0][0] = dealii::DoFTools::always;
        coupling[1][1] = dealii::DoFTools::always;
        coupling[1][0] = dealii::DoFTools::always;
        coupling[0][1] = dealii::DoFTools::always;

        dealii::DynamicSparsityPattern dsp(slab->high->dof->n_dofs());

        dealii::DoFTools::make_sparsity_pattern(
                        *slab->high->dof,
                        coupling,
                        dsp,
                        *slab->high->constraints,
                        false // keep constrained dofs?
        );

        dealii::SparsityTools::distribute_sparsity_pattern(
            dsp,
            *slab->high->locally_owned_dofs,
            mpi_comm,
            *slab->high->locally_relevant_dofs);

        primal.A = std::make_shared< dealii::TrilinosWrappers::SparseMatrix >();
        primal.A->reinit(*slab->high->locally_owned_dofs,
                         *slab->high->locally_owned_dofs,
                         dsp,mpi_comm);

    }

    //interpolate initial condition or previous solution
    //Using un here as it has no ghost entries
    if (slab == grid->slabs.begin()){
        dealii::VectorTools::interpolate(
            *slab->high->dof,
            *function.u_0,
            *primal.un
        );
        slab->high->constraints->distribute(*primal.un);
    } else{
        // for n > 1 interpolate between two (different) spatial meshes
        // the solution u(t_n)|_{I_{n-1}}  to  u(t_m)|_{I_n}
        dealii::VectorTools::interpolate_to_different_mesh(
            // solution on I_{n-1}:
            *std::prev(slab)->high->dof,
            *std::prev(u)->x[0],
            // solution on I_n:
            *slab->high->dof,
            *slab->high->constraints,
            *primal.un
        );
    }
    *primal.um = *primal.un;


    //// ASSEMBLY MASS MULT TIMES OLD SOLUTION ////////////////////////////////////
    {
        *primal.Mum =0;
        combustion::Assemble::L2::MassMult::
        Assembler<dim> assemble_mass(
            primal.Mum,
            slab->high->dof,
            slab->high->fe,
            slab->high->constraints
        );

        DTM::pout << "pu-dwr-combustion: assemble mass matrix mult...";
        assemble_mass.assemble(1.0,primal.um);
        DTM::pout << " (done)" << std::endl;
    }
}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
primal_assemble_system_low_order(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u
) {
   *primal.A = 0;

    // ASSEMBLY ROBIN MATRIX PART ////////////////////////////////////////////
    {
        combustion::Assemble::L2::RobinMatrix::
        Assembler<dim> assemble_robin_cell_terms (
            primal.A,
            slab->low->dof,
            slab->low->fe,
            slab->low->constraints
        );
        assemble_robin_cell_terms.set_parameters(
            parameter.robin_factor_theta,
            parameter.robin_factor_Y
        );
        assemble_robin_cell_terms.assemble(slab->tau_n(),parameter_set->robin_assembler_n_quadrature_points);
    }


    //// ASSEMBLY MASS MATRIX ////////////////////////////////////////////////////
    {
        combustion::Assemble::L2::Mass::
        Assembler<dim> assemble_mass(
            primal.A,
            slab->low->dof,
            slab->low->fe,
            slab->low->constraints
        );

        assemble_mass.assemble(1.0);
    }

    // ASSEMBLY STIFFNESS MATRIX ///////////////////////////////////////////////
    {
        combustion::Assemble::L2::Laplace::
        Assembler<dim> assemble_stiffness_cell_terms (
            primal.A,
            slab->low->dof,
            slab->low->fe,
            slab->low->constraints
        );

        assemble_stiffness_cell_terms.assemble(slab->tau_n());
    }



    // ASSEMBLY ARRHENIUS DIRECTIONAL DERIVATIVE ////////////////////////////////
    {
        combustion::Assemble::L2::Arrhenius::Deriv::
        Assembler<dim> assemble_Arrhenius(
            primal.A,
            slab->low->dof,
            slab->low->fe,
            slab->low->constraints
        );

        assemble_Arrhenius.set_parameters(
            parameter.alpha,
            parameter.beta,
            parameter.Le
        );

        assemble_Arrhenius.assemble(slab->tau_n(),u->x[0]);
    }

}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
primal_assemble_system_high_order(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u
) {
    *primal.A = 0;

    // ASSEMBLY ROBIN MATRIX PART ////////////////////////////////////////////
    {
        combustion::Assemble::L2::RobinMatrix::
        Assembler<dim> assemble_robin_cell_terms (
            primal.A,
            slab->high->dof,
            slab->high->fe,
            slab->high->constraints
        );
        assemble_robin_cell_terms.set_parameters(
            parameter.robin_factor_theta,
            parameter.robin_factor_Y
        );
        assemble_robin_cell_terms.assemble(slab->tau_n(),parameter_set->robin_assembler_n_quadrature_points);
    }

    // ASSEMBLY MASS MATRIX ////////////////////////////////////////////////////
    {
        combustion::Assemble::L2::Mass::
        Assembler<dim> assemble_mass(
            primal.A,
            slab->high->dof,
            slab->high->fe,
            slab->high->constraints
        );

        assemble_mass.assemble(1.0);
    }

    // ASSEMBLY STIFFNESS MATRIX ///////////////////////////////////////////////
    {
        combustion::Assemble::L2::Laplace::
        Assembler<dim> assemble_stiffness_cell_terms (
            primal.A,
            slab->high->dof,
            slab->high->fe,
            slab->high->constraints
        );

        assemble_stiffness_cell_terms.assemble(slab->tau_n());
    }

    // ASSEMBLY ARRHENIUS DIRECTIONAL DERIVATIVE ////////////////////////////////
    {
        combustion::Assemble::L2::Arrhenius::Deriv::
        Assembler<dim> assemble_Arrhenius(
            primal.A,
            slab->high->dof,
            slab->high->fe,
            slab->high->constraints
        );

        assemble_Arrhenius.set_parameters(
            parameter.alpha,
            parameter.beta,
            parameter.Le
        );

        assemble_Arrhenius.assemble(slab->tau_n(),u->x[0]);
    }

}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
primal_assemble_rhs_nonlin_part_low_order(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector> u,
    [[maybe_unused]] const double t0
) {
   //Arrhenius Function Assembly
   primal.f_Arr = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
   primal.f_Arr->reinit( *slab->low->locally_owned_dofs );
   *primal.f_Arr = 0;

   {
        combustion::Assemble::L2::Arrhenius::Function::
        Assembler<dim> assemble_Arr(
            primal.f_Arr,
            slab->low->dof,
            slab->low->fe,
            slab->low->constraints
        );

        assemble_Arr.set_parameters(
            parameter.alpha,
            parameter.beta,
            parameter.Le
        );

        assemble_Arr.assemble(u);
    }


   *primal.b = 0;
    //// ASSEMBLY MASS MULT TIMES OLD STEP ////////////////////////////////////
    {
        combustion::Assemble::L2::MassMult::
        Assembler<dim> assemble_mass(
            primal.b,
            slab->low->dof,
            slab->low->fe,
            slab->low->constraints
        );

        assemble_mass.assemble(-1.0,u);
    }

    //// ASSEMBLY LAPLACE MULT TIMES OLD STEP ////////////////////////////////////
    {
        combustion::Assemble::L2::LaplaceMult::
        Assembler<dim> assemble_laplace(
            primal.b,
            slab->low->dof,
            slab->low->fe,
            slab->low->constraints
        );

        assemble_laplace.assemble(-slab->tau_n(),u);
    }


    //// ASSEMBLY ROBIN MULT TIMES OLD STEP ////////////////////////////////////
    {
        combustion::Assemble::L2::RobinMatrixMult::
        Assembler<dim> assemble_robin(
            primal.b,
            slab->low->dof,
            slab->low->fe,
            slab->low->constraints
        );
        assemble_robin.set_parameters(
            parameter.robin_factor_theta,
            parameter.robin_factor_Y
        );

        assemble_robin.assemble(-slab->tau_n(),u,parameter_set->robin_assembler_n_quadrature_points);
    }
}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
primal_assemble_rhs_nonlin_part_high_order(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector> u,
    [[maybe_unused]] const double t0
) {
   //Arrhenius Function Assembly
   primal.f_Arr = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
   primal.f_Arr->reinit( *slab->high->locally_owned_dofs,mpi_comm );
   *primal.f_Arr = 0;

   {
        combustion::Assemble::L2::Arrhenius::Function::
        Assembler<dim> assemble_Arr(
            primal.f_Arr,
            slab->high->dof,
            slab->high->fe,
            slab->high->constraints
        );

        assemble_Arr.set_parameters(
            parameter.alpha,
            parameter.beta,
            parameter.Le
        );

        assemble_Arr.assemble(u);
    }


   *primal.b = 0;
    //// ASSEMBLY MASS MULT TIMES OLD STEP ////////////////////////////////////
    {
        combustion::Assemble::L2::MassMult::
        Assembler<dim> assemble_mass(
            primal.b,
            slab->high->dof,
            slab->high->fe,
            slab->high->constraints
        );

        assemble_mass.assemble(-1.0,u);
    }

    //// ASSEMBLY LAPLACE MULT TIMES OLD STEP ////////////////////////////////////
    {
        combustion::Assemble::L2::LaplaceMult::
        Assembler<dim> assemble_laplace(
            primal.b,
            slab->high->dof,
            slab->high->fe,
            slab->high->constraints
        );

        assemble_laplace.assemble(-slab->tau_n(),u);
    }


    //// ASSEMBLY ROBIN MULT TIMES OLD STEP ////////////////////////////////////
    {
        combustion::Assemble::L2::RobinMatrixMult::
        Assembler<dim> assemble_robin(
            primal.b,
            slab->high->dof,
            slab->high->fe,
            slab->high->constraints
        );
        assemble_robin.set_parameters(
            parameter.robin_factor_theta,
            parameter.robin_factor_Y
        );

        assemble_robin.assemble(-slab->tau_n(),u,parameter_set->robin_assembler_n_quadrature_points);
    }
}
template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
primal_construct_rhs(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
){
    Assert(primal.b.use_count(), dealii::ExcNotInitialized());
    Assert(primal.Mum.use_count(), dealii::ExcNotInitialized());
    Assert(primal.A.use_count(), dealii::ExcNotInitialized());
    primal.b->add(1.0,*primal.Mum);

    //nonlin rhs part -tau_n w(u_n)
    // negative sign is so far inside assembly
    primal.b->add(slab->tau_n(),*primal.f_Arr);

    if ( parameter_set->fe.primal.high_order){
        slab->high->constraints->distribute(*primal.b);
    } else {
        slab->low->constraints->distribute(*primal.b);
    }

}


template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
primal_solve_slab_newton_problem(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
    const double t0
) {
    ////////////////////////////////////////////////////////////////////////////
    // set initial value of Newton solver to previous timestep solution
    *primal.un = *primal.um;

    ////////////////////////////////////////////////////////////////////////////
    // apply inhomogeneous Dirichlet boundary values to initial solution
    //

    DTM::pout << "pu-dwr-combustion: dealii::MatrixTools::apply_boundary_values...\n";
    std::map<dealii::types::global_dof_index, double> boundary_values;

    Assert(function.u_D.use_count(), dealii::ExcNotInitialized());
    function.u_D->set_time(t0);

    if ( parameter_set->fe.primal.high_order){
        dealii::VectorTools::interpolate_boundary_values(
            *slab->high->dof,
            static_cast< dealii::types::boundary_id > (
                            combustion::types::boundary_id::Dirichlet
            ),
            *function.u_D,
            boundary_values
        );
    }else {
        dealii::VectorTools::interpolate_boundary_values(
            *slab->low->dof,
            static_cast< dealii::types::boundary_id > (
                            combustion::types::boundary_id::Dirichlet
            ),
            *function.u_D,
            boundary_values
        );
    }

    std::pair<unsigned int, unsigned int> range;
    range = primal.un->local_range();
    for ( std::map<unsigned int, double>::const_iterator i = boundary_values.begin(); i!= boundary_values.end() ;i++ )
        if ( i -> first >= range.first && i -> first < range.second)
            primal.un->operator()(i->first) = i->second;

    primal.un->compress(dealii::VectorOperation::insert);

    *u->x[0] = *primal.un;

    DTM::pout << " (done)" << std::endl;


    ////////////////////////////////////////////////////////////////////////////
    // Calculation/construction of initial residual
    DTM::pout << "pu-dwr-combustion: construct initial rhs vector...\n";

    timer->enter_subsection("nonlin ass primal");
    if(parameter_set->fe.primal.high_order){
        primal_assemble_rhs_nonlin_part_high_order(slab,u->x[0],t0);
    } else {
        primal_assemble_rhs_nonlin_part_low_order(slab,u->x[0],t0);
    }

    timer->leave_subsection();


    timer->enter_subsection("construct rhs primal");
        primal_construct_rhs(slab);
    timer->leave_subsection();
    DTM::pout << " (done)" << std::endl;

    DTM::pout << "pu-dwr-combustion: starting Newton loop...\n";
    DTM::pout << "It.\tResidual\tReduction\tRebuild\tLSrch\t#LinIts\n";
    double newton_residual = primal.b->linfty_norm();
    double old_newton_residual = newton_residual;
    double new_newton_residual;

    unsigned int newton_step = 1;
    unsigned int line_search_step;
    unsigned int n_linear_it = 0;

    DTM::pout << std::setprecision(5) << "0\t" << newton_residual << std::endl;

    int n_lin = 1000;
    dealii::SolverControl sc(n_lin,1.0e-10,false,false);
    // if Trilinos was not installed with MUMPS, set this to Amesos_Klu or Amesos_Superludist
    dealii::TrilinosWrappers::SolverDirect::AdditionalData ad(false,"Amesos_Mumps");

    auto iA = std::make_shared<dealii::TrilinosWrappers::SolverDirect> (sc,ad);

    while ( newton_residual > newton.lower_bound && newton_step < newton.max_steps )
    {

        old_newton_residual = newton_residual;
        timer->enter_subsection("nonlin. ass primal");
        *u->x[0] = *primal.un;
        if(parameter_set->fe.primal.high_order){
            primal_assemble_rhs_nonlin_part_high_order(slab,u->x[0],t0);
        } else {
            primal_assemble_rhs_nonlin_part_low_order(slab,u->x[0],t0);
        }
        timer->leave_subsection();

        timer->enter_subsection("construct rhs primal");
        primal_construct_rhs(slab);
        timer->leave_subsection();

        newton_residual = primal.b->linfty_norm();

        if (newton_residual < newton.lower_bound )
        {
           DTM::pout << "res\t" << newton_residual << std::endl;
           break;
        }

        if ( newton_residual/old_newton_residual > newton.rebuild )
        {

            timer->enter_subsection("nonlin. ass primal");

            if(parameter_set->fe.primal.high_order){
                primal_assemble_system_high_order(slab,u);
            } else {
                primal_assemble_system_low_order(slab,u);
            }
            timer->leave_subsection();

            timer->enter_subsection("factorize primal");
            iA = nullptr;
            iA = std::make_shared<dealii::TrilinosWrappers::SolverDirect> (sc,ad);
            iA->initialize(*primal.A);
            timer->leave_subsection();

        }
        timer->enter_subsection("solve primal");

        iA->solve(*primal.du,*primal.b);

        if ( parameter_set->fe.primal.high_order){
            slab->high->constraints->distribute(
                *primal.du
            );
        } else {
            slab->low->constraints->distribute(
                *primal.du
            );
        }
        timer->leave_subsection();

        for ( line_search_step = 0; line_search_step < newton.line_search_steps ;
              line_search_step++ ){
            *primal.un += *primal.du;
            *u->x[0] = *primal.un;

            timer->enter_subsection("nonlin. ass primal");
            if(parameter_set->fe.primal.high_order){
                primal_assemble_rhs_nonlin_part_high_order(slab,u->x[0],t0);
            } else {
                primal_assemble_rhs_nonlin_part_low_order(slab,u->x[0],t0);
            }
            timer->leave_subsection();

            timer->enter_subsection("construct rhs primal");
                primal_construct_rhs(slab);
            timer->leave_subsection();

            new_newton_residual = primal.b->linfty_norm();

            if ( new_newton_residual < newton_residual )
                break;
            else
                *primal.un -= *primal.du;

            *primal.du *= newton.line_search_damping;
        }
        DTM::pout << std::setprecision(5) << newton_step << "\t"
        << std::scientific << newton_residual << "\t"
        << std::scientific << newton_residual/old_newton_residual << "\t" ;

        if ( newton_residual/old_newton_residual > newton.rebuild )
            DTM::pout << "r\t";
        else
            DTM::pout << " \t";

        DTM::pout << line_search_step << "\t" << std::scientific
                  << n_linear_it << "\t" << std::scientific
                  << std::endl;

        newton_step++;

    }

    iA = nullptr;
    *u->x[0] = *primal.un;
    DTM::pout << " (done)" << std::endl;

}


template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
primal_do_forward_TMS() {
    ////////////////////////////////////////////////////////////////////////////
    // prepare time marching scheme (TMS) loop
    //

    ////////////////////////////////////////////////////////////////////////////
    // grid: init slab iterator to first space-time slab: Omega x I_1
    //
    timer->enter_subsection("prep TMS primal");
    Assert(grid.use_count(), dealii::ExcNotInitialized());
    Assert(grid->slabs.size(), dealii::ExcNotInitialized());
    auto slab = grid->slabs.begin();

    ////////////////////////////////////////////////////////////////////////////
    // storage: init iterators to storage_data_trilinos_vectors
    //          corresponding to first space-time slab: Omega x I_1
    //

    Assert(primal.storage.u.use_count(), dealii::ExcNotInitialized());
    Assert(primal.storage.u->size(), dealii::ExcNotInitialized());
    auto u = primal.storage.u->begin();

    ////////////////////////////////////////////////////////////////////////////
    // interpolate (or project) initial value(s)
    //
    function.u_0->set_time(slab->t_m);

    // init error computations (for global L2(L2) goal functional)
    primal_init_error_computations();
    ////////////////////////////////////////////////////////////////////////////
    // do TMS loop
    //

    DTM::pout
        << std::endl
        << "*******************************************************************"
        << "*************" << std::endl
        << "primal: solving forward TMS problem..." << std::endl
        << std::endl;

    unsigned int n{1};
    timer->leave_subsection();
    while (slab != grid->slabs.end()) {
        timer->enter_subsection("prep slab primal");
        // local time variables: \f$ t0 \in I_n = (t_m, t_n) \f$
        const double tm = slab->t_m;

        double t0{0};
        if (parameter_set->fe.primal.time_type_support_points.compare("Gauss-Radau")==0) {
            t0 = slab->t_n;
        }
        else if (parameter_set->fe.primal.time_type_support_points.compare("Gauss")==0) {
            t0 = tm + slab->tau_n()/2.;
        }

        const double tn = slab->t_n;

        DTM::pout
            << "primal: solving problem on "
            << "I_" << n << " = (" << tm << ", " << tn << ") "
            << std::endl;

        if ( parameter_set->fe.primal.high_order){
            primal_setup_slab_high_order(slab, u);
        }else{
            primal_setup_slab_low_order(slab, u);
        }

        timer->leave_subsection();
        primal_solve_slab_newton_problem(slab,u,tm);

        ////////////////////////////////////////////////////////////////////////
        // do postprocessings on the solution
        //
        timer->enter_subsection("error comp");

        primal.u_tmp = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();
        primal.u_tmp ->reinit(*slab->low->locally_owned_dofs,mpi_comm);
        *primal.u_tmp = 0;
        if ( parameter_set->fe.primal.high_order){
            dealii::FETools::interpolate(
                //dual solution
                *slab->high->dof,
                *u->x[0],
                //primal solution
                *slab->low->dof,
                *slab->low->constraints,
                *primal.u_tmp
            );
        }
        else{
            primal.u_tmp->equ(1.0,*u->x[0]);
        }
        // do error computations ( for global L2(L2) goal )
        if ( parameter_set->dwr.goal.type.compare("average")==0){
            DTM::pout << "primal: calculating goal functionals...";
            const dealii::QGauss<dim> quadrature_formula(5);
            const dealii::QGauss<dim-1> face_quadrature_formula(5);

        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_face_q_points = face_quadrature_formula.size();

        std::vector< dealii::Vector<double> > solution_values(
                                         n_q_points,
                                         dealii::Vector<double> (2) );

        std::vector< dealii::Vector<double> > face_solution_values(
                                                 n_face_q_points,
                                                 dealii::Vector<double> (2) );

        double mean_rsc = 0.;
        double mean_rr = 0.;


        if ( parameter_set->fe.primal.high_order) {


            if ( parameter_set->dwr.goal.calc_reaction_rate){
                primal_assemble_rhs_nonlin_part_high_order(slab,u->x[0],t0);
                *primal.relevant_tmp = *primal.f_Arr;
            }
            dealii::FEValues<dim> fe_values(*slab->high->fe,
                                            quadrature_formula,
                                            dealii::update_values |
                                            dealii::update_JxW_values);

            dealii::FEFaceValues<dim> fe_face_values(*slab->high->fe,
                                                    face_quadrature_formula,
                                                    dealii::update_values |
                                                    dealii::update_JxW_values);

            typename dealii::DoFHandler<dim>::active_cell_iterator
            cell = slab->high->dof->begin_active(),
            endc = slab->high->dof->end();

            for ( ; cell != endc ; ++cell)
                if (cell->is_locally_owned())
                {
                    if ( parameter_set->dwr.goal.calc_reaction_rate)
                    {
                        fe_values.reinit(cell);
                        fe_values.get_function_values(*primal.relevant_tmp,solution_values);
                        for ( unsigned int q = 0 ; q < n_q_points ; q++){
                            mean_rr += fe_values.JxW(q)*solution_values[q](0); //
                        }
                    }

                    if ( parameter_set->dwr.goal.calc_rod_species_concentration)
                    {
                        if ( cell->at_boundary() ){
                            for ( unsigned int face_no = 0 ; face_no < dealii::GeometryInfo<dim>::faces_per_cell ; face_no ++)
                            if ( (cell ->face(face_no)->at_boundary() && cell->face(face_no)->boundary_id() ==
                                static_cast<dealii::types::boundary_id> (
                                    combustion::types::boundary_id::Robin)))
                            {
                                fe_face_values.reinit(cell,face_no);
                                fe_face_values.get_function_values(*u->x[0],face_solution_values);
                                for ( unsigned int q = 0 ; q < n_face_q_points ; q++ )
                                {
                                    mean_rsc += fe_face_values.JxW(q)*face_solution_values[q](1); // Y
                                }
                            }
                        }
                    }
                }

            }  else {
                if ( parameter_set->dwr.goal.calc_reaction_rate){
                    primal_assemble_rhs_nonlin_part_low_order(slab,u->x[0],t0);
                    *primal.relevant_tmp = *primal.f_Arr;
                }

                dealii::FEValues<dim> fe_values(*slab->low->fe,
                                                quadrature_formula,
                                                dealii::update_values |
                                                dealii::update_JxW_values
                );

                dealii::FEFaceValues<dim> fe_face_values(*slab->low->fe,
                                                        face_quadrature_formula,
                                                        dealii::update_values |
                                                        dealii::update_JxW_values
                );

                typename dealii::DoFHandler<dim>::active_cell_iterator
                cell = slab->low->dof->begin_active(),
                endc = slab->low->dof->end();

                for ( ; cell != endc ; ++cell)
                if (cell->is_locally_owned())
                {
                    if ( parameter_set->dwr.goal.calc_reaction_rate)
                    {
                        fe_values.reinit(cell);
                        fe_values.get_function_values(*primal.relevant_tmp,solution_values);
                        for ( unsigned int q = 0 ; q < n_q_points ; q++){
                            mean_rr += fe_values.JxW(q)*solution_values[q](0); //
                        }
                    }

                    if ( parameter_set->dwr.goal.calc_rod_species_concentration)
                    {
                        if ( cell->at_boundary() ){
                            for ( unsigned int face_no = 0 ; face_no < dealii::GeometryInfo<dim>::faces_per_cell ; face_no ++)
                            if ( (cell ->face(face_no)->at_boundary() && cell->face(face_no)->boundary_id() ==
                                static_cast<dealii::types::boundary_id> (
                                    combustion::types::boundary_id::Robin)))
                            {
                                fe_face_values.reinit(cell,face_no);
                                fe_face_values.get_function_values(*u->x[0],face_solution_values);
                                for ( unsigned int q = 0 ; q < n_face_q_points ; q++ )
                                {
                                    mean_rsc += fe_face_values.JxW(q)*face_solution_values[q](1); // Y
                                }
                            }
                        }
                    }
                }
            }


            primal.functionals.average.rod_species_concentration +=
                slab->tau_n()*dealii::Utilities::MPI::sum(mean_rsc,mpi_comm)
                    /(cardinality.rod*parameter_set->T);


            primal.functionals.average.reaction_rate +=
                slab->tau_n()*dealii::Utilities::MPI::sum(mean_rr,mpi_comm)
                    /(cardinality.domain*parameter_set->T);


            DTM::pout << " (done)" << std::endl;
    }
    timer->leave_subsection();

    timer->enter_subsection("prep slab primal");

    ////////////////////////////////////////////////////////////////////////
    // prepare next I_n slab problem:
    //

    ++n;
    ++slab;
    ++u;

    ////////////////////////////////////////////////////////////////////////
    // allow garbage collector to clean up memory
    //

    primal.f0 = nullptr;
    primal.f_Arr = nullptr;

    primal.A->clear();
    primal.A = nullptr;
    primal.b = nullptr;
    primal.du = nullptr;
    primal.b_tmp = nullptr;


    DTM::pout << std::endl;
    timer->leave_subsection();
    }

    DTM::pout
            << "primal: forward TMS problem done" << std::endl
            << "*******************************************************************"
            << "*************" << std::endl
            << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // allow garbage collector to clean up memory
    //

    primal.um = nullptr;
    primal.un = nullptr;

    ////////////////////////////////////////////////////////////////////////////
    // finish error computation ( for global L2(L2) goal functional )
    //
    DTM::pout
    << "*******************************************************************"
    << "*************";
    timer->enter_subsection("error comp");
    if ( parameter_set->dwr.goal.type.compare("average")==0){
        DTM::pout
        << "\n time average goal functionals:"
        << "\n\t reaction rate: \t\t"
        << std::setprecision(8) << primal.functionals.average.reaction_rate
        << "\n\t rod species concentration: \t"
        << std::setprecision(8) << primal.functionals.average.rod_species_concentration
        << "\nprimal error for chosen functional ";

        if ( parameter_set->dwr.goal.functional.compare("reaction rate")==0 )
        {
            primal_error = std::abs(parameter_set->dwr.reference.average.reaction_rate
                                       -primal.functionals.average.reaction_rate);
            DTM::pout << "(reaction rate)\n"
                      << "\t" << primal_error << std::endl;
        } else if (parameter_set->dwr.goal.functional.
                                        compare("rod species concentration") == 0 )
        {
            primal_error = std::abs(parameter_set->dwr.reference.average.rod_species_concentration
                                            -primal.functionals.average.rod_species_concentration);
            DTM::pout << "(rod species concentration)\n"
                      << "\t" << primal_error << std::endl;
        }

    }
    timer->leave_subsection();
}


////////////////////////////////////////////////////////////////////////////////
// primal: L2(L2) error computation
//

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
primal_init_error_computations() {
	primal.functionals.average.reaction_rate = 0;
	primal.functionals.average.rod_species_concentration = 0;
}



////////////////////////////////////////////////////////////////////////////////
// primal data output
//

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
primal_init_data_output() {
    Assert(parameter_set.use_count(), dealii::ExcNotInitialized());

    // set up which dwr loop(s) are allowed to make data output:
    if ( !parameter_set->data_output.primal.dwr_loop.compare("none") ) {
        return;
    }

    // may output data: initialise (mode: all, last or specific dwr loop)
    DTM::pout
        << "primal solution data output: patches = "
        << parameter_set->data_output.primal.patches
        << std::endl;

    std::vector<std::string> data_field_names;
    data_field_names.push_back("theta");
    data_field_names.push_back("Y");


    std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation > dci_field;
    dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar); //theta
    dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar); //Y

    primal.data_output = std::make_shared< DTM::DataOutput<dim> >();
    primal.data_output->set_data_field_names(data_field_names);
    primal.data_output->set_data_component_interpretation_field(dci_field);

    primal.data_output->set_data_output_patches(
        parameter_set->data_output.primal.patches
    );

    // check if we use a fixed trigger interval, or, do output once on a I_n
    if ( !parameter_set->data_output.primal.trigger_type.compare("fixed") ) {
        primal.data_output_trigger_type_fixed = true;
    }
    else {
        primal.data_output_trigger_type_fixed = false;
    }

    // only for fixed
    primal.data_output_trigger = parameter_set->data_output.primal.trigger;

    if (primal.data_output_trigger_type_fixed) {
        DTM::pout
            << "primal solution data output: using fixed mode with trigger = "
            << primal.data_output_trigger
            << std::endl;
    }
    else {
        DTM::pout
            << "primal solution data output: using I_n mode (trigger adapts to I_n automatically)"
            << std::endl;
    }

    primal.data_output_time_value = parameter_set->t0;
}


template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
primal_do_data_output_on_slab(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
    const unsigned int dwr_loop,
    const bool dG_initial_value) {

    if (primal.data_output_trigger <= 0) return;

    auto u_trigger = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

    if ( parameter_set->fe.primal.high_order){
        u_trigger->reinit(
            *slab->high->locally_owned_dofs,
            *slab->high->locally_relevant_dofs,
            mpi_comm
        );
        primal.data_output->set_DoF_data(
            slab->high->dof
        );
    }else{
        u_trigger->reinit(
            *slab->low->locally_owned_dofs,
            *slab->low->locally_relevant_dofs,
            mpi_comm
        );
        primal.data_output->set_DoF_data(
            slab->low->dof
        );
    }

	std::ostringstream filename;
	filename
            << "solution-dwr_loop-"
            << std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop+1;

	double &t{primal.data_output_time_value};


	if (dG_initial_value) {
            // NOTE: for dG-in-time discretisations the initial value function
            //       does not belong to the set of dof's. Thus, we need a special
            //       implementation here to output "primal.um".

            *u_trigger=*u->x[0]; // NOTE: this must be primal.um!

            primal.data_output->write_data(
                filename.str(),
                u_trigger,
                t
            );

            t += primal.data_output_trigger;
	}
	else {
            // adapt trigger value for I_n output mode
            if (!primal.data_output_trigger_type_fixed) {
                primal.data_output_trigger = slab->tau_n();
                primal.data_output_time_value = slab->t_n;
            }

            for ( ; t <= slab->t_n; t += primal.data_output_trigger) {
                // evalute space-time solution
                *u_trigger=*u->x[0];

                primal.data_output->write_data(
                    filename.str(),
                    u_trigger,
                    t
                );
            }
	}

	// check if data for t=T was written
	if (std::next(slab) == grid->slabs.end()) {
	if (primal.data_output_trigger_type_fixed) {
            const double overshoot_tol{
                    std::min(slab->tau_n(), primal.data_output_trigger) * 1e-7
            };

            if ((t > slab->t_n) && (std::abs(t - slab->t_n) < overshoot_tol)) {
                // overshoot of time variable; manually set to t = T and do data output
                t = slab->t_n;

                // evalute space-time solution
                *u_trigger=1.0, *u->x[0];

                primal.data_output->write_data(
                    filename.str(),
                    u_trigger,
                    t
                );
            }
	}}
}


template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
primal_do_data_output(
    const unsigned int dwr_loop,
    bool last
) {
    // set up which dwr loop(s) are allowed to make data output:
    Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
    if ( !parameter_set->data_output.primal.dwr_loop.compare("none") ) {
        return;
    }

    if (!parameter_set->data_output.primal.dwr_loop.compare("last")) {
        // output only the last (final) dwr loop
        if (last) {
                primal.data_output_dwr_loop = dwr_loop;
        }
        else {
                return;
        }
    }
    else {
        if (!parameter_set->data_output.primal.dwr_loop.compare("all")) {
            // output all dwr loops
            if (!last) {
                primal.data_output_dwr_loop = dwr_loop;
            }
            else {
                return;
            }
        }
        else {
            // output on a specific dwr loop
            if (!last) {
                primal.data_output_dwr_loop =
                    std::stoi(parameter_set->data_output.primal.dwr_loop)-1;
            }
            else {
                    return;
            }
        }

    }

    if (primal.data_output_dwr_loop < 0)
        return;

    if ( static_cast<unsigned int>(primal.data_output_dwr_loop) != dwr_loop )
        return;

    DTM::pout
        << "primal solution data output: dwr loop = "
        << primal.data_output_dwr_loop
        << std::endl;

    primal.data_output_time_value = parameter_set->t0;

    Assert(grid->slabs.size(), dealii::ExcNotInitialized());
    auto slab = grid->slabs.begin();
    auto u = primal.storage.u->begin();

    // primal: dG: additionally output interpolated u_0(t0)
    {
        // n == 1: initial value function u_0
        primal.um = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
        primal.un = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
        function.u_0->set_time(slab->t_m);
        Assert(primal.um.use_count(), dealii::ExcNotInitialized());

        if ( parameter_set->fe.primal.high_order){
            primal.um->reinit( *slab->high->locally_owned_dofs );
            primal.un->reinit( *slab->high->locally_owned_dofs );

            dealii::VectorTools::interpolate(
                    *slab->high->dof,
                    *function.u_0,
                    *primal.um
            );

            slab->high->constraints->distribute(*primal.um);
        }
        else{
            primal.um->reinit( *slab->low->locally_owned_dofs );
            primal.un->reinit( *slab->low->locally_owned_dofs );

            dealii::VectorTools::interpolate(
                    *slab->low->dof,
                    *function.u_0,
                    *primal.um
            );

            slab->low->constraints->distribute(*primal.um);
        }
        // output "initial value solution" at initial time t0
        *primal.un = *u->x[0];
        *u->x[0] = *primal.um;
        primal_do_data_output_on_slab(slab,u,dwr_loop,true);
        *u->x[0] = *primal.un;
    }

    while (slab != grid->slabs.end()) {
        primal_do_data_output_on_slab(slab,u,dwr_loop,false);

        ++slab;
        ++u;
    }
}


////////////////////////////////////////////////////////////////////////////////
// dual problem
//

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
dual_reinit_storage() {
    ////////////////////////////////////////////////////////////////////////////
    // init storage containers for vector data:
    // NOTE: * dual space: time cG(1) method (having 2 independent solutions)
    //       * dual solution dof vectors: z
    //

    Assert(grid.use_count(), dealii::ExcNotInitialized());
    // get number of time steps N
    const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};

    dual.storage.z = std::make_shared< DTM::types::storage_data_trilinos_vectors<1> > ();
    dual.storage.z->resize(N);

    {
        auto slab = grid->slabs.begin();
        for (auto &element : *dual.storage.z) {
            for (unsigned int j{0}; j < element.x.size(); ++j) {
                element.x[j] = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

                Assert(slab != grid->slabs.end(), dealii::ExcInternalError());

                if ( parameter_set->fe.dual.high_order){
                    Assert(slab->high->dof.use_count(), dealii::ExcNotInitialized());
                    Assert(slab->high->locally_owned_dofs.use_count(),dealii::ExcNotInitialized());

                    // initialize
                    element.x[j]->reinit(*slab->high->locally_owned_dofs,
                                         *slab->high->locally_relevant_dofs,
                                         mpi_comm);

                }else {
                    Assert(slab->low->dof.use_count(),dealii::ExcNotInitialized());
                    Assert(slab->low->locally_owned_dofs.use_count(),dealii::ExcNotInitialized());

                    element.x[j]->reinit(*slab->low->locally_owned_dofs,
                                         *slab->low->locally_relevant_dofs,
                                         mpi_comm);
                }
            }
            ++slab;
        }
    }
}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
dual_setup_slab_high_order(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &z,
    bool mesh_interpolate
) {

    Assert(slab->high->locally_owned_dofs.use_count(),dealii::ExcNotInitialized());
    Assert((slab != grid->slabs.end()), dealii::ExcInternalError());

    dual.zm = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    dual.zm->reinit(*slab->high->locally_owned_dofs);

    dual.Mzm = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    dual.Mzm->reinit(*slab->high->locally_owned_dofs);

    dual.owned_tmp = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    dual.owned_tmp->reinit(*slab->high->locally_owned_dofs);

    dual.relevant_tmp = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    dual.relevant_tmp->reinit(*slab->high->locally_owned_dofs,
                              *slab->high->locally_relevant_dofs,
                              mpi_comm);

    dual.zn = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    dual.zn->reinit(*slab->high->locally_owned_dofs);

    dual.b = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    dual.b->reinit(*slab->high->locally_owned_dofs);

    //Setup sparsity pattern and matrix
    {
        dealii::Table<2,dealii::DoFTools::Coupling> coupling(2,2);
        coupling[0][0] = dealii::DoFTools::always;
        coupling[1][1] = dealii::DoFTools::always;
        coupling[1][0] = dealii::DoFTools::always;
        coupling[0][1] = dealii::DoFTools::always;

        dealii::DynamicSparsityPattern dsp(slab->high->dof->n_dofs());

        dealii::DoFTools::make_sparsity_pattern(
                        *slab->high->dof,
                        coupling,
                        dsp,
                        *slab->high->constraints,
                        false // keep constrained dofs?
                        ,dealii::Utilities::MPI::this_mpi_process(mpi_comm)
        );

        dealii::SparsityTools::distribute_sparsity_pattern(
                        dsp,
                        *slab->high->locally_owned_dofs,
                        mpi_comm,
                        *slab->high->locally_relevant_dofs);

        dual.A = std::make_shared< dealii::TrilinosWrappers::SparseMatrix >();
        dual.A->reinit(*slab->high->locally_owned_dofs,*slab->high->locally_owned_dofs,dsp);

    }

    if ( mesh_interpolate){
        dealii::VectorTools::interpolate_to_different_mesh(
            // solution on I_{n+1}:
            *std::next(slab)->high->dof,
            *std::next(z)->x[0],
            // solution on I_n:
            *slab->high->dof,
            *slab->high->constraints,
            *dual.zm
        );
    }
}


template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
dual_setup_slab_low_order(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &z,
    bool mesh_interpolate
) {
    Assert(slab->low->locally_owned_dofs.use_count(),dealii::ExcNotInitialized());
    Assert((slab != grid->slabs.end()), dealii::ExcInternalError());

    dual.zm = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    dual.zm->reinit(*slab->low->locally_owned_dofs,mpi_comm);

    dual.Mzm = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    dual.Mzm->reinit(*slab->low->locally_owned_dofs,mpi_comm);


    dual.owned_tmp = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    dual.owned_tmp->reinit(*slab->low->locally_owned_dofs,mpi_comm);

    dual.relevant_tmp = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    dual.relevant_tmp->reinit(*slab->low->locally_owned_dofs,
                              *slab->low->locally_relevant_dofs,
                              mpi_comm);

    dual.zn = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    dual.zn->reinit(*slab->low->locally_owned_dofs,mpi_comm);

    dual.b = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
    dual.b->reinit(*slab->low->locally_owned_dofs,mpi_comm);

    //Setup sparsity pattern and matrix
    {
        dealii::Table<2,dealii::DoFTools::Coupling> coupling(2,2);
        coupling[0][0] = dealii::DoFTools::always;
        coupling[1][1] = dealii::DoFTools::always;
        coupling[1][0] = dealii::DoFTools::always;
        coupling[0][1] = dealii::DoFTools::always;

        dealii::DynamicSparsityPattern dsp(slab->low->dof->n_dofs());

        dealii::DoFTools::make_sparsity_pattern(
            *slab->low->dof,
            coupling,
            dsp,
            *slab->low->constraints,
            false // keep constrained dofs?
            ,dealii::Utilities::MPI::this_mpi_process(mpi_comm)
        );

        dealii::SparsityTools::distribute_sparsity_pattern(
            dsp,
            *slab->low->locally_owned_dofs,
            mpi_comm,
            *slab->low->locally_relevant_dofs
        );

        dual.A = std::make_shared< dealii::TrilinosWrappers::SparseMatrix >();
        dual.A->reinit(*slab->low->locally_owned_dofs,*slab->low->locally_owned_dofs,dsp);

    }

    if ( mesh_interpolate){
        dealii::VectorTools::interpolate_to_different_mesh(
            // solution on I_{n+1}:
            *std::next(slab)->low->dof,
            *std::next(z)->x[0],
            // solution on I_n:
            *slab->low->dof,
            *slab->low->constraints,
            *dual.zm
        );
    }
}


template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
dual_assemble_system_lin_part_high_order(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
) {

    // ASSEMBLY MASS MATRIX ////////////////////////////////////////////////////
    {
        combustion::Assemble::L2::Mass::
        Assembler<dim> assemble_mass(
            dual.A,
            slab->high->dof,
            slab->high->fe,
            slab->high->constraints
        );

        DTM::pout << "pu-dwr-combustion: assemble mass matrix...";
        assemble_mass.assemble(1.0);
        DTM::pout << " (done)" << std::endl;
    }
    dual.A->vmult(*dual.Mzm,*dual.zm);

    // ASSEMBLY STIFFNESS MATRIX ///////////////////////////////////////////////
    {
        combustion::Assemble::L2::Laplace::
        Assembler<dim> assemble_stiffness_cell_terms (
            dual.A,
            slab->high->dof,
            slab->high->fe,
            slab->high->constraints
        );

        DTM::pout << "pu-dwr-combustion: assemble cell stiffness matrix...";
        assemble_stiffness_cell_terms.assemble(slab->tau_n());
        DTM::pout << " (done)" << std::endl;
    }

    // ASSEMBLY ROBIN MATRIX PART ////////////////////////////////////////////////
    {
        combustion::Assemble::L2::RobinMatrix::
        Assembler<dim> assemble_robin_cell_terms (
            dual.A,
            slab->high->dof,
            slab->high->fe,
            slab->high->constraints
        );

        assemble_robin_cell_terms.set_parameters(
            parameter.robin_factor_theta,
            parameter.robin_factor_Y
        );
        DTM::pout << "pu-dwr-combustion: assemble cell Robin matrix...";
        assemble_robin_cell_terms.assemble(slab->tau_n(),parameter_set->robin_assembler_n_quadrature_points);
        DTM::pout << " (done)" << std::endl;
    }
}


template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
dual_assemble_system_lin_part_low_order(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
) {

    // ASSEMBLY MASS MATRIX ////////////////////////////////////////////////////
    {
        combustion::Assemble::L2::Mass::
        Assembler<dim> assemble_mass(
            dual.A,
            slab->low->dof,
            slab->low->fe,
            slab->low->constraints
        );

        DTM::pout << "pu-dwr-combustion: assemble mass matrix...";
        assemble_mass.assemble(1.0);
        DTM::pout << " (done)" << std::endl;
    }
    dual.A->vmult(*dual.Mzm,*dual.zm);

    // ASSEMBLY STIFFNESS MATRIX ///////////////////////////////////////////////
    {
        combustion::Assemble::L2::Laplace::
        Assembler<dim> assemble_stiffness_cell_terms (
                dual.A,
                slab->low->dof,
                slab->low->fe,
                slab->low->constraints
        );

        DTM::pout << "pu-dwr-combustion: assemble cell stiffness matrix...";
        assemble_stiffness_cell_terms.assemble(slab->tau_n());
        DTM::pout << " (done)" << std::endl;
    }

    // ASSEMBLY ROBIN MATRIX PART ////////////////////////////////////////////////
    {
        combustion::Assemble::L2::RobinMatrix::
        Assembler<dim> assemble_robin_cell_terms (
            dual.A,
            slab->low->dof,
            slab->low->fe,
            slab->low->constraints
        );

        assemble_robin_cell_terms.set_parameters(
            parameter.robin_factor_theta,
            parameter.robin_factor_Y
        );
        DTM::pout << "pu-dwr-combustion: assemble cell Robin matrix...";
        assemble_robin_cell_terms.assemble(slab->tau_n(),parameter_set->robin_assembler_n_quadrature_points);
        DTM::pout << " (done)" << std::endl;
    }
}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
dual_assemble_system_nonlin_part_high_order(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
    [[maybe_unused]] const unsigned int &n,
    [[maybe_unused]] const double &t0
){
    /////////////////////////////
    // u_h(t1) = u_h(t1)|_{I_{n}}
    //

    // interpolate primal solution u_h(t1) to dual solution space
    dual.u1 = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
    dual.u1->reinit( *slab->high->locally_owned_dofs, *slab->high->locally_relevant_dofs, mpi_comm );

    if( parameter_set->fe.primal.high_order)
    {
        *dual.u1=*u->x[0];
    }else {
        dealii::TrilinosWrappers::MPI::Vector owned_u(*slab->high->locally_owned_dofs,mpi_comm);

        dealii::FETools::interpolate(
            *slab->low->dof,
            *u->x[0],
            *slab->high->dof,
            *slab->high->constraints,
            owned_u
        );

        *dual.u1 = owned_u;
    }
    // ASSEMBLY ARRHENIUS DIRECTIONAL DERIVATIVE ///////////////////////////////////
    {
        combustion::Assemble::L2::Arrhenius::Deriv::
        Assembler<dim> assemble_Arrhenius(
            dual.A,
            slab->high->dof,
            slab->high->fe,
            slab->high->constraints,
            true
        );

        assemble_Arrhenius.set_parameters(
            parameter.alpha,
            parameter.beta,
            parameter.Le
        );

        assemble_Arrhenius.assemble(slab->tau_n(),dual.u1);
    }

}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
dual_assemble_system_nonlin_part_low_order(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
    [[maybe_unused]] const unsigned int &n,
    [[maybe_unused]] const double &t0
){
    //special case as low order dual solution comes only with low order space solution in equal low order
    // ASSEMBLY ARRHENIUS DIRECTIONAL DERIVATIVE ///////////////////////////////////
    {
        combustion::Assemble::L2::Arrhenius::Deriv::
        Assembler<dim> assemble_Arrhenius(
            dual.A,
            slab->low->dof,
            slab->low->fe,
            slab->low->constraints,
            true
        );
        assemble_Arrhenius.set_parameters(
            parameter.alpha,
            parameter.beta,
            parameter.Le
        );
        assemble_Arrhenius.assemble(slab->tau_n(),u->x[0]);
    }

}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
dual_assemble_rhs_functionals_high_order(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    [[maybe_unused]] const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
    [[maybe_unused]] const unsigned int &n,
    [[maybe_unused]] const double &t0,
    [[maybe_unused]] const double &t1
) {

    dual.Je1 = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
    dual.Je1->reinit( *slab->high->locally_owned_dofs);
    *dual.Je1 = 0;
    //J(u) = average(mean(omega(u)))
    if ( parameter_set -> dwr.goal.functional.compare("reaction rate") == 0)
    {
        ///////////////////////////////
        // u_h(t1) = u_h(t1)|_{I_{n}}
        //
        Assert( dual.u1.use_count(), dealii::ExcNotInitialized() );

        DTM::pout << "pu-dwr-combustion: assemble Je1...";

        auto assemble_Je = std::make_shared<
            combustion::Assemble::L2::Je_reaction_rate::Assembler<dim> >(
                slab->high->dof,
                slab->high->fe,
                slab->high->constraints
            );

        assemble_Je->set_parameters(
            parameter.alpha,
            parameter.beta,
            parameter.Le,
            cardinality.domain
        );
        assemble_Je->assemble(
            dual.Je1,
            dual.u1,
            0,   // n_q_points: 0 -> q+1 in auto mode
            true //auto mode
        );

        DTM::pout << " (done)" << std::endl;

        ////////////////////////////////////////////////////////////////////////
        // NOTE: for parameter_set->fe.dual.time_type_support_points.compare("Gauss")
        //       we have \f$ \beta_{1,1} = 0 \f$ and therefore we do not need
        //       to assemble anything for Je^1!
    }
    //J(u) = average(mean(Y|_{\gamma_R}))
    if ( parameter_set -> dwr.goal.functional.compare("rod species concentration") == 0)
    {
        auto assemble_Je = std::make_shared<
        combustion::Assemble::L2::Je_rod_species_concentration::Assembler<dim> >(
            slab->high->dof,
            slab->high->fe,
            slab->high->constraints
        );

        assemble_Je->set_parameters(
            cardinality.rod
        );

        DTM::pout << "pu-dwr-combustion: assemble Je1...";

        assemble_Je->assemble(
            dual.Je1,
            0,   // n_q_points: 0 -> q+1 in auto mode
            true //auto mode
        );

        DTM::pout << " (done)" << std::endl;
    }
}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
dual_assemble_rhs_functionals_low_order(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
    [[maybe_unused]] const unsigned int &n,
    [[maybe_unused]] const double &t0,
    [[maybe_unused]] const double &t1
) {
    dual.Je1 = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
    dual.Je1->reinit( *slab->low->locally_owned_dofs);
    *dual.Je1 = 0;
    //J(u) = average(mean(omega(u)))
    if ( parameter_set -> dwr.goal.functional.compare("reaction rate") == 0)
    {
        ///////////////////////////////
        // u_h(t1) = u_h(t1)|_{I_{n}}
        //

        DTM::pout << "pu-dwr-combustion: assemble Je1...";


        auto assemble_Je = std::make_shared<
        combustion::Assemble::L2::Je_reaction_rate::Assembler<dim> >(
            slab->low->dof,
            slab->low->fe,
            slab->low->constraints
        );

        assemble_Je->set_parameters(
            parameter.alpha,
            parameter.beta,
            parameter.Le,
            cardinality.domain
        );
        assemble_Je->assemble(
            dual.Je1,
            u->x[0],
            0,   // n_q_points: 0 -> q+1 in auto mode
            true //auto mode
        );


        DTM::pout << " (done)" << std::endl;

        ////////////////////////////////////////////////////////////////////////
        // NOTE: for parameter_set->fe.dual.time_type_support_points.compare("Gauss")
        //       we have \f$ \beta_{1,1} = 0 \f$ and therefore we do not need
        //       to assemble anything for Je^1!
    }
    //J(u) = average(mean(Y|_{\gamma_R}))
    if ( parameter_set -> dwr.goal.functional.compare("rod species concentration") == 0)
    {
        auto assemble_Je = std::make_shared<
            combustion::Assemble::L2::Je_rod_species_concentration::Assembler<dim> >(
                slab->low->dof,
                slab->low->fe,
                slab->low->constraints
        );

        assemble_Je->set_parameters(
            cardinality.rod
        );

        DTM::pout << "pu-dwr-combustion: assemble Je1...";

        assemble_Je->assemble(
            dual.Je1,
            0,   // n_q_points: 0 -> q+1 in auto mode
            true //auto mode
        );

        DTM::pout << " (done)" << std::endl;
    }
}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
dual_construct_rhs(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
){
    Assert(dual.b.use_count(), dealii::ExcNotInitialized());
    Assert(dual.Mzm.use_count(), dealii::ExcNotInitialized());

    *dual.b = *dual.Mzm;

    if ( parameter_set->fe.dual.time_type_support_points.compare("Gauss-Lobatto")==0 ){
        if ( parameter_set->dwr.goal.type.compare("average")==0)
        {
            dual.b->add(slab->tau_n() /parameter_set->T
                        ,*dual.Je1);
        }
    }
    else{
        AssertThrow(false, dealii::ExcNotImplemented());
    }


}


template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
dual_solve_slab_linear_problem(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &z
) {
    ////////////////////////////////////////////////////////////////////////////
    // solve linear system with direct solver
    //
    timer->enter_subsection("solve dual");
    DTM::pout << "pu-dwr-combustion: setup direct lss and solve...";
    int n_lin = 1000;
    dealii::SolverControl sc(n_lin,1.0e-12,false,false);
    dealii::TrilinosWrappers::SolverDirect::AdditionalData ad(false,"Amesos_Mumps");
    dealii::TrilinosWrappers::SolverDirect iA(sc,ad);
    iA.initialize(*dual.A);
    iA.solve(*dual.zn,*dual.b);
    DTM::pout << " (done)" << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // distribute hanging nodes constraints on solution
    //

    DTM::pout << "pu-dwr-combustion: dual.constraints->distribute...";
    if ( parameter_set->fe.dual.high_order){
        slab->high->constraints->distribute(
            *dual.zn
        );
    } else {
        slab->low->constraints->distribute(
            *dual.zn
        );
    }

    *z->x[0] = *dual.zn;
    DTM::pout << " (done)" << std::endl;
    timer->leave_subsection();
}


template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
dual_do_backward_TMS() {
    ////////////////////////////////////////////////////////////////////////////
    // prepare time marching scheme (TMS) loop
    //

    //////////////////////////////////////////////////////////////////////////
    // grid: init slab iterator to last space-time slab: Omega x I_N
    //
    timer->enter_subsection("prep TMS dual");
    Assert(grid.use_count(), dealii::ExcNotInitialized());
    Assert(grid->slabs.size(), dealii::ExcNotInitialized());
    auto slab = std::prev(grid->slabs.end());

    ////////////////////////////////////////////////////////////////////////////
    // storage: init iterators to storage_data_trilinos_vectors
    //          corresponding to last space-time slab: Omega x I_N
    //

    Assert(dual.storage.z.use_count(), dealii::ExcNotInitialized());
    Assert(dual.storage.z->size(), dealii::ExcNotInitialized());
    auto z = std::prev(dual.storage.z->end());


    Assert(primal.storage.u.use_count(), dealii::ExcNotInitialized());
    Assert(primal.storage.u->size(), dealii::ExcNotInitialized());
    auto u = std::prev(primal.storage.u->end());

    ////////////////////////////////////////////////////////////////////////////
    // do TMS loop
    //

    DTM::pout
        << std::endl
        << "*******************************************************************"
        << "*************" << std::endl
        << "dual: solving backward TMS problem..." << std::endl
        << std::endl;

    const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};
    unsigned int n{N};
    timer->leave_subsection();
    while (n) {
            // local time variables: \f$ t0, t1 \in I_n = (t_m, t_n) \f$
            timer->enter_subsection("prep slab dual");
            const double tm = slab->t_m;
            double t0{0};

            if (parameter_set->fe.dual.time_type_support_points.compare("Gauss-Lobatto")==0) {
                    t0 = slab->t_m;
            }

            const double t1 = slab->t_n;
            const double tn = slab->t_n;

            DTM::pout
                << "dual: solving problem on "
                << "I_" << n << " = (" << tm << ", " << tn << ") "
                << std::endl;


            if (parameter_set->fe.dual.high_order)
            {
                dual_setup_slab_high_order(slab, z, (n < N) );
                timer->leave_subsection("prep slab dual");

                timer->enter_subsection("nonlin. ass dual");
                    dual_assemble_system_nonlin_part_high_order(slab,u,n,t0);
                timer->leave_subsection();

                // assemble slab problem
                timer->enter_subsection("lin. ass dual");
                    dual_assemble_system_lin_part_high_order(slab);
                    dual_assemble_rhs_functionals_high_order(slab,u,n,t0,t1);
                timer->leave_subsection();


            }
            else {
                dual_setup_slab_low_order(slab, z , (n < N));
                timer->leave_subsection("prep slab dual");


                // assemble slab problem
                timer->enter_subsection("lin. ass dual");
                    dual_assemble_system_lin_part_low_order(slab);
                    dual_assemble_rhs_functionals_low_order(slab,u,n,t0,t1);
                timer->leave_subsection();

                timer->enter_subsection("nonlin. ass dual");
                    dual_assemble_system_nonlin_part_low_order(slab,u,n,t0);
                timer->leave_subsection();
            }

            timer->enter_subsection("construct rhs dual");
                dual_construct_rhs(slab);
            timer->leave_subsection();
            // solve slab problem (i.e. apply boundary values and solve for z0)

            dual_solve_slab_linear_problem(slab,z);

            timer->enter_subsection("prep slab dual");
            --n;
            --slab;
            --u;
            --z;

            ////////////////////////////////////////////////////////////////////////
            // allow garbage collector to clean up memory
            //

            dual.Je0 = nullptr;
            dual.Je1 = nullptr;

            dual.A->clear();
            dual.A = nullptr;
            dual.b = nullptr;
            dual.u1 = nullptr;

            dual.zm = nullptr;

            DTM::pout << std::endl;
            timer->leave_subsection();
    }

	DTM::pout
            << "dual: backwards TMS problem done" << std::endl
            << "*******************************************************************"
            << "*************" << std::endl
            << std::endl;

	////////////////////////////////////////////////////////////////////////////
	// allow garbage collector to clean up memory
	//

	if (dual.zm.use_count())
		dual.zm = nullptr;


	if (dual.zn.use_count())
		dual.zn = nullptr;

}

////////////////////////////////////////////////////////////////////////////////
// dual data output
//

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
dual_init_data_output() {
    Assert(parameter_set.use_count(), dealii::ExcNotInitialized());

    // set up which dwr loop(s) are allowed to make data output:
    if ( !parameter_set->data_output.dual.dwr_loop.compare("none") ) {
        return;
    }

    // may output data: initialise (mode: all, last or specific dwr loop)
    DTM::pout
        << "dual solution data output: patches = "
        << parameter_set->data_output.dual.patches
        << std::endl;

    std::vector<std::string> data_field_names;
    data_field_names.push_back("z_theta");
    data_field_names.push_back("z_Y");

    std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation > dci_field;
    dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar);

    dual.data_output = std::make_shared< DTM::DataOutput<dim> >();
    dual.data_output->set_data_field_names(data_field_names);
    dual.data_output->set_data_component_interpretation_field(dci_field);

    dual.data_output->set_data_output_patches(
            parameter_set->data_output.dual.patches
    );

    // check if we use a fixed trigger interval, or, do output once on a I_n
    if ( !parameter_set->data_output.dual.trigger_type.compare("fixed") ) {
        dual.data_output_trigger_type_fixed = true;
    }
    else {
        dual.data_output_trigger_type_fixed = false;
    }

    // only for fixed
    dual.data_output_trigger = parameter_set->data_output.dual.trigger;

    if (dual.data_output_trigger_type_fixed) {
        DTM::pout
            << "dual solution data output: using fixed mode with trigger = "
            << dual.data_output_trigger
            << std::endl;
    }
    else {
        DTM::pout
            << "dual solution data output: using I_n mode (trigger adapts to I_n automatically)"
            << std::endl;
    }

    dual.data_output_time_value = parameter_set->T;
}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
dual_do_data_output_on_slab(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &z,
    const unsigned int dwr_loop)
{
    if (dual.data_output_trigger <= 0) return;

    // adapt trigger value for I_n output mode
    if (!dual.data_output_trigger_type_fixed) {
        dual.data_output_trigger = slab->tau_n();

        if (slab == std::prev(grid->slabs.end())) {
            dual.data_output_time_value = slab->t_n;
        }
        else {
            dual.data_output_time_value = slab->t_m;
        }
    }

    auto z_trigger = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

    if ( parameter_set->fe.dual.high_order){
        dual.data_output->set_DoF_data(
            slab->high->dof
        );

        z_trigger->reinit(
            *slab->high->locally_owned_dofs,
            *slab->high->locally_relevant_dofs,
            mpi_comm
        );
    } else{
        dual.data_output->set_DoF_data(
            slab->low->dof
        );

        z_trigger->reinit(
            *slab->low->locally_owned_dofs,
            *slab->low->locally_relevant_dofs,
            mpi_comm
        );
    }

    std::ostringstream filename;
    filename
        << "dual-dwr_loop-"
        << std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop+1;

    double &t{dual.data_output_time_value};

    for ( ; t >= slab->t_m; t -= dual.data_output_trigger) {
        // evalute space-time solution
        z_trigger->equ(1.0, *z->x[0]);

        dual.data_output->write_data(
            filename.str(),
            z_trigger,
            t
        );
    }

    // check if data for t=0 (t_0) was written
    if (slab == grid->slabs.begin()) {
    if (dual.data_output_trigger_type_fixed) {
        const double overshoot_tol{
            std::min(slab->tau_n(), dual.data_output_trigger) * 1e-7
        };


        if ((t < slab->t_m) && (std::abs(t - slab->t_m) < overshoot_tol)) {
            // undershoot of time variable; manually set t = 0 and do data output
            t = slab->t_m;

            // evalute space-time solution
            *z_trigger=*z->x[0];

            dual.data_output->write_data(
                filename.str(),
                z_trigger,
                t
            );
        }
    }}
}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
dual_do_data_output(
    const unsigned int dwr_loop,
    bool last
) {
    // set up which dwr loop(s) are allowed to make data output:
    Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
    if ( !parameter_set->data_output.dual.dwr_loop.compare("none") ) {
        return;
    }

    if (!parameter_set->data_output.dual.dwr_loop.compare("last")) {
        // output only the last (final) dwr loop
        if (last) {
            dual.data_output_dwr_loop = dwr_loop;
        }
        else {
            return;
        }
    }
    else {
        if (!parameter_set->data_output.dual.dwr_loop.compare("all")) {
            // output all dwr loops
            if (!last) {
                dual.data_output_dwr_loop = dwr_loop;
            }
            else {
                return;
            }
        }
        else {
            // output on a specific dwr loop
            if (!last) {
                dual.data_output_dwr_loop =
                    std::stoi(parameter_set->data_output.dual.dwr_loop)-1;
            }
            else {
                return;
            }
        }

    }

    if (dual.data_output_dwr_loop < 0)
        return;

    if ( static_cast<unsigned int>(dual.data_output_dwr_loop) != dwr_loop )
        return;

    DTM::pout
        << "dual solution data output: dwr loop = "
        << dual.data_output_dwr_loop
        << std::endl;

    dual.data_output_time_value = parameter_set->T;

    Assert(grid->slabs.size(), dealii::ExcNotInitialized());
    auto slab = std::prev(grid->slabs.end());
    auto z = std::prev(dual.storage.z->end());

    unsigned int n{static_cast<unsigned int>(grid->slabs.size())};
    while (n) {
        dual_do_data_output_on_slab(slab,z,dwr_loop);

        --n;
        --slab;
        --z;
    }
}
////////////////////////////////////////////////////////////////////////////////
// error estimation
//

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
eta_reinit_storage() {
    ////////////////////////////////////////////////////////////////////////////
    // init storage containers for vector data:
    // NOTE: * error indicators \f$ \eta \f$ (one per slab)
    //

    Assert(grid.use_count(), dealii::ExcNotInitialized());
    // get number of time steps N
    const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};

    error_estimator.storage.primal.eta_h = std::make_shared< DTM::types::storage_data_trilinos_vectors<1> > ();
    error_estimator.storage.primal.eta_h->resize(N);
    {
        auto slab = grid->slabs.begin();
        for (auto &element : *error_estimator.storage.primal.eta_h) {
            for (unsigned int j{0}; j < element.x.size(); ++j) {
                element.x[j] = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

                Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
                Assert(slab->tria.use_count(), dealii::ExcNotInitialized());


                Assert(
                    slab->pu->dof->n_dofs(),
                    dealii::ExcMessage("Error: slab->pu->dof->n_dofs() == 0")
                );

                // initialise dealii::Vector<double> with #PU-dofs
                element.x[j]->reinit(
                    *slab->pu->locally_owned_dofs, mpi_comm
                );
                *element.x[j]=0;
            }
            ++slab;
        }
    }

    error_estimator.storage.primal.eta_k = std::make_shared< DTM::types::storage_data_trilinos_vectors<1> > ();
    error_estimator.storage.primal.eta_k->resize(N);
    {
        auto slab = grid->slabs.begin();
        for (auto &element : *error_estimator.storage.primal.eta_k) {
            for (unsigned int j{0}; j < element.x.size(); ++j) {
                element.x[j] = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

                Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
                Assert(slab->tria.use_count(), dealii::ExcNotInitialized());
                Assert(
                    slab->pu->dof->n_dofs(),
                    dealii::ExcMessage("Error: slab->pu->dof->n_dofs() == 0")
                );

                // initialise dealii::Vector<double> with #PU-dofs
                element.x[j]->reinit(
                    *slab->pu->locally_owned_dofs, mpi_comm
                );
                *element.x[j]=0;

            }
            ++slab;
        }
    }


    error_estimator.storage.adjoint.eta_h = std::make_shared< DTM::types::storage_data_trilinos_vectors<1> > ();
    error_estimator.storage.adjoint.eta_h->resize(N);
    {
        auto slab = grid->slabs.begin();
        for (auto &element : *error_estimator.storage.adjoint.eta_h) {
            for (unsigned int j{0}; j < element.x.size(); ++j) {
                element.x[j] = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

                Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
                Assert(slab->tria.use_count(), dealii::ExcNotInitialized());

                Assert(
                    slab->pu->dof->n_dofs(),
                    dealii::ExcMessage("Error: slab->pu->dof->n_dofs() == 0")
                );

                // initialise dealii::Vector<double> with #PU-dofs
                element.x[j]->reinit(
                    *slab->pu->locally_owned_dofs, mpi_comm
                );

                *element.x[j]=0;

            }
            ++slab;
        }
    }

    error_estimator.storage.adjoint.eta_k = std::make_shared< DTM::types::storage_data_trilinos_vectors<1> > ();
    error_estimator.storage.adjoint.eta_k->resize(N);
    {
        auto slab = grid->slabs.begin();
        for (auto &element : *error_estimator.storage.adjoint.eta_k) {
            for (unsigned int j{0}; j < element.x.size(); ++j) {
                element.x[j] = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

                Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
                Assert(
                    slab->pu->dof->n_dofs(),
                    dealii::ExcMessage("Error: slab->pu->dof->n_dofs() == 0")
                );

                // initialise dealii::Vector<double> with #PU-dofs
                element.x[j]->reinit(
                    *slab->pu->locally_owned_dofs, mpi_comm
                );

                *element.x[j]=0;
            }
            ++slab;
        }
    }
}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
eta_interpolate_slab(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
    const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &z,
    std::shared_ptr<combustion::dwr::estimator::Arguments> args,
    bool initial_slab,
    bool last_slab
){
    args->tm.u_kh = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();
    args->tm.u_kh->reinit(*slab->high->locally_owned_dofs,*slab->high->locally_relevant_dofs,mpi_comm);

    args->tm.z_k = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();
    args->tm.z_k->reinit(*slab->high->locally_owned_dofs,*slab->high->locally_relevant_dofs,mpi_comm);

    args->tn.u_k = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();
    args->tn.u_k->reinit(*slab->high->locally_owned_dofs,*slab->high->locally_relevant_dofs,mpi_comm);

    args->tn.u_kh = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();
    args->tn.u_kh->reinit(*slab->high->locally_owned_dofs,*slab->high->locally_relevant_dofs,mpi_comm);

    args->tn.z_k = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();
    args->tn.z_k->reinit(*slab->high->locally_owned_dofs,*slab->high->locally_relevant_dofs,mpi_comm);

    args->tn.z_kh = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();
    args->tn.z_kh->reinit(*slab->high->locally_owned_dofs,*slab->high->locally_relevant_dofs,mpi_comm);

    args->tnp1.z_kh = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();
    args->tnp1.z_kh->reinit(*slab->high->locally_owned_dofs,*slab->high->locally_relevant_dofs,mpi_comm);

    args->tnp1.u_k = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();
    args->tnp1.u_k->reinit(*slab->high->locally_owned_dofs,*slab->high->locally_relevant_dofs,mpi_comm);

    auto tmp_low_relevant = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
    tmp_low_relevant->reinit(*slab->low->locally_owned_dofs, *slab->low->locally_relevant_dofs, mpi_comm);

    auto tmp_low_owned = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
    tmp_low_owned->reinit(*slab->low->locally_owned_dofs, mpi_comm);

    auto tmp_high_relevant = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
    tmp_high_relevant->reinit(*slab->high->locally_owned_dofs, *slab->high->locally_relevant_dofs, mpi_comm);

    auto tmp_high_owned = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
    tmp_high_owned->reinit(*slab->high->locally_owned_dofs, mpi_comm);

    function.u_0->set_time(0);
    if (parameter_set->fe.primal.high_order){
        // do equal high order
        if ( initial_slab) {
            dealii::VectorTools::interpolate(
                *slab->high->dof,
                *function.u_0,
                *tmp_high_owned
            );
            slab->high->constraints->distribute(*tmp_high_owned);

            //Z_0 = Z_1
            *args->tm.z_k = *z->x[0];
        }
        else {
            dealii::VectorTools::interpolate_to_different_mesh(
                /* solution on I_{n-1}:*/ *std::prev(slab)->high->dof,                          *std::prev(z)->x[0],
                /* solution on I_n: */               *slab->high->dof, *slab->high->constraints, *tmp_high_owned
            );

            *args->tm.z_k = *tmp_high_owned;

            dealii::VectorTools::interpolate_to_different_mesh(
                /* solution on I_{n-1}:*/ *std::prev(slab)->high->dof,                          *std::prev(u)->x[0],
                /* solution on I_n: */               *slab->high->dof, *slab->high->constraints, *tmp_high_owned
            );
        }

        *tmp_high_relevant = *tmp_high_owned;
        //calculate restricted primal solution at previous time
        dealii::FETools::interpolate(
            /*high  */ *slab->high->dof,  *tmp_high_relevant,
            /*low */ *slab->low->dof, *slab->low->constraints, *tmp_low_owned
        );

            *tmp_low_relevant = *tmp_low_owned;
            dealii::FETools::interpolate(
                        /*primal solution */ *slab->low->dof,		                  *tmp_low_relevant,
                        /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
            );

            *args->tm.u_kh = *tmp_high_owned;

            *args->tn.u_k = *u->x[0];

            //calculate restricted primal solution at current time
            dealii::FETools::interpolate(
                /*high  */ *slab->high->dof,  *args->tn.u_k,
                /*low */ *slab->low->dof, *slab->low->constraints, *tmp_low_owned
            );

            *tmp_low_relevant = *tmp_low_owned;
            dealii::FETools::interpolate(
                /*primal solution */ *slab->low->dof,				  *tmp_low_relevant,
                /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
            );

            *args->tn.u_kh = *tmp_high_owned;


            *args->tn.z_k = *z->x[0];

            //calculate restricted primal solution at current time
            dealii::FETools::interpolate(
                /*high  */ *slab->high->dof,  *args->tn.z_k,
                /*low */ *slab->low->dof, *slab->low->constraints, *tmp_low_owned
            );

            *tmp_low_relevant = *tmp_low_owned;
            dealii::FETools::interpolate(
                /*primal solution */ *slab->low->dof,							*tmp_low_relevant,
                /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
            );

            *args->tn.z_kh = *tmp_high_owned;

            if ( last_slab ){
                *args->tnp1.z_kh = 0;

                *args->tnp1.u_k = *u->x[0];
            } else {
                dealii::VectorTools::interpolate_to_different_mesh(
                    /* solution on I_{n-1}:*/ *std::next(slab)->high->dof,                          *std::next(z)->x[0],
                    /* solution on I_n: */               *slab->high->dof, *slab->high->constraints, *tmp_high_owned
                );
                *tmp_high_relevant = *tmp_high_owned;
                //calculate restricted dual solution at next time
                dealii::FETools::interpolate(
                    /*high  */ *slab->high->dof,                       *tmp_high_relevant,
                    /*low */ *slab->low->dof, *slab->low->constraints, *tmp_low_owned
                );

                *tmp_low_relevant = *tmp_low_owned;
                dealii::FETools::interpolate(
                    /*primal solution */ *slab->low->dof,       		      *tmp_low_relevant,
                    /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
                );
                *args->tnp1.z_kh = *tmp_high_owned;

                dealii::VectorTools::interpolate_to_different_mesh(
                    /* solution on I_{n-1}:*/ *std::next(slab)->high->dof,                           *std::next(u)->x[0],
                    /* solution on I_n: */               *slab->high->dof, *slab->high->constraints, *tmp_high_owned
                );
                *args->tnp1.u_k = *tmp_high_owned;
            }
    } else if (!parameter_set->fe.dual.high_order){
        //do equal low order
        if (initial_slab) {
            //interpolate IV function to low space
            dealii::VectorTools::interpolate(
                *slab->low->dof,
                *function.u_0,
                *tmp_low_owned
            );
            slab->low->constraints->distribute(*tmp_low_owned);

            dealii::FETools::extrapolate(
                /*primal solution */ *slab->low->dof,				  *z->x[0],
                /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
            );

            *args->tm.z_k = *tmp_high_owned;
        } else{
            dealii::VectorTools::interpolate_to_different_mesh(
                /* solution on I_{n-1}:*/ *std::prev(slab)->low->dof,                          *std::prev(z)->x[0],
                /* solution on I_n: */               *slab->low->dof, *slab->low->constraints, *tmp_low_owned
            );

            *tmp_low_relevant = *tmp_low_owned;
            dealii::FETools::extrapolate(
                /*primal solution */ *slab->low->dof,				  *tmp_low_relevant,
                /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
            );
            *args->tm.z_k = *tmp_high_owned;

            dealii::VectorTools::interpolate_to_different_mesh(
                /* solution on I_{n-1}:*/ *std::prev(slab)->low->dof,                          *std::prev(u)->x[0],
                /* solution on I_n: */               *slab->low->dof, *slab->low->constraints, *tmp_low_owned
            );

        }

        *tmp_low_relevant = *tmp_low_owned;
        dealii::FETools::interpolate(
            /*primal solution */ *slab->low->dof,			      *tmp_low_relevant,
            /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
        );

        *args->tm.u_kh = *tmp_high_owned;


        //starting point tm done, now do endpoint tn i.e. current interval values
        dealii::FETools::interpolate(
            /*primal solution */ *slab->low->dof,			      *u->x[0],
            /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
        );
        *args->tn.u_kh = *tmp_high_owned;

        dealii::FETools::extrapolate(
            /*primal solution */ *slab->low->dof,			      *u->x[0],
            /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
        );
        *args->tn.u_k = *tmp_high_owned;

        dealii::FETools::interpolate(
            /*primal solution */ *slab->low->dof,			      *z->x[0],
            /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
        );
        *args->tn.z_kh = *tmp_high_owned;

        dealii::FETools::extrapolate(
            /*primal solution */ *slab->low->dof,			      *z->x[0],
            /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
        );
        *args->tn.z_k = *tmp_high_owned;

        if ( last_slab) {
            *args->tnp1.z_kh = 0;

            dealii::FETools::extrapolate(
                /*primal solution */ *slab->low->dof,				  *u->x[0],
                /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
            );
            *args->tnp1.u_k = *tmp_high_owned;
        } else {
            dealii::VectorTools::interpolate_to_different_mesh(
                /* solution on I_{n+1}:*/ *std::next(slab)->low->dof,                          *std::next(z)->x[0],
                /* solution on I_n: */               *slab->low->dof, *slab->low->constraints, *tmp_low_owned
            );
            *tmp_low_relevant = *tmp_low_owned;
            dealii::FETools::interpolate(
                /*primal solution */ *slab->low->dof,				  *tmp_low_relevant,
                /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
            );
            *args->tnp1.z_kh = *tmp_high_owned;

            dealii::VectorTools::interpolate_to_different_mesh(
                /* solution on I_{n+1}:*/ *std::next(slab)->low->dof,                          *std::next(u)->x[0],
                /* solution on I_n: */               *slab->low->dof, *slab->low->constraints, *tmp_low_owned
            );
            *tmp_low_relevant = *tmp_low_owned;

            dealii::FETools::extrapolate(
                /*primal solution */ *slab->low->dof,				  *tmp_low_relevant,
                /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
            );
            *args->tnp1.u_k = *tmp_high_owned;
        }
    } else{
        //do mixed order
        if ( initial_slab) {
            //interpolate IV function to low space
            dealii::VectorTools::interpolate(
                *slab->low->dof,
                *function.u_0,
                *tmp_low_owned
            );
            slab->low->constraints->distribute(*tmp_low_owned);

            *args->tm.z_k = *z->x[0];
        } else {
            dealii::VectorTools::interpolate_to_different_mesh(
                /* solution on I_{n-1}:*/ *std::prev(slab)->high->dof,                          *std::prev(z)->x[0],
                /* solution on I_n: */               *slab->high->dof, *slab->high->constraints, *tmp_high_owned
            );

            *args->tm.z_k = *tmp_high_owned;


            dealii::VectorTools::interpolate_to_different_mesh(
                /* solution on I_{n-1}:*/ *std::prev(slab)->low->dof,                          *std::prev(u)->x[0],
                /* solution on I_n: */               *slab->low->dof, *slab->low->constraints, *tmp_low_owned
            );
        }

        *tmp_low_relevant = *tmp_low_owned;
        dealii::FETools::interpolate(
            /*primal solution */ *slab->low->dof,				  *tmp_low_relevant,
            /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
        );

        *args->tm.u_kh = *tmp_high_owned;

        //starting point tm done, now do endpoint tn i.e. current interval values
        dealii::FETools::interpolate(
            /*primal solution */ *slab->low->dof,				  *u->x[0],
            /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
        );
        *args->tn.u_kh = *tmp_high_owned;

        dealii::FETools::extrapolate(
            /*primal solution */ *slab->low->dof,				  *u->x[0],
            /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
        );
        *args->tn.u_k = *tmp_high_owned;

        //calculate restricted dual solution at current time
        dealii::FETools::interpolate(
            /*high  */ *slab->high->dof,                       *z->x[0],
            /*low */ *slab->low->dof, *slab->low->constraints, *tmp_low_owned
        );

        *tmp_low_relevant = *tmp_low_owned;
        dealii::FETools::interpolate(
            /*primal solution */ *slab->low->dof,				  *tmp_low_relevant,
            /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
        );

        *args->tn.z_kh = *tmp_high_owned;

        *args->tn.z_k = *z->x[0];


        if (last_slab) {
            *args->tnp1.z_kh = 0;

            *args->tnp1.u_k = *args->tn.u_k;
        } else {

            dealii::VectorTools::interpolate_to_different_mesh(
                /* solution on I_{n-1}:*/ *std::next(slab)->high->dof,                           *std::next(z)->x[0],
                /* solution on I_n: */               *slab->high->dof, *slab->high->constraints, *tmp_high_owned
            );
            *tmp_high_relevant = *tmp_high_owned;
            //calculate restricted dual solution at next time
            dealii::FETools::interpolate(
                /*high  */ *slab->high->dof,                       *tmp_high_relevant,
                /*low */ *slab->low->dof, *slab->low->constraints, *tmp_low_owned
            );

            *tmp_low_relevant = *tmp_low_owned;
            dealii::FETools::interpolate(
                /*primal solution */ *slab->low->dof,			      *tmp_low_relevant,
                /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
            );
            *args->tnp1.z_kh = *tmp_high_owned;

            dealii::VectorTools::interpolate_to_different_mesh(
                /* solution on I_{n-1}:*/ *std::next(slab)->low->dof,                          *std::next(u)->x[0],
                /* solution on I_n: */               *slab->low->dof, *slab->low->constraints, *tmp_low_owned
            );

            *tmp_low_relevant = *tmp_low_owned;

            dealii::FETools::extrapolate(
                /*primal solution */ *slab->low->dof,			      *tmp_low_relevant,
                /*dual solution */   *slab->high->dof,  *slab->high->constraints, *tmp_high_owned
            );
            *args->tnp1.u_k = *tmp_high_owned;

        }

    }
    tmp_low_relevant = nullptr;
    tmp_high_relevant = nullptr;
    tmp_low_owned = nullptr;
    tmp_high_owned = nullptr;
}


template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
compute_pu_dof_error_indicators() {
    error_estimator.pu_dof = std::make_shared<combustion::dwr::estimator::PUDoFErrorEstimator<dim> > (mpi_comm);

    error_estimator.pu_dof->set_functions(
        function.u_0,
        function.u_N,
        function.u_R
    );
    error_estimator.pu_dof->set_parameters(
        parameter.alpha,
        parameter.beta,
        parameter.Le,
        parameter.robin_factor_theta,
        parameter.robin_factor_Y,
        parameter_set->T,
        cardinality.domain,
        cardinality.rod,
        parameter_set->dwr.goal.functional
    );

    auto slab = grid->slabs.begin();
    auto u = primal.storage.u->begin();
    auto z = dual.storage.z->begin();
    auto eta_h_p = error_estimator.storage.primal.eta_h->begin();
    auto eta_k_p = error_estimator.storage.primal.eta_k->begin();
    auto eta_h_a = error_estimator.storage.adjoint.eta_h->begin();
    auto eta_k_a = error_estimator.storage.adjoint.eta_k->begin();

    while (slab!= grid->slabs.end()) {
        auto args = std::make_shared<combustion::dwr::estimator::Arguments> ();

        eta_interpolate_slab(slab, u, z, args, (slab == grid->slabs.begin()), (slab == std::prev(grid->slabs.end())));

        error_estimator.pu_dof->estimate_primal(
            slab,
            args,
            eta_h_p->x[0],
            eta_k_p->x[0]
        );

        error_estimator.pu_dof->estimate_dual(
            slab,
            args,
            eta_h_a->x[0],
            eta_k_a->x[0]
        );

        ++slab;
        ++u; ++z;
        ++eta_h_p; ++eta_k_p;
        ++eta_h_a; ++eta_k_a;
    }



}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
eta_init_data_output() {
    Assert(parameter_set.use_count(), dealii::ExcNotInitialized());

    // set up which dwr loop(s) are allowed to make data output:
    if ( !parameter_set->data_output.dual.dwr_loop.compare("none") ) {
        return;
    }

    // may output data: initialise (mode: all, last or specific dwr loop)
    DTM::pout
        << "error estimator data output: patches = "
        << parameter_set->data_output.dual.patches
        << std::endl;

    std::vector<std::string> data_field_names;
    data_field_names.push_back("eta");

    std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation > dci_field;
    dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar);

    error_estimator.data_output = std::make_shared< DTM::DataOutput<dim> >();
    error_estimator.data_output->set_data_field_names(data_field_names);
    error_estimator.data_output->set_data_component_interpretation_field(dci_field);

    error_estimator.data_output->set_data_output_patches(1);

    // check if we use a fixed trigger interval, or, do output once on a I_n
    if ( !parameter_set->data_output.dual.trigger_type.compare("fixed") ) {
        error_estimator.data_output_trigger_type_fixed = true;
    }
    else {
        error_estimator.data_output_trigger_type_fixed = false;
    }

    // only for fixed
    error_estimator.data_output_trigger = parameter_set->data_output.dual.trigger;

    if (error_estimator.data_output_trigger_type_fixed) {
        DTM::pout
            << "error estimator data output: using fixed mode with trigger = "
            << dual.data_output_trigger
            << std::endl;
    }
    else {
        DTM::pout
            << "error estimator data output: using I_n mode (trigger adapts to I_n automatically)"
            << std::endl;
    }

    error_estimator.data_output_time_value = parameter_set->T;
}


template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
eta_do_data_output_on_slab(
    const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
    const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &eta,
    const unsigned int dwr_loop,
    std::string eta_type)
{
    if (error_estimator.data_output_trigger <= 0) return;

    // adapt trigger value for I_n output mode
    if (!error_estimator.data_output_trigger_type_fixed) {
        error_estimator.data_output_trigger = slab->tau_n();

        if (slab == std::prev(grid->slabs.end())) {
            error_estimator.data_output_time_value = slab->t_n;
        }
        else {
            error_estimator.data_output_time_value = slab->t_m;
        }
    }

    auto z_trigger = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

    error_estimator.data_output->set_DoF_data(
        slab->pu->dof
    );

    z_trigger->reinit(
        *slab->pu->locally_owned_dofs,
        *slab->pu->locally_relevant_dofs,
        mpi_comm
    );

    std::ostringstream filename;
    filename
        << eta_type << "-dwr_loop-"
        << std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop+1;

    double &t{error_estimator.data_output_time_value};

    for ( ; t >= slab->t_m; t -= error_estimator.data_output_trigger) {
        // evalute space-time solution
        z_trigger->equ(1.0, *eta->x[0]);

        error_estimator.data_output->write_data(
            filename.str(),
            z_trigger,
            t
        );
    }

    // check if data for t=0 (t_0) was written
    if (slab == grid->slabs.begin()) {
    if (error_estimator.data_output_trigger_type_fixed) {
        const double overshoot_tol{
            std::min(slab->tau_n(), error_estimator.data_output_trigger) * 1e-7
        };


        if ((t < slab->t_m) && (std::abs(t - slab->t_m) < overshoot_tol)) {
            // undershoot of time variable; manually set t = 0 and do data output
            t = slab->t_m;

            // evalute space-time solution
            *z_trigger=*eta->x[0];

            error_estimator.data_output->write_data(
                filename.str(),
                z_trigger,
                t
            );
        }
    }}
}


template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
eta_do_data_output(
    const unsigned int dwr_loop,
    bool last
) {
    // set up which dwr loop(s) are allowed to make data output:
    Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
    if ( !parameter_set->data_output.dual.dwr_loop.compare("none") ) {
        return;
    }

    if (!parameter_set->data_output.dual.dwr_loop.compare("last")) {
        // output only the last (final) dwr loop
        if (last) {
            error_estimator.data_output_dwr_loop = dwr_loop;
        }
        else {
            return;
        }
    }
    else {
        if (!parameter_set->data_output.dual.dwr_loop.compare("all")) {
            // output all dwr loops
            if (!last) {
                error_estimator.data_output_dwr_loop = dwr_loop;
            }
            else {
                return;
            }
        }
        else {
            // output on a specific dwr loop
            if (!last) {
                error_estimator.data_output_dwr_loop =
                    std::stoi(parameter_set->data_output.dual.dwr_loop)-1;
            }
            else {
                return;
            }
        }

    }

    if (error_estimator.data_output_dwr_loop < 0)
        return;

    if ( static_cast<unsigned int>(error_estimator.data_output_dwr_loop) != dwr_loop )
        return;

    DTM::pout
        << "error estimator data output: dwr loop = "
        << error_estimator.data_output_dwr_loop
        << std::endl;

    error_estimator.data_output_time_value = parameter_set->T;

    Assert(grid->slabs.size(), dealii::ExcNotInitialized());
    auto slab = std::prev(grid->slabs.end());
    auto p_eta_k = std::prev(error_estimator.storage.primal.eta_k->end());
    auto p_eta_h= std::prev(error_estimator.storage.primal.eta_h->end());
    auto a_eta_k = std::prev(error_estimator.storage.adjoint.eta_k->end());
    auto a_eta_h = std::prev(error_estimator.storage.adjoint.eta_h->end());


    unsigned int n{static_cast<unsigned int>(grid->slabs.size())};
    while (n) {
        eta_do_data_output_on_slab(slab,p_eta_k,dwr_loop,"p_eta_k");
        eta_do_data_output_on_slab(slab,p_eta_h,dwr_loop,"p_eta_h");
        eta_do_data_output_on_slab(slab,a_eta_k,dwr_loop,"a_eta_k");
        eta_do_data_output_on_slab(slab,a_eta_h,dwr_loop,"a_eta_h");

        --n;
        --slab;
        --p_eta_k;
        --p_eta_h;
        --a_eta_k;
        --a_eta_h;
    }
}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
compute_effectivity_index() {
    double eta_k_primal{0.};
    for ( const auto &eta_it : *error_estimator.storage.primal.eta_k ) {
        eta_k_primal += eta_it.x[0]->mean_value()*eta_it.x[0]->size();
    }

    double eta_k_adjoint{0.};
    for ( const auto &eta_it : *error_estimator.storage.adjoint.eta_k ) {
        eta_k_adjoint += eta_it.x[0]->mean_value()*eta_it.x[0]->size();
    }

    double eta_h_primal{0.};
    double eta_h_adjoint{0.};
    unsigned int K_max{0};
    const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};

    auto eta_it_p{error_estimator.storage.primal.eta_h->begin()};
    auto eta_it_a{error_estimator.storage.adjoint.eta_h->begin()};
    auto slab{grid->slabs.begin()};
    auto ends{grid->slabs.end()};

    for ( ; slab != ends ; ++slab, ++eta_it_p, ++eta_it_a ){
        eta_h_primal += parameter_set->T/(N*slab->tau_n())*
                        eta_it_p->x[0]->mean_value()* eta_it_p->x[0]->size();

        eta_h_adjoint += parameter_set->T/(N*slab->tau_n())*
                         eta_it_a->x[0]->mean_value()* eta_it_a->x[0]->size();
        K_max = ( K_max > slab->tria->n_global_active_cells())  ? K_max : slab->tria->n_global_active_cells();
    }

    const double eta_k = 0.5*(eta_k_primal+eta_k_adjoint);
    const double eta_h = 0.5*(eta_h_primal+eta_h_adjoint);
    const double eta_primal{std::abs(eta_k_primal)+std::abs(eta_h_primal)};
    const double eta_adjoint{std::abs(eta_k_adjoint)+std::abs(eta_h_adjoint)};
    const double eta{std::abs(eta_k)+std::abs(eta_h)};
    const double I_eff_primal{(eta_primal/primal_error)};
    const double I_eff_adjoint{(eta_adjoint/primal_error)};
    const double I_eff{(eta/primal_error)};

    DTM::pout << "\neta_k = " << eta_k
              << "\neta_k_primal = " << eta_k_primal
              << "\neta_k_adjoint = " << eta_k_adjoint
              << "\neta_h = " << eta_h
              << "\neta_h_primal = " << eta_h_primal
              << "\neta_h_adjoint = " << eta_h_adjoint
              << "\neta   = " << eta
              << "\neta_primal   = " << eta_primal
              << "\neta_adjoint   = " << eta_adjoint
              << "\nprimal_error = " << primal_error
              << "\nI_eff = " << I_eff
              << "\nI_eff_primal = " << I_eff_primal
              << "\nI_eff_adjoint = " << I_eff_adjoint << std::endl;

    // push local variables to convergence_table to avoid additional costs later.
    convergence_table.add_value("N_max", N);
    convergence_table.add_value("K_max", K_max);
    convergence_table.add_value("primal_error", primal_error);
    convergence_table.add_value("eta_k", eta_k);
    convergence_table.add_value("eta_h", eta_h);
    convergence_table.add_value("eta", eta);
    convergence_table.add_value("I_eff", I_eff);
}


template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
refine_and_coarsen_space_time_grid() {
    if ( parameter_set->dwr.refine_and_coarsen.adaptive )
    {
        refine_and_coarsen_space_time_grid_sv_dof();

    } else
    {
        refine_and_coarsen_space_time_grid_global();
    }

}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
refine_and_coarsen_space_time_grid_global() {

    unsigned int K_max{0};
    auto slab{grid->slabs.begin()};
    auto ends{grid->slabs.end()};
    for (unsigned int n{0} ; slab != ends; ++slab, ++n) {

        DTM::pout << "\tn = " << n << std::endl;

        const auto n_active_cells_on_slab{slab->tria->n_global_active_cells()};
        DTM::pout << "\t#K = " << n_active_cells_on_slab << std::endl;
        K_max = (K_max > n_active_cells_on_slab) ? K_max : n_active_cells_on_slab;

        //global refinement in space
        slab->tria->refine_global(1);
        grid->refine_slab_in_time(slab);
    }
    DTM::pout << "\t#Kmax (before refinement) = " << K_max << std::endl;
}

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
refine_and_coarsen_space_time_grid_sv_dof() {

    bool dofull = true;
    Assert(
        error_estimator.storage.primal.eta_k->size()==grid->slabs.size(),
        dealii::ExcInternalError()
    );
    Assert(
        error_estimator.storage.primal.eta_h->size()==grid->slabs.size(),
        dealii::ExcInternalError()
    );

    const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};
    std::vector<double> eta_k(N);

    double eta_k_global {0.};
    double eta_h_global {0.};

    // compute eta^n on I_n for n=1..N as well as global estimators
    {
        auto eta_it_p{error_estimator.storage.primal.eta_k->begin()};
        auto eta_it_a{error_estimator.storage.adjoint.eta_k->begin()};
        for (unsigned n{0}; n < N; ++n, ++eta_it_p,++eta_it_a) {
            Assert(
                (eta_it_p != error_estimator.storage.primal.eta_k->end()),
                dealii::ExcInternalError()
            );
            Assert(
                (eta_it_a != error_estimator.storage.adjoint.eta_k->end()),
                dealii::ExcInternalError()
            );

            double eta_k_K_p = eta_it_p->x[0]->mean_value()*eta_it_p->x[0]->size();
            double eta_k_K_a = eta_it_a->x[0]->mean_value()*eta_it_a->x[0]->size();
            if (dofull){
                eta_k[n] = std::abs(eta_k_K_p+eta_k_K_a);
                eta_k_global += 0.5*(eta_k_K_p+eta_k_K_a);
            } else {
                eta_k[n] = std::abs(eta_k_K_p);
                eta_k_global += eta_k_K_p;
            }
        }
    }


    //Per definition eta_k[0] is 0 for the primal problem, so just set it to the next time step
    if (!dofull){
        eta_k[0] = eta_k[1];
    }


    {
        auto eta_it_p{error_estimator.storage.primal.eta_h->begin()};
        auto eta_it_a{error_estimator.storage.adjoint.eta_h->begin()};
        auto slab{grid->slabs.begin()};
        auto ends{grid->slabs.end()};
        for (unsigned n{0}; n < N; ++n, ++eta_it_p, ++eta_it_a, ++slab) {
            Assert(
                (eta_it_p != error_estimator.storage.primal.eta_h->end()),
                dealii::ExcInternalError()
            );
            Assert(
                (eta_it_a != error_estimator.storage.adjoint.eta_h->end()),
                dealii::ExcInternalError()
            );

            eta_h_global += parameter_set->T/(N*slab->tau_n())* (
                             eta_it_p->x[0]->mean_value()*eta_it_p->x[0]->size()
                            +eta_it_a->x[0]->mean_value()*eta_it_a->x[0]->size()
                            );
        }
    }

    /*
     * Choose if temporal or spatial discretization should be refined
     * according to Algorithm 4.1 in Schmich & Vexler
     *
     */
    double equilibration_factor{1.0e7};
    // mark for temporal refinement
    if ( std::abs(eta_k_global)*equilibration_factor >= std::abs(eta_h_global))
    {
        Assert(
            ((parameter_set->dwr.refine_and_coarsen.time.top_fraction >= 0.) &&
            (parameter_set->dwr.refine_and_coarsen.time.top_fraction <= 1.)),
            dealii::ExcMessage(
                "parameter_set->dwr.refine_and_coarsen.time.top_fraction "
                "must be in [0,1]"
            )
        );

        if (parameter_set->dwr.refine_and_coarsen.time.top_fraction > 0.) {
            std::vector<double> eta_sorted(eta_k);
            std::sort(eta_sorted.begin(), eta_sorted.end(),std::greater<double>());


            double threshold = 0.;
            //do Doerfler marking
            if ( parameter_set->dwr.refine_and_coarsen.time.strategy.compare("fixed_fraction") == 0){
                double D_goal = std::accumulate(
                    eta_k.begin(),
                    eta_k.end(),
                    0.
                ) * parameter_set->dwr.refine_and_coarsen.time.top_fraction;

                double D_sum = 0.;
                for ( unsigned int n{0} ; n < N ; n++ )
                {
                    D_sum += eta_sorted[n];
                    if ( D_sum >= D_goal ){
                        threshold = eta_sorted[n];
                        n = N;
                    }
                }

            } else if (parameter_set->dwr.refine_and_coarsen.time.strategy.compare("fixed_number") == 0) {
                // check if index for eta_criterium_for_mark_time_refinement is valid
                Assert(static_cast<int>(std::ceil(static_cast<double>(N)
                        * parameter_set->dwr.refine_and_coarsen.time.top_fraction)) >= 0,
                    dealii::ExcInternalError()
                );

                unsigned int index_for_mark_time_refinement {
                    static_cast<unsigned int> (
                        static_cast<int>(std::ceil(
                            static_cast<double>(N)
                            * parameter_set->dwr.refine_and_coarsen.time.top_fraction
                        ))
                    )
                };

                    threshold = eta_sorted[ index_for_mark_time_refinement < N ?
                                            index_for_mark_time_refinement : N-1 ];

            }

            auto slab{grid->slabs.begin()};
            auto ends{grid->slabs.end()};
            for (unsigned int n{0} ; slab != ends; ++slab, ++n) {
                Assert((n < N), dealii::ExcInternalError());

                if (eta_k[n] >= threshold) {
                    slab->set_refine_in_time_flag();
                }
            }
        }
    }

    // spatial refinement
    if ( std::abs(eta_k_global) <= equilibration_factor*std::abs(eta_h_global))
    {
        unsigned int K_max{0};
        auto slab{grid->slabs.begin()};
        auto ends{grid->slabs.end()};
        auto eta_it_p{error_estimator.storage.primal.eta_h->begin()};
        auto eta_it_a{error_estimator.storage.primal.eta_h->begin()};
        for (unsigned int n{0} ; slab != ends; ++slab, ++eta_it_p, ++eta_it_a, ++n) {

            Assert(
                    (eta_it_p != error_estimator.storage.primal.eta_h->end()),
                    dealii::ExcInternalError()
            );
            Assert(
                    (eta_it_a != error_estimator.storage.adjoint.eta_h->end()),
                    dealii::ExcInternalError()
            );

            DTM::pout << "\tn = " << n << std::endl;

            const auto n_active_cells_on_slab{slab->tria->n_active_cells()}; //n_global_active
            DTM::pout << "\t#K = " << n_active_cells_on_slab << std::endl;
            K_max = (K_max > n_active_cells_on_slab) ? K_max : n_active_cells_on_slab;


            if ( parameter_set->dwr.refine_and_coarsen.space.top_fraction1 == 1.0 )
            {
                slab->tria->refine_global(1);
            }
            else {
                auto eta_relevant_p = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();
                eta_relevant_p->reinit(*slab->pu->locally_owned_dofs,*slab->pu->locally_relevant_dofs,mpi_comm);

                *eta_relevant_p = *eta_it_p->x[0];

                auto eta_relevant_a = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();
                eta_relevant_a->reinit(*slab->pu->locally_owned_dofs,*slab->pu->locally_relevant_dofs,mpi_comm);

                *eta_relevant_a = *eta_it_a->x[0];

                const unsigned int dofs_per_cell_pu = slab->pu->fe->dofs_per_cell;
                std::vector< unsigned int > local_dof_indices(dofs_per_cell_pu);
                unsigned int max_n = n_active_cells_on_slab *
                                parameter_set->dwr.refine_and_coarsen.space.max_growth_factor_n_active_cells;

                typename dealii::DoFHandler<dim>::active_cell_iterator
                cell{slab->pu->dof->begin_active()},
                endc{slab->pu->dof->end()};

                dealii::Vector<double> indicators(n_active_cells_on_slab);
                indicators= 0.;

                for ( unsigned int cell_no{0} ; cell!= endc; ++cell, ++cell_no){
                    if ( cell->is_locally_owned()){
                        cell->get_dof_indices(local_dof_indices);

                        for ( unsigned int i = 0 ; i < dofs_per_cell_pu ; i++) {
                            indicators[cell_no] += (*eta_relevant_p)(local_dof_indices[i])/dofs_per_cell_pu;
                            if (dofull){
                                indicators[cell_no] += (*eta_relevant_a)(local_dof_indices[i])/dofs_per_cell_pu;
                            }
                        }
                        indicators[cell_no ] = std::abs(indicators[cell_no]);
                    }
                }

                // mark for refinement strategy with fixed fraction
                // (similar but not identical to Hartmann Diploma thesis Ex. Sec. 1.4.2)
                const double top_fraction{ slab->refine_in_time ?
                    parameter_set->dwr.refine_and_coarsen.space.top_fraction1 :
                    parameter_set->dwr.refine_and_coarsen.space.top_fraction2
                };

                if( parameter_set->dwr.refine_and_coarsen.space.strategy.compare("fixed_fraction") == 0){
                    dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
                        *slab->tria,
                        indicators,
                        top_fraction,
                        parameter_set->dwr.refine_and_coarsen.space.bottom_fraction
                    );
                } else if (parameter_set->dwr.refine_and_coarsen.space.strategy.compare("fixed_number") == 0){
                    dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
                        *slab->tria,
                        indicators,
                        top_fraction,
                        parameter_set->dwr.refine_and_coarsen.space.bottom_fraction,
                        max_n
                    );
                } else{
                    AssertThrow(false,dealii::ExcMessage("unknown spatial refinement"));
                }



                // execute refinement in space under the conditions of mesh smoothing
                slab->tria->execute_coarsening_and_refinement();
            }
        }
        DTM::pout << "\t#Kmax (before refinement) = " << K_max << std::endl;
    }
    //do actual refine in time loop
    {
        auto slab{grid->slabs.begin()};
        auto ends{grid->slabs.end()};

        for (; slab != ends; ++slab) {
            if (slab->refine_in_time) {
                grid->refine_slab_in_time(slab);
                slab->refine_in_time = false;
            }
        }
    }
    DTM::pout << "refined in time" << std::endl;
}
////////////////////////////////////////////////////////////////////////////////
// other

template<int dim>
void
Combustion_DWR__cGp_dG0__cGq_dG0<dim>::
write_convergence_table_to_tex_file() {
    convergence_table.set_precision("primal_error", 5);
    convergence_table.set_precision("eta", 5);
    convergence_table.set_precision("I_eff", 3);

    convergence_table.set_scientific("primal_error", true);
    convergence_table.set_scientific("eta", true);
    convergence_table.set_scientific("eta_h", true);
    convergence_table.set_scientific("eta_k", true);

    std::cout << std::endl;
    convergence_table.write_text(std::cout);

    // Set tex captions and formation of respective columns
    convergence_table.set_tex_caption("DWR-loop","DWR-loop");
    convergence_table.set_tex_caption("N_max","$N_{\\text{max}}$");
    convergence_table.set_tex_caption("K_max","$K_{\\text{max}}$");
    convergence_table.set_tex_caption(
            "primal_error","$\\|J(u)-J(u_{kh})\\|_{(0,T)\\times\\Omega}$"
    );
    convergence_table.set_tex_caption("eta_k","$\\eta_k$");
    convergence_table.set_tex_caption("eta_h","$\\eta_h$");
    convergence_table.set_tex_caption("eta","$\\eta$");
    convergence_table.set_tex_caption("I_eff","I$_{\\text{eff}}$");
    convergence_table.set_tex_format("DWR-loop","c");
    convergence_table.set_tex_format("N_max","r");
    convergence_table.set_tex_format("K_max","r");
    convergence_table.set_tex_format("primal_error","c");
    convergence_table.set_tex_format("eta","c");
    convergence_table.set_tex_format("I_eff","c");

    std::vector<std::string> new_order;
    new_order.push_back("DWR-loop");
    new_order.push_back("N_max");
    new_order.push_back("K_max");
    new_order.push_back("primal_error");
    new_order.push_back("eta");
    new_order.push_back("eta_k");
    new_order.push_back("eta_h");
    new_order.push_back("I_eff");
    convergence_table.set_column_order (new_order);

    convergence_table.evaluate_convergence_rates(
        "primal_error",
        dealii::ConvergenceTable::reduction_rate
    );
    convergence_table.evaluate_convergence_rates(
        "primal_error",
        dealii::ConvergenceTable::reduction_rate_log2
    );

    // write TeX/LaTeX file of the convergence table with deal.II
    {
        std::string filename = "convergence-table.tex";
        std::ofstream out(filename.c_str());
        convergence_table.write_tex(out);
    }

    // read/write TeX/LaTeX file to make pdflatex *.tex working for our headers
    {
        std::ifstream in("convergence-table.tex");

        std::string filename = "my-convergence-table.tex";
        std::ofstream out(filename.c_str());

        std::string line;
        std::getline(in, line);
        out << line << std::endl;
        // add the missing amsmath latex package
        out << "\\usepackage{amsmath}" << std::endl;

        for ( ; std::getline(in, line) ; )
                out << line << std::endl;
        out.close();
    }
}

} // namespace

#include "Combustion_DWR__cGp_dG0__cGq_dG0.inst.in"
