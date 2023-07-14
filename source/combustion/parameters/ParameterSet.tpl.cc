/**
 * @file   ParameterSet.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 *
 * @brief Keeps all parsed input parameters in a struct.
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
#include <combustion/parameters/ParameterSet.hh>

// DEAL.II includes
#include <deal.II/base/exceptions.h>

// C++ includes
#include <limits>

namespace combustion {
namespace dwr {

ParameterSet::
ParameterSet(
    std::shared_ptr< dealii::ParameterHandler > handler) {
    Assert(handler.use_count(), dealii::ExcNotInitialized());

    dim = static_cast<unsigned int> (handler->get_integer("dim"));
	
    handler->enter_subsection("Problem Specification"); {
        fe.primal.space_type = handler->get("primal space type");
        fe.primal.space_type_support_points = handler->get(
            "primal space type support points"
        );
        fe.primal.p = static_cast<unsigned int> (handler->get_integer("primal p"));

        fe.primal.time_type = handler->get("primal time type");
        fe.primal.time_type_support_points = handler->get(
            "primal time type support points"
        );
        fe.primal.r = static_cast<unsigned int> (handler->get_integer("primal r"));


        fe.dual.space_type = handler->get("dual space type");
        fe.dual.space_type_support_points = handler->get(
            "dual space type support points"
        );
        fe.dual.q = static_cast<unsigned int> (handler->get_integer("dual q"));

        fe.dual.time_type = handler->get("dual time type");
        fe.dual.time_type_support_points = handler->get(
            "dual time type support points"
        );
        fe.dual.s = static_cast<unsigned int> (handler->get_integer("dual s"));

        std::string approach = handler->get("order approach");

        if ( approach.compare("equal high order") == 0 )
        {
            fe.primal.high_order = true;
            fe.dual.high_order = true;
        }
        else if ( approach.compare("equal low order") == 0)
        {
            fe.primal.high_order = false;
            fe.dual.high_order = false;
        } else // default mixed order
        {
            fe.primal.high_order = false;
            fe.dual.high_order = true;
        }

    }
    handler->leave_subsection();
	
    handler->enter_subsection("Mesh Specification"); {
        use_mesh_input_file = handler->get_bool("use mesh input file");
        mesh_input_filename = handler->get("mesh input filename");

        TriaGenerator = handler->get("TriaGenerator");
        TriaGenerator_Options = handler->get("TriaGenerator Options");

        Grid_Class = handler->get("Grid Class");
        Grid_Class_Options = handler->get("Grid Class Options");

        global_refinement = static_cast<unsigned int> (
            handler->get_integer("global refinement")
        );
    }
    handler->leave_subsection();

    handler->enter_subsection("Time Integration"); {
        t0 = handler->get_double("initial time");
        T = handler->get_double("final time");
        tau_n = handler->get_double("time step size");
    }
    handler->leave_subsection();
	
        
    handler->enter_subsection("Newton"); {
        newton.max_steps = handler->get_integer("max steps");
        newton.lower_bound = handler->get_double("lower bound");
        newton.rebuild = handler->get_double("rebuild parameter");
        newton.line_search_steps = handler->get_integer("line search steps");
        newton.line_search_damping = handler->get_double("line search damping");
    }
    handler->leave_subsection();
        
    handler->enter_subsection("DWR"); {
        dwr.goal.type = handler->get("functional type");

        dwr.goal.functional = handler->get("functional");

        dwr.reference.average.reaction_rate =
            handler->get_double("average reaction rate reference value");

        dwr.reference.average.rod_species_concentration =
            handler->get_double("average rod species concentration reference value");
                
        dwr.goal.calc_reaction_rate = handler->get_bool("reaction rate calculation");
        dwr.goal.calc_rod_species_concentration = handler->get_bool("rod species concentration calculation");

        dwr.solver_control.in_use = handler->get_bool("solver control in use");
        if (dwr.solver_control.in_use) {
            dwr.solver_control.reduction_mode = handler->get_bool(
                "solver control reduction mode"
            );

            dwr.solver_control.max_iterations = static_cast<unsigned int> (
                handler->get_integer("solver control max iterations")
            );
            dwr.loops = dwr.solver_control.max_iterations;

            dwr.solver_control.tolerance = handler->get_double(
                "solver control tolerance"
            );

            dwr.solver_control.reduction = handler->get_double(
                "solver control reduction"
            );
        }
        else {
            dwr.loops = static_cast<unsigned int> (handler->get_integer("loops"));
        }


        dwr.refine_and_coarsen.space.strategy = handler->get(
            "refine and coarsen space strategy"
        );

        dwr.refine_and_coarsen.space.top_fraction1 = handler->get_double(
            "refine and coarsen space top fraction1"
        );

        dwr.refine_and_coarsen.space.top_fraction2 = handler->get_double(
            "refine and coarsen space top fraction2"
        );

        dwr.refine_and_coarsen.space.bottom_fraction = handler->get_double(
            "refine and coarsen space bottom fraction"
        );

        dwr.refine_and_coarsen.space.max_growth_factor_n_active_cells =
            static_cast<unsigned int> (handler->get_integer(
            "refine and coarsen space max growth factor n_active_cells"
        ));

        dwr.refine_and_coarsen.space.theta1 = handler->get_double(
            "refine and coarsen space Schwegler theta1"
        );

        dwr.refine_and_coarsen.space.theta2 = handler->get_double(
            "refine and coarsen space Schwegler theta2"
        );


        dwr.refine_and_coarsen.time.strategy = handler->get(
            "refine and coarsen time strategy"
        );

        dwr.refine_and_coarsen.time.top_fraction = handler->get_double(
            "refine and coarsen time top fraction"
        );

        dwr.refine_and_coarsen.adaptive = true;
        if( ( (dwr.refine_and_coarsen.time.strategy.compare("global")  == 0 ) || (dwr.refine_and_coarsen.time.strategy.compare("none")  == 0) )&&
            ( (dwr.refine_and_coarsen.space.strategy.compare("global") == 0 ) || (dwr.refine_and_coarsen.space.strategy.compare("none") == 0) )
        ){
            dwr.refine_and_coarsen.adaptive = false;
        }
    }
    handler->leave_subsection();
	
    handler->enter_subsection("Parameter Specification"); {
        lewis_number = handler->get_double(
            "Lewis Number"
        );

        arrhenius_alpha = handler->get_double(
            "Arrhenius alpha"
        );

        arrhenius_beta = handler->get_double(
            "Arrhenius beta"
        );

        x_tilde = handler->get_double(
            "x_tilde"
        );

        dirichlet_boundary_u_D_function = handler->get(
            "dirichlet boundary u_D function"
        );

        dirichlet_boundary_u_D_options = handler->get(
            "dirichlet boundary u_D options"
        );

        dirichlet_assembler_n_quadrature_points = static_cast<unsigned int> (
            handler->get_integer(
                    "dirichlet assembler quadrature points"
            )
        );
        if (handler->get_bool("dirichlet assembler quadrature auto mode")) {
            dirichlet_assembler_n_quadrature_points += fe.primal.p + 1;
        }


        neumann_boundary_u_N_function = handler->get(
            "neumann boundary u_N function"
        );

        neumann_boundary_u_N_options = handler->get(
            "neumann boundary u_N options"
        );

        neumann_assembler_n_quadrature_points = static_cast<unsigned int> (
            handler->get_integer(
                "neumann assembler quadrature points"
            )
        );
        if (handler->get_bool("neumann assembler quadrature auto mode")) {
            neumann_assembler_n_quadrature_points += fe.dual.q + 1;
        }

        robin_boundary_u_R_function = handler->get(
            "robin boundary u_R function"
        );

        robin_boundary_u_R_options = handler->get(
            "robin boundary u_R options"
        );

        robin_boundary_u_R_factors = handler->get(
            "robin boundary u_R factors"
        );

        robin_assembler_n_quadrature_points = static_cast<unsigned int> (
            handler->get_integer(
                "robin assembler quadrature points"
            )
        );

        if (handler->get_bool("robin assembler quadrature auto mode")) {
            robin_assembler_n_quadrature_points += fe.dual.q + 5;
        }


        initial_value_u0_function = handler->get(
            "initial value u0 function"
        );

        initial_value_u0_options = handler->get(
            "initial value u0 options"
        );
    }
    handler->leave_subsection();
	
	
    handler->enter_subsection("Output Quantities"); {
        data_output.primal.dwr_loop = handler->get("primal data output dwr loop");

        data_output.primal.trigger_type = handler->get("primal data output trigger type");
        data_output.primal.trigger = handler->get_double("primal data output trigger time");

        if (handler->get_bool("primal data output patches auto mode")) {
            data_output.primal.patches = fe.primal.p;
        }
        else {
            data_output.primal.patches = static_cast<unsigned int> (
                handler->get_integer("primal data output patches")
            );
        }

        data_output.dual.dwr_loop = handler->get("dual data output dwr loop");

        data_output.dual.trigger_type = handler->get("dual data output trigger type");
        data_output.dual.trigger = handler->get_double("dual data output trigger time");

        if (handler->get_bool("dual data output patches auto mode")) {
            data_output.dual.patches = fe.dual.q;
        }
        else {
            data_output.dual.patches = static_cast<unsigned int> (
                handler->get_integer("dual data output patches")
            );
        }
    }
    handler->leave_subsection();
}

}} // namespace
