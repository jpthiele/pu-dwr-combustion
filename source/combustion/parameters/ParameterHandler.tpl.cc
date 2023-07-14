/**
 * @file   ParameterHandler.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
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
#include <combustion/parameters/ParameterHandler.hh>

// DEAL.II includes
#include <deal.II/base/parameter_handler.h>

// C++ includes

namespace combustion {
namespace dwr {

ParameterHandler::
ParameterHandler() {
    declare_entry(
        "dim",
        "2",
        dealii::Patterns::Integer(),
        "dim"
    );
	
    enter_subsection("Problem Specification"); {
        declare_entry(
            "primal space type",
            "cG",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "primal space type support points",
            "canonical",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "primal p",
            "1",
            dealii::Patterns::Integer()
        );


        declare_entry(
            "primal time type",
            "dG",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "primal time type support points",
            "Gauss-Radau",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "primal r",
            "0",
            dealii::Patterns::Integer()
        );


        declare_entry(
            "dual space type",
            "cG",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "dual space type support points",
            "canonical",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "dual q",
            "2",
            dealii::Patterns::Integer()
        );


        declare_entry(
            "dual time type",
            "cG",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "dual time type support points",
            "Gauss-Lobatto",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "dual s",
            "1",
            dealii::Patterns::Integer()
        );

        declare_entry(
            "order approach",
            "mixed order",
            dealii::Patterns::Anything()
        );
    }
    leave_subsection();
	
    enter_subsection("Mesh Specification"); {
        declare_entry(
            "use mesh input file",
            "false",
            dealii::Patterns::Bool(),
            "determines whether to use an input file or a deal.II GridGenerator"
        );

        declare_entry(
            "mesh input filename",
            "./input/.empty",
            dealii::Patterns::Anything(),
            "filename of the mesh which can be read in with dealii::GridIn"
        );

        declare_entry(
            "TriaGenerator",
            "invalid",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "TriaGenerator Options",
            "invalid",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "Grid Class",
            "invalid",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "Grid Class Options",
            "invalid",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "global refinement",
            "0",
            dealii::Patterns::Integer(),
            "Global refinements of the intial mesh"
        );
    }
    leave_subsection();
	
    enter_subsection("Time Integration"); {
        declare_entry(
            "initial time",
            "0.",
            dealii::Patterns::Double(),
            "initial time t0"
        );

        declare_entry(
            "final time",
            "0.",
            dealii::Patterns::Double(),
            "final time T"
        );

        declare_entry(
            "time step size",
            "1e-2",
            dealii::Patterns::Double(),
            "initial time step size"
        );
    }
    leave_subsection();

    enter_subsection("Newton"); {
        declare_entry(
            "max steps",
            "60",
            dealii::Patterns::Integer()
        );

        declare_entry(
            "lower bound",
            "1e-7",
            dealii::Patterns::Double()
        );

        declare_entry(
            "rebuild parameter",
            "0.1",
            dealii::Patterns::Double()
        );

        declare_entry(
            "line search steps",
            "10",
            dealii::Patterns::Integer()
        );

        declare_entry(
            "line search damping",
            "0.6",
            dealii::Patterns::Double()
        );

    }
    leave_subsection();
        
    enter_subsection("DWR"); {
        declare_entry(
            "functional type",
            "average",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "functional",
            "reaction rate",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "loops",
            "2",
            dealii::Patterns::Integer()
        );

        declare_entry(
            "average reaction rate reference value",
            "0.",
            dealii::Patterns::Double()
        );

        declare_entry(
            "average rod species concentration reference value",
            "0.",
            dealii::Patterns::Double()
        );

        declare_entry(
            "reaction rate calculation",
            "true",
            dealii::Patterns::Bool()
        );

        declare_entry(
            "rod species concentration calculation",
            "true",
            dealii::Patterns::Bool()
        );

        declare_entry(
            "solver control in use",
            "false",
            dealii::Patterns::Bool()
        );

        declare_entry(
            "solver control reduction mode",
            "true",
            dealii::Patterns::Bool()
        );

        declare_entry(
            "solver control max iterations",
            "5",
            dealii::Patterns::Integer()
        );

        declare_entry(
            "solver control tolerance",
            "1e-10",
            dealii::Patterns::Double()
        );

        declare_entry(
            "solver control reduction",
            "1e-8",
            dealii::Patterns::Double()
        );


        declare_entry(
            "refine and coarsen space strategy",
            "global",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "refine and coarsen space top fraction1",
            "1.0",
            dealii::Patterns::Double()
        );

        declare_entry(
            "refine and coarsen space top fraction2",
            "0.5",
            dealii::Patterns::Double()
        );

        declare_entry(
            "refine and coarsen space bottom fraction",
            "0.0",
            dealii::Patterns::Double()
        );

        declare_entry(
            "refine and coarsen space max growth factor n_active_cells",
            "4",
            dealii::Patterns::Integer()
        );

        declare_entry(
            "refine and coarsen space Schwegler theta1",
            "1.0",
            dealii::Patterns::Double()
        );

        declare_entry(
            "refine and coarsen space Schwegler theta2",
            "0.0",
            dealii::Patterns::Double()
        );

        declare_entry(
            "refine and coarsen time strategy",
            "global",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "refine and coarsen time top fraction",
            "1.0",
            dealii::Patterns::Double()
        );
    }
    leave_subsection();
	
    enter_subsection("Parameter Specification"); {
        declare_entry(
           "Lewis Number",
           "0.0",
           dealii::Patterns::Double()
        );

        declare_entry(
           "Arrhenius alpha",
           "0.0",
           dealii::Patterns::Double()
        );

        declare_entry(
           "Arrhenius beta",
           "0.0",
           dealii::Patterns::Double()
        );

        declare_entry(
           "x_tilde",
           "0.0",
           dealii::Patterns::Double()
        );

        declare_entry(
            "dirichlet boundary u_D function",
            "invalid",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "dirichlet boundary u_D options",
            "",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "dirichlet assembler quadrature auto mode",
            "false",
            dealii::Patterns::Bool()
        );

        declare_entry(
            "dirichlet assembler quadrature points",
            "0",
            dealii::Patterns::Integer()
        );


        declare_entry(
            "neumann boundary u_N function",
            "ZeroFunction",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "neumann boundary u_N options",
            "",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "neumann assembler quadrature auto mode",
            "false",
            dealii::Patterns::Bool()
        );

        declare_entry(
            "neumann assembler quadrature points",
            "0",
            dealii::Patterns::Integer()
        );

        declare_entry(
            "robin boundary u_R function",
            "ZeroFunction",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "robin boundary u_R options",
            "",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "robin boundary u_R factors",
            "",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "robin assembler quadrature auto mode",
            "false",
            dealii::Patterns::Bool()
        );

        declare_entry(
            "robin assembler quadrature points",
            "0",
            dealii::Patterns::Integer()
        );

        declare_entry(
            "initial value u0 function",
            "invalid",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "initial value u0 options",
            "",
            dealii::Patterns::Anything()
        );

    }
    leave_subsection();
	
    enter_subsection("Output Quantities"); {
        declare_entry(
            "primal data output dwr loop",
            "all",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "primal data output trigger type",
            "fixed",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "primal data output trigger time",
            "-1.",
            dealii::Patterns::Double()
        );

        declare_entry(
            "primal data output patches auto mode",
            "true",
            dealii::Patterns::Bool(),
            "primal data output patches auto mode => using p data output patches"
        );

        declare_entry(
            "primal data output patches",
            "1",
            dealii::Patterns::Integer()
        );

        declare_entry(
            "dual data output dwr loop",
            "all",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "dual data output trigger type",
            "fixed",
            dealii::Patterns::Anything()
        );

        declare_entry(
            "dual data output trigger time",
            "-1.",
            dealii::Patterns::Double()
        );

        declare_entry(
            "dual data output patches auto mode",
            "true",
            dealii::Patterns::Bool(),
            "dual data output patches auto mode => using q data output patches"
        );

        declare_entry(
            "dual data output patches",
            "1",
            dealii::Patterns::Integer()
        );
    }
    leave_subsection();
}

}} // namespace
