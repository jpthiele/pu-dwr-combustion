/**
 * @file main.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 *
 * @brief PU-DWR-Diffusion: Solve the combustion-eq with PU-DWR adaptivity.
 * 
 * @mainpage
 * The pu-dwr-combustion project is written to
 * simulate the combustion equation
 * using goal-oriented space-time adaptive finite element methods.
 * 
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

// DEFINES

////////////////////////////////////////////////////////////////////////////////
//Change this to use thread parallel assembly with Workstream
#define MPIX_THREADS 1
////////////////////////////////////////////////////////////////////////////////

// PROJECT includes
#include <DTM++/base/LogStream.hh>
#include <DTM++/base/Problem.hh>

#include <combustion/parameters/ParameterHandler.hh>
#include <combustion/Combustion_DWR__cGp_dG0__cGq_dG0.tpl.hh>

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>

// C++ includes
#include <iostream>
#include <fstream>
#include <memory>

int main(int argc, char *argv[]) {

    // Init MPI (or MPI+X)
    dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, MPIX_THREADS);

    // EVALUATE wall time now.
    auto wall_time_start = MPI_Wtime();

    // Prepare DTM++ process logging to file
    DTM::pout.open();


    // Get MPI Variables
    const unsigned int MyPID(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
    const unsigned int NumProc(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));

    //
    ////////////////////////////////////////////////////////////////////////////
	
    try {
        ////////////////////////////////////////////////////////////////////////
        // Init application
        //

        // Attach deallog to process output
        dealii::deallog.attach(DTM::pout);
        dealii::deallog.depth_console(0);
        DTM::pout
            << "****************************************"
            << "****************************************"
            << std::endl;

        DTM::pout
            << "Hej, here is process " << MyPID+1 << " from " << NumProc << std::endl;

        // Check input arguments
        AssertThrow(
            !(argc < 2),
            dealii::ExcMessage (
                    std::string ("===>\tUSAGE: ./dwr-combustion <Input_Parameter_File.prm>"))
        );

        // Check if the given input parameter file can be opened
        const std::string input_parameter_filename(argv[1]);
        {
            std::ifstream input_parameter_file(input_parameter_filename.c_str());
            AssertThrow(
                input_parameter_file,
                dealii::ExcMessage (
                    std::string ("===>\tERROR: Input parameter file <")
                    + input_parameter_filename + "> not found."
                )
            );
        }

        // Prepare input parameter handling:
        auto parameter_handler =
            std::make_shared< combustion::dwr::ParameterHandler > ();
        parameter_handler->parse_input(argv[1]);


        // Get minimal set of input parameters to drive the correct simulator
        unsigned int dimension;
        {
            // Problem dimension
            dimension = static_cast<unsigned int> (
                parameter_handler->get_integer("dim")
            );
        }

        //
        ////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // Begin application
        //

        // create simulator
        std::shared_ptr< DTM::Problem > problem;

        // select simulator
        {
            DTM::pout
                << "dwr-combustion solver: primal cG(p)-dG(0) with dual cG(q)-cG(1)"
                << std::endl;

            switch (dimension) {
            case 2: {
                problem =
                    std::make_shared< combustion::Combustion_DWR__cGp_dG0__cGq_dG0<2> > ();
                break;
            }

            case 3: {
                problem =
                    std::make_shared< combustion::Combustion_DWR__cGp_dG0__cGq_dG0<3> > ();
                break;
            }

            default:
                dealii::ExcNotImplemented();
            }
        }

        DTM::pout
            << "dwr-combustion: dimension dim = " << dimension << std::endl
            << std::endl;

        DTM::pout
            << std::endl
            << "*********************************************************"
            << std::endl << std::endl;

        // run the simulator
        problem->set_input_parameters(parameter_handler);

        problem->run();

        DTM::pout << std::endl << "Goodbye." << std::endl;

        //
        // End application
        ////////////////////////////////////////////////////////////////////////////
    }
    catch (std::exception &exc) {
        // EVALUATE program run time in terms of the consumed wall time.
        auto wall_time_end = MPI_Wtime();
        DTM::pout
            << std::endl
            << "Elapsed wall time: " << wall_time_end-wall_time_start
            << std::endl;

        if (!dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) {
            std::cerr
                << std::endl
                << "****************************************"
                << "****************************************"
                << std::endl << std::endl
                << "An EXCEPTION occured: Please READ the following output CAREFULLY!"
                << std::endl;

            std::cerr << exc.what() << std::endl;

            std::cerr
                << std::endl
                << "APPLICATION TERMINATED unexpectedly due to an exception."
                << std::endl << std::endl
                << "****************************************"
                << "****************************************"
                << std::endl << std::endl;
        }

        // LOG error message to individual process output file.
        DTM::pout
            << std::endl
            << "****************************************"
            << "****************************************"
            << std::endl << std::endl
            << "An EXCEPTION occured: Please READ the following output CAREFULLY!"
            << std::endl;

        DTM::pout << exc.what() << std::endl;

        DTM::pout
            << std::endl
            << "APPLICATION TERMINATED unexpectedly due to an exception."
            << std::endl << std::endl
            << "****************************************"
            << "****************************************"
            << std::endl << std::endl;

        // Close output file stream
        DTM::pout.close();

        return 1;
    }
    catch (...) {
        // EVALUATE program run time in terms of the consumed wall time.
        auto wall_time_end = MPI_Wtime();
        DTM::pout
            << std::endl
            << "Elapsed wall time: " << wall_time_end-wall_time_start
            << std::endl
        ;

        if (!dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) {
            std::cerr
                << std::endl
                << "****************************************"
                << "****************************************"
                << std::endl << std::endl
                << "An UNKNOWN EXCEPTION occured!"
                << std::endl;

            std::cerr
                << std::endl
                << "----------------------------------------"
                << "----------------------------------------"
                << std::endl << std::endl
                << "Further information:" << std::endl
                << "\tThe main() function catched an exception"
                << std::endl
                << "\twhich is not inherited from std::exception."
                << std::endl
                << "\tYou have probably called 'throw' somewhere,"
                << std::endl
                << "\tif you do not have done this, please contact the authors!"
                << std::endl << std::endl
                << "----------------------------------------"
                << "----------------------------------------"
                << std::endl;

            std::cerr
                << std::endl
                << "APPLICATION TERMINATED unexpectedly due to an exception."
                << std::endl << std::endl
                << "****************************************"
                << "****************************************"
                << std::endl << std::endl;
        }

        // LOG error message to individual process output file.
        DTM::pout
            << std::endl
            << "****************************************"
            << "****************************************"
            << std::endl << std::endl
            << "An UNKNOWN EXCEPTION occured!"
            << std::endl;

        DTM::pout
            << std::endl
            << "----------------------------------------"
            << "----------------------------------------"
            << std::endl << std::endl
            << "Further information:" << std::endl
            << "\tThe main() function catched an exception"
            << std::endl
            << "\twhich is not inherited from std::exception."
            << std::endl
            << "\tYou have probably called 'throw' somewhere,"
            << std::endl
            << "\tif you do not have done this, please contact the authors!"
            << std::endl << std::endl
            << "----------------------------------------"
            << "----------------------------------------"
            << std::endl;

        DTM::pout
            << std::endl
            << "APPLICATION TERMINATED unexpectedly due to an exception."
            << std::endl << std::endl
            << "****************************************"
            << "****************************************"
            << std::endl << std::endl;

        // Close output file stream
        DTM::pout.close();

        return 1;
    }

    // EVALUATE program run time in terms of the consumed wall time.
    auto wall_time_end = MPI_Wtime();
    DTM::pout
        << std::endl
        << "Elapsed wall time: " << wall_time_end-wall_time_start
        << std::endl;

    // Close output file stream
    DTM::pout.close();
	
    return 0;
}
