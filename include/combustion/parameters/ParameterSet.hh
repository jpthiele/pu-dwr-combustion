/**
 * @file   ParameterSet.hh
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 *
 * @brief Keeps all parsed input parameters in a struct.
 */

/*  Copyright (C) 2012-2023 by Jan Philipp Thiele and Uwe Koecher             */
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

#ifndef __ParameterSet_hh
#define __ParameterSet_hh

// PROJECT includes

// DEAL.II includes
#include <deal.II/base/parameter_handler.h>

// C++ includes
#include <string>
#include <memory>

namespace combustion {
namespace dwr {

struct ParameterSet {
    ParameterSet( std::shared_ptr< dealii::ParameterHandler > handler );

    unsigned int dim;

    // problem specification
    struct {
        struct {
            std::string space_type;
            std::string space_type_support_points;
            unsigned int p;

            std::string time_type;
            std::string time_type_support_points;
            unsigned int r;

            bool high_order;
        } primal;

        struct {
            std::string space_type;
            std::string space_type_support_points;
            unsigned int q;

            std::string time_type;
            std::string time_type_support_points;
            unsigned int s;

            bool high_order;
        } dual;

    } fe;

    // mesh specification
    bool use_mesh_input_file;
    std::string mesh_input_filename;

    std::string TriaGenerator;
    std::string TriaGenerator_Options;

    std::string Grid_Class;
    std::string Grid_Class_Options;

    unsigned int global_refinement;

    // time integration
    double t0;
    double T;
    double tau_n;

    //Newton
    struct {
        unsigned int max_steps;
        double       lower_bound;
        double       rebuild;
        unsigned int line_search_steps;
        double       line_search_damping;
    } newton;

    // dwr
    struct {
        struct {
            std::string type;

            std::string functional;

            bool calc_reaction_rate;
            bool calc_rod_species_concentration;
        } goal;

        struct {
            struct {
                double reaction_rate;
                double rod_species_concentration;
            } average;
        } reference;

        struct {
            bool in_use;
            bool reduction_mode;

            unsigned int max_iterations;
            double tolerance;
            double reduction;
        } solver_control;

        unsigned int loops;
        
        struct {
            struct {
                std::string strategy; // global, fixed-fraction, Schwegler

                double top_fraction1;
                double top_fraction2;
                double bottom_fraction;
                unsigned int max_growth_factor_n_active_cells;

                double theta1; // Schwegler
                double theta2; // Schwegler
            } space;

            struct {
                std::string strategy; // global, fixed-fraction

                double top_fraction;
            } time;

            bool adaptive;
                
        } refine_and_coarsen;
    } dwr;

    // parameter specification
    double lewis_number;
    double arrhenius_alpha;
    double arrhenius_beta;
    double x_tilde;

    std::string dirichlet_boundary_u_D_function;
    std::string dirichlet_boundary_u_D_options;
    unsigned int dirichlet_assembler_n_quadrature_points;

    std::string neumann_boundary_u_N_function;
    std::string neumann_boundary_u_N_options;
    unsigned int neumann_assembler_n_quadrature_points;

    std::string robin_boundary_u_R_function;
    std::string robin_boundary_u_R_options;
    std::string robin_boundary_u_R_factors;
    unsigned int robin_assembler_n_quadrature_points;

    std::string initial_value_u0_function;
    std::string initial_value_u0_options;

    // data output
    struct {
        struct {
            std::string dwr_loop;
            std::string trigger_type;

            double trigger;
            unsigned int patches;
        } primal;

        struct {
            std::string dwr_loop;
            std::string trigger_type;

            double trigger;
            unsigned int patches;
        } dual;
    } data_output;
};

}} // namespace

#endif
