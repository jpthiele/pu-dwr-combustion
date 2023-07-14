/**
 * @file TriaGenerator.tpl.cc
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

// DTM++ includes
#include <DTM++/base/LogStream.hh>

// Project includes
#include <combustion/grid/TriaGenerator.tpl.hh>

// DEAL.II includes
#include <deal.II/grid/grid_generator.h>

// class declaration
namespace combustion {

/** Generates sophisticated triangulations.
 * <ul>
 * <li>SV_ParallelRods: generates the domain as described in the publication by
 * Schmich & Vexler (see Readme) </li>
 * 
 * This function can easily be extended to other triangulations/geometrical mesh
 * descriptions. See the original dwr-diffusion for other examples
 */
template<int dim>
void
TriaGenerator<dim>::
generate(
    const std::string &TriaGenerator_Type,
    const std::string &TriaGenerator_Options,
    std::shared_ptr< dealii::Triangulation<dim> > tria ) {
    // check if tria is initialized
    Assert(
        tria.use_count(),
        dealii::ExcNotInitialized()
    );

    ////////////////////////////////////////////////////////////////////////////
    // parse the input string, arguments are splitted with spaces
    //
    std::string argument;
    std::vector< std::string > options;
    for (auto &character : TriaGenerator_Options) {
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

    if (TriaGenerator_Type.compare("SV_ParallelRods") == 0)
    {
        AssertThrow(dim==2,
            dealii::ExcMessage("dim = 2 needed for ParallelRod Grid")
        );

        AssertThrow(
            options.size() == 2,
            dealii::ExcMessage(
                    "TriaGenerator Options invalid, "
                    "please check your input file data."
            )
        );

        double H = std::stod(options.at(0));
        double L = std::stod(options.at(1));
        int n_x = 16;
        int n_y = 4;

        std::vector<std::vector<double>> spacing;

        std::vector<double> x_spacing;
        for ( int i = 0 ; i < n_x ; i++ )
        x_spacing.push_back(L/n_x);
        spacing.push_back(x_spacing);

        std::vector<double> y_spacing;
        for ( int i = 0 ; i < n_y ; i++)
        y_spacing.push_back(H/n_y);
        spacing.push_back(y_spacing);

        dealii::Point<dim> p(0.0,0.0);

        dealii::TableIndices<dim> indices;
        indices[0] = 16;
        indices[1] = 4;

        dealii::Table<dim,dealii::types::material_id> mats;
        mats.reinit(indices);

        for ( int i = 4; i < 8 ; i++ )
        {
            dealii::TableIndices<dim> local;
            local[0] = i;
            local[1] = 0;
            mats(local) = -1;
            local[1] = 3;
            mats(local) = -1;
        }
        dealii::GridGenerator::subdivided_hyper_rectangle(*tria,
                                                          spacing,
                                                          p,
                                                          mats);

        return;
    }


    ////////////////////////////////////////////////////////////////////////////
    //
    AssertThrow(
        false,
        dealii::ExcMessage("TriaGenerator_Type unknown, please check your input file data.")
    );
	
}

} // namespaces

#include "TriaGenerator.inst.in"
