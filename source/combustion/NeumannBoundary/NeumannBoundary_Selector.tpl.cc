/**
 * @file NeumannBoundary_Selector.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhäuser (MPB)
 *
 */

/*  Copyright (C) 2012-2023 by Jan Philipp Thiele, Uwe Koecher                */
/*                          and Marius Paul Bruchhaeuser                      */
/*                                                                            */
/*  This file is part of pu-dwr-combustion                                    */
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

#include <DTM++/base/LogStream.hh>

#include <combustion/NeumannBoundary/NeumannBoundary_Selector.tpl.hh>
#include <combustion/NeumannBoundary/NeumannBoundaries.hh>

// C++ includes
#include <vector>

namespace combustion {
namespace neumann_boundary {

template<int dim>
void
Selector<dim>::
create_function(
    const std::string &_type,
    const std::string &_options,
    std::shared_ptr< dealii::Function<dim> > &function
) const {

    ////////////////////////////////////////////////////////////////////////////
    // parse the input string, arguments are splitted with spaces
    //
    std::string argument;
    std::vector< std::string > options;
    for (auto &character : _options) {
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
    ////////////////////////////////////////////////////////////////////////////
    //

    DTM::pout << "* found configuration: neumann_boundary function = " << _type << std::endl;
    DTM::pout << "* found configuration: neumann_boundary options = " << std::endl;
    for (auto &option : options) {
        DTM::pout << "\t" << option << std::endl;
    }
    DTM::pout << std::endl;

    DTM::pout << "* generating function" << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    //
    if (_type.compare("ZeroFunction") == 0) {
        AssertThrow(
            options.size() == 0,
            dealii::ExcMessage(
                "neumann_boundary options invalid, "
                "please check your input file data."
            )
        );

        function =
            std::make_shared< dealii::Functions::ZeroFunction<dim> > (2);

        DTM::pout
            << "neumann_boundary selector: created zero function" << std::endl
            << std::endl;

        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    //
    if (_type.compare("ConstantFunction") == 0) {
        AssertThrow(
            options.size() == 2,
            dealii::ExcMessage(
                "neumann_boundary options invalid, "
                "please check your input file data."
            )
        );
        std::vector<double> values;
        values.push_back(std::stod(options.at(0)));
        values.push_back(std::stod(options.at(1)));

        function = std::make_shared<
            dealii::Functions::ConstantFunction<dim> > (
            values
        );

        DTM::pout
            << "neumann_boundary selector: created ConstantFunction "
            << "as neumann_boundary function, with " << std::endl
            << "\tdn theta(x) = " << values.at(0) << " , " << std::endl
            << "\tdn Y(x) = " << values.at(1) << " . " << std::endl
            << std::endl;

        return;
    }


    ////////////////////////////////////////////////////////////////////////////
    //
    AssertThrow(
        false,
        dealii::ExcMessage("neumann_boundary function unknown, please check your input file data.")
    );
}

}} //namespaces

#include "NeumannBoundary_Selector.inst.in"
