/**
 * @file InitialValue_Selector.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhäuser (MPB)
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

#include <DTM++/base/LogStream.hh>

#include <combustion/InitialValue/InitialValue_Selector.tpl.hh>
#include <combustion/InitialValue/InitialValues.hh>

// C++ includes
#include <vector>

namespace combustion {
namespace initial_value {

template<int dim>
void
Selector<dim>::
create_functions(
    const std::string &_type,
    const std::string &_options,
    std::shared_ptr< dealii::Function<dim> > &u0
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

    DTM::pout << "* found configuration: initial_value function = " << _type << std::endl;
    DTM::pout << "* found configuration: initial_value options = " << std::endl;
    for (auto &option : options) {
        DTM::pout << "\t" << option << std::endl;
    }
    DTM::pout << std::endl;

    DTM::pout << "* generating function" << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    //
    if (_type.compare("InitialValue_SchmichVexler") == 0) {
        AssertThrow(
            options.size() == 2,
            dealii::ExcMessage(
                "initial_value options invalid, "
                "please check your input file data."
            )
        );

        u0  = std::make_shared< combustion::initial_value::SchmichVexler<dim> >(
            std::stod(options.at(0)), //x_tilde
            std::stod(options.at(1)) //Le
        );

        DTM::pout
            << "initial_value selector: created SchmichVexler "
            << "as initial_value functions, with " << std::endl
            << "\tx_tilde = " << std::stod(options.at(0)) << std::endl
            << "\tLe = " << std::stod(options.at(1)) << std::endl
            << std::endl;

        return;
    }


    ////////////////////////////////////////////////////////////////////////////
    //
    AssertThrow(
        false,
        dealii::ExcMessage("initial_value function unknown, please check your input file data.")
    );
}

}} //namespaces

#include "InitialValue_Selector.inst.in"
