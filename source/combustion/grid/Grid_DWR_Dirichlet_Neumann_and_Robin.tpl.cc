/**
 * @file Grid_DWR_Dirichlet_Neumann_and_Robin.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * 
 */

/*  Copyright (C) 2012-2023 by Jan Philipp Thiele                             */
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
#include <combustion/grid/Grid_DWR_Dirichlet_Neumann_and_Robin.tpl.hh>

namespace combustion {
namespace grid {

template<int dim, int spacedim>
void
Grid_DWR_Dirichlet_Neumann_and_Robin<dim,spacedim>::
set_boundary_indicators() {
    std::string argument;
    std::vector<std::string> options;

    for ( auto& character : Grid_Class_Options)
    {
        if ( !std::isspace(character) && character!='\"')
        {
            argument += character;
        }
        else
        {
            if (argument.size()) {
                options.push_back(argument);
                argument.clear();
            }
        }
    }

    if ( argument.size())
    {
        options.push_back(argument);
        argument.clear();
    }

    AssertThrow ( options.size() == 2,
                  dealii::ExcMessage(
                    "Grid_Class_Options invalid,"
                    "please check zour input file data."
                  )
    );

    double H = std::stod(options[0]);
    double L = std::stod(options[1]);

    // set boundary indicators
    auto slab(this->slabs.begin());
    auto ends(this->slabs.end());
    unsigned int i = 0;
    for (; slab != ends; ++slab, ++i) {
        auto cell(slab->tria->begin_active());
        auto endc(slab->tria->end());

        for (; cell != endc; ++cell) {
        if (cell->at_boundary()) {
        for (unsigned int face(0);
            face < dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                dealii::Point<dim> center = cell->face(face)->center();

                if (center[0] == 0.) {
                    cell->face(face)->set_boundary_id(
                        static_cast<dealii::types::boundary_id> (
                            combustion::types::boundary_id::Neumann)
                    );
                }
                else {
                  if ( center[0] == L/4.0  ||
                       center[0] == L/2.0  ||
                       center[1] == H/4.0  ||
                       center[1] == 3.0*H/4.0)
                  {
                        cell->face(face)->set_boundary_id(
                             static_cast<dealii::types::boundary_id> (
                                combustion::types::boundary_id::Robin)
                        );
                  }
                  else {
                       cell->face(face)->set_boundary_id(
                             static_cast<dealii::types::boundary_id> (
                                combustion::types::boundary_id::Neumann)
                        );
                  }
               }
            }
        }}}
    }
}

}} // namespaces

#include "Grid_DWR_Dirichlet_Neumann_and_Robin.inst.in"
