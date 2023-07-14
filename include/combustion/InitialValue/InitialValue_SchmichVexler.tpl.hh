/**
 * @file InitialValue_SchmichVexler.tpl.hh
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
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

#ifndef __InitialValue_SchmichVexler_tpl_hh
#define __InitialValue_SchmichVexler_tpl_hh

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

namespace combustion {
namespace initial_value {

/**
 * component 0 implements the initial value for the temperature \f$ \theta : \Omega \times {0} \to \mathbb{R} \f$,
 * \f$ \Omega \subset \mathbb{R}^2 \f$, as given by:
 * \f[
 *  \theta(x,0) = \begin{cases}
 *      1 &\text{for }  x_1 \leq \tilde{x}_1\\
 *      \exp(\tilde{x}_1-x_1) & \text{for } x_1 > \tilde{x}_1
 * \f]
 * 
 * component 1 implements the initial value for the species concentration 
 * \f$ Y : \Omega \times {0} \to \mathbb{R} \f$,
 * \f$ \Omega \subset \mathbb{R}^2 \f$, as given by:
 * \f[
 *  Y(x,0) = \begin{cases}
 *      0 &\text{for }  x_1 \leq \tilde{x}_1\\
 *      1-\exp(Le(\tilde{x}_1-x_1)) & \text{for } x_1 > \tilde{x}_1
 * \f]
 * with the parameter values \f$ \tilde{x}_1 = 9 \f$ and \f$ Le = 1 \f$ for example.
 * 
 */
template<int dim>
class SchmichVexler : public dealii::Function<dim> {
public:
    SchmichVexler(
        const double &x_tilde,
        const double &Le
    ) : dealii::Function<dim> (2), x_tilde(x_tilde),  Le(Le)
    {};

    virtual ~SchmichVexler() = default;

    virtual
    double
    value(
        const dealii::Point<dim> &x,
        const unsigned int c
    ) const;

private:
    const double x_tilde;
    const double Le;
};

}}

#endif
