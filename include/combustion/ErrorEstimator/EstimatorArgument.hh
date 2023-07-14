/**
 * @file EstimatorArguments.tpl.hh
 *
 * @author Jan Philipp Thiele (JPT)
 *
 */


/*  Copyright (C) 2012-2023 by Jan Philipp Thiele                             */
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

#ifndef __EstimatorArguments_hh
#define __EstimatorArguments_hh

#include <deal.II/lac/trilinos_vector.h>

namespace combustion {
namespace dwr {


namespace estimator {
// This class provides a storage for the interpolated solutions needed
// to evaluate the partition-of-unity estimator.
class Arguments{
public:
    struct {
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector> u_kh;

        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector> z_k;
    } tm; //interval start

    struct {
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector> u_k;
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector> u_kh;

        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector> z_k;
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector> z_kh;
    } tn; //interval end

    struct {
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector> z_kh;

        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector> u_k;
    } tnp1; //next interval end
};
}}} //namespaces

#endif
