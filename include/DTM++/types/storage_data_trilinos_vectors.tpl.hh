/**
 * @file storage_data_vectors.tpl.hh
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 *
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

#ifndef __storage_data_trilinos_vectors_tpl_hh
#define __storage_data_trilinos_vectors_tpl_hh

// dealii includes
#include <deal.II/lac/trilinos_vector.h>

// C++ includes
#include <array>
#include <list>

namespace DTM {
namespace types {

namespace storage {
namespace data {

template <int N>
struct s_trilinos_vectors {
    /// internal storage: N shared pointers to the data containers
    std::array< std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector>, N > x;
};

template <int N>
using trilinos_vectors = struct s_trilinos_vectors<N>;

}}

/// storage container for data vectors
template <int N>
using storage_data_trilinos_vectors = std::list< storage::data::trilinos_vectors<N> >;

}}

#endif
