/**
 * @file L2_RobinMatrixAssembly.tpl.hh
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 */

/*  Copyright (C) 2012-2023 by Jan Philipp Thiele and Uwe Koecher             */
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


#ifndef __L2_RobinMatrixAssembly_tpl_hh
#define __L2_RobinMatrixAssembly_tpl_hh

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

// C++ includes
#include <memory>
#include <vector>

namespace combustion {
namespace Assemble {
namespace L2 {
namespace RobinMatrix {

namespace Assembly {
namespace Scratch {

template<int dim>
struct RobinMatrixAssembly {
    RobinMatrixAssembly(
        const dealii::FiniteElement<dim> &fe,
        const dealii::Quadrature<dim-1> &quad_face,
        const dealii::UpdateFlags &uflags_face
    );

    RobinMatrixAssembly(const RobinMatrixAssembly &scratch);

    dealii::FEFaceValues<dim> fe_face_values;
    std::vector<double>       phi;
    double                    JxW;
    unsigned int              dofs_per_cell;

    // other
    unsigned int face_no;
    unsigned int q;
    unsigned int k;
    unsigned int i;
    unsigned int j;
};

} // namespace Scratch
namespace CopyData {

template<int dim>
struct RobinMatrixAssembly {
    RobinMatrixAssembly(const dealii::FiniteElement<dim> &fe);
    RobinMatrixAssembly(const RobinMatrixAssembly &copydata);

    dealii::FullMatrix<double> ui_vi_matrix;
    std::vector<unsigned int> local_dof_indices;
};

} // namespace CopyData
} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////

/**
 *
 */
template<int dim>
class Assembler {
public:
    Assembler(
        std::shared_ptr< dealii::TrilinosWrappers::SparseMatrix > R,
        std::shared_ptr< dealii::DoFHandler<dim> > dof,
        std::shared_ptr< dealii::FiniteElement<dim> > fe,
        std::shared_ptr< dealii::AffineConstraints<double> > constraints
    );

    ~Assembler() = default;

    void set_parameters (
        double theta_factor,
        double Y_factor
    );

    void assemble(
        double factor,
        const unsigned int n_quadrature_points =0
    );
protected:
    void local_assemble_cell(
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
        Assembly::Scratch::RobinMatrixAssembly<dim> &scratch,
        Assembly::CopyData::RobinMatrixAssembly<dim> &copydata
    );

    void copy_local_to_global_cell(
        const Assembly::CopyData::RobinMatrixAssembly<dim> &copydata
    );
	
private:
    std::shared_ptr< dealii::TrilinosWrappers::SparseMatrix > R;

    std::shared_ptr< dealii::DoFHandler<dim> > dof;
    std::shared_ptr< dealii::FiniteElement<dim> > fe;
    std::shared_ptr< dealii::AffineConstraints<double> > constraints;
    dealii::UpdateFlags uflags_face;

    std::vector<double> factors;
    double factor;

};

}}}} // namespaces

#endif
