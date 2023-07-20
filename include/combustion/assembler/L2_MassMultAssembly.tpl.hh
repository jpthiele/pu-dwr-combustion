/**
 * @file L2_MassMultAssembly.tpl.hh
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 *
 * @brief Purpose: Assemble Mass Matrix times vector (L2-integrals)
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

#ifndef __L2_MassMultAssembly_tpl_hh
#define __L2_MassMultAssembly_tpl_hh

// PROJECT includes

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/types.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

// C++ includes
#include <iterator>
#include <functional>
#include <memory>
#include <vector>

namespace combustion {
namespace Assemble {
namespace L2 {
namespace MassMult {

namespace Assembly {
namespace Scratch {

/// Struct for scratch on local cell matrix.
template<int dim>
struct MassMultAssembly {
    MassMultAssembly(
        const dealii::FiniteElement<dim> &fe,
        const dealii::Quadrature<dim> &quad,
        const dealii::UpdateFlags &uflags_cell
    );
	
    MassMultAssembly(const MassMultAssembly &scratch);

    dealii::FEValues<dim,dim>       fe_values;
    std::vector<double>         phi;
    double                      JxW;
    unsigned int                dofs_per_cell;
    std::vector<dealii::Vector<double> > old_solution_values;
	
    // other
    unsigned int q;
    unsigned int k;
    unsigned int i;
};


} // namespace Scratch
namespace CopyData {

/// Struct for copydata on local cell matrix.
template<int dim>
struct MassMultAssembly {
    MassMultAssembly(const dealii::FiniteElement<dim> &fe);
    MassMultAssembly(const MassMultAssembly &copydata);

    dealii::Vector<double> ui_vi_vector;
    std::vector<dealii::types::global_dof_index> local_dof_indices;
};


} // namespace CopyData
} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////

/**
 * The MassMult matrix \f$ \boldsymbol M \f$ for the corresponding
 * index sets \f$ 1 \le i \le N_u \f$ of test and \f$ 1 \le j \le N_u \f$ 
 * for trial basis function is given by
 * \f[
 * \boldsymbol M = ( m_{i j} )_{i j}\,,\quad
 * m_{i j} = \displaystyle \sum_{K \in \mathcal{T}_h}
 * \displaystyle \sum_{i=1}^{N_u}  \displaystyle \sum_{j=1}^{N_u}
 * \displaystyle \int_K
 * \varphi^{i}(\boldsymbol x)\, \rho\,\varphi^{j}(\boldsymbol x)\,
 * \,\text{d} \boldsymbol x\,,
 * \f]
 * where  \f$ N_u \f$ denotes the degrees of freedom in space for a single 
 * temporal degree of freedem of the fully discrete solution 
 * \f$ u_{\tau, h}^{\text{dG}} \f$.
 */
template<int dim>
class Assembler {
public:
    Assembler(
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > Mu,
        std::shared_ptr< dealii::DoFHandler<dim> > dof,
        std::shared_ptr< dealii::FiniteElement<dim> > fe,
        std::shared_ptr< dealii::AffineConstraints<double> > constraints
    );

    ~Assembler() = default;

    /** Assemble matrix. Matrix must be initialized before!
     *  If @param n_quadrature_points = 0 is given,
     *  the dynamic default fe.tensor_degree()+1 will be used.
     */
    void assemble(
    double factor,
    std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > _u,
        const unsigned int n_quadrature_points = 0
    );

protected:
    void local_assemble_cell(
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
        Assembly::Scratch::MassMultAssembly<dim> &scratch,
        Assembly::CopyData::MassMultAssembly<dim> &copydata
    );

    void copy_local_to_global_cell(
        const Assembly::CopyData::MassMultAssembly<dim> &copydata
    );

private:
    std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > Mu;

    std::shared_ptr< dealii::DoFHandler<dim> > dof;
    std::shared_ptr< dealii::FiniteElement<dim> > fe;
    std::shared_ptr< dealii::AffineConstraints<double> > constraints;
    dealii::UpdateFlags uflags;

    struct{
        std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector > u;
    } primal;

    double factor;

};

}}}} // namespaces

#endif
