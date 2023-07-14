/**
 * @file L2_LaplaceAssembly.tpl.hh
 *
 * @author Jan Philipp Thiele (PT)
 * @author Uwe Koecher (UK)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 *
 * @brief Purpose: Assemble directional derivative of Arrhenius equation @f$ (\nabla v, \epsilon \nabla u) @f$
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


#ifndef __L2_ArrheniusFunctionAssembly_tpl_hh
#define __L2_ArrheniusFunctionAssembly_tpl_hh

// PROJECT includes

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

// C++ includes
#include <memory>
#include <vector>


namespace combustion {
namespace Assemble {
namespace L2 {
namespace Arrhenius{
namespace Function{
        
namespace Assembly {
namespace Scratch {

/// Struct for scratch on local cell matrix.
template<int dim>
struct FunctionAssembly {
    FunctionAssembly(
        const dealii::FiniteElement<dim> &fe,
        const dealii::Quadrature<dim> &quad,
        const dealii::UpdateFlags &uflags_cell
    );

    FunctionAssembly(const FunctionAssembly &scratch);

    dealii::FEValues<dim>       fe_values;
    std::vector<double>         phi;
    unsigned int                dofs_per_cell;
    double                      JxW;
    std::vector<dealii::Vector<double> > old_solution_values;
    double                               theta;
    double                               Y;
    double                               tm1;

    // other
    unsigned int q;
    unsigned int k;
    unsigned int i;
};

} // namespace Scratch
namespace CopyData {

/// Struct for copydata on local cell matrix.
template<int dim>
struct FunctionAssembly{
    FunctionAssembly(const dealii::FiniteElement<dim> &fe);
    FunctionAssembly(const FunctionAssembly &copydata);

    dealii::Vector<double> fi_vi_vector;
    std::vector<dealii::types::global_dof_index> local_dof_indices;
};

} // namespace CopyData
} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////

template<int dim>
class Assembler {
public:
    Assembler(
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > f,
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
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > _u,
        const unsigned int n_quadrature_points = 0
    );

    void set_parameters (
        double alpha,
        double beta,
        double Le
    );
                
	
protected:
    void local_assemble_cell(
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
        Assembly::Scratch::FunctionAssembly<dim> &scratch,
        Assembly::CopyData::FunctionAssembly<dim> &copydata
    );

    void copy_local_to_global_cell(
        const Assembly::CopyData::FunctionAssembly<dim> &copydata
    );

private:
    std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > f;

    std::shared_ptr< dealii::DoFHandler<dim> > dof;
    std::shared_ptr< dealii::FiniteElement<dim> > fe;
    std::shared_ptr< dealii::AffineConstraints<double> > constraints;
    dealii::UpdateFlags uflags;

    struct{
        std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector > u;
    } primal;

    struct {
        double alpha;
        double beta;
        double Le;
        double c;
    } param;
};

}}}}} // namespaces


#endif
