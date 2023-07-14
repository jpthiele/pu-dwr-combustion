/**
 * @file L2_Je_reaction_rate_Assembly.tpl.hh
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhäuser (MPB)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 *
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

#ifndef __L2_Je_reaction_rate_Assembly_tpl_hh
#define __L2_Je_reaction_rate_Assembly_tpl_hh

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/trilinos_vector.h>

// C++ includes
#include <iterator>
#include <functional>
#include <memory>
#include <vector>

namespace combustion {
namespace Assemble {
namespace L2 {
namespace Je_reaction_rate {

namespace Assembly {
namespace Scratch {

template<int dim>
struct Je_reaction_rateAssembly {
    Je_reaction_rateAssembly(
        const dealii::FiniteElement<dim> &fe,
        const dealii::Quadrature<dim> &quad,
        const dealii::UpdateFlags &uflags
    );

    Je_reaction_rateAssembly(const Je_reaction_rateAssembly &scratch);

    dealii::FEValues<dim>                   fe_values;
    std::vector<double>       phi;
    unsigned int                            dofs_per_cell;
    double                                  JxW;
    std::vector<dealii::Vector<double> >    old_solution_values;
    double                                  theta;
    double                                  Y;
    double                                  tm1;

    // other
    unsigned int q;
    unsigned int k;
    unsigned int i;
};

} // namespace Scratch
namespace CopyData {

template<int dim>
struct Je_reaction_rateAssembly {
    Je_reaction_rateAssembly(const dealii::FiniteElement<dim> &fe);
    Je_reaction_rateAssembly(const Je_reaction_rateAssembly &copydata);

    dealii::Vector<double> vi_Jei_vector;
    std::vector<unsigned int> local_dof_indices;
};

} // namespace CopyData
} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////

template<int dim>
class Assembler {
public:
    Assembler(
        std::shared_ptr< dealii::DoFHandler<dim> > dof,
        std::shared_ptr< dealii::FiniteElement<dim> > fe,
        std::shared_ptr< dealii::AffineConstraints<double> > constraints
    );

    ~Assembler() = default;

    /** Assemble vector.
     *  If @param n_quadrature_points = 0 is given,
     *  the dynamic default fe.tensor_degree()+1 will be used.
     */
    void assemble(
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > Je,
        std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > u_h,
        const unsigned int n_quadrature_points = 0,
        const bool quadrature_points_auto_mode = true
    );

    void set_parameters (
        double alpha,
        double beta,
        double Le,
        double area
    );


protected:
    void local_assemble_cell(
        const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
        Assembly::Scratch::Je_reaction_rateAssembly<dim> &scratch,
        Assembly::CopyData::Je_reaction_rateAssembly<dim> &copydata
    );

    void copy_local_to_global_cell(
        const Assembly::CopyData::Je_reaction_rateAssembly<dim> &copydata
    );

private:
    std::shared_ptr< dealii::DoFHandler<dim> > dof;
    std::shared_ptr< dealii::FiniteElement<dim> > fe;
    std::shared_ptr< dealii::AffineConstraints<double> > constraints;
    dealii::UpdateFlags uflags;

    std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > Je;

    struct{
        std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector > u;
    } primal;

    double domain_area;

    struct {
        double alpha;
        double beta;
        double c;
    } param;
};

}}}} // namespaces

#endif
