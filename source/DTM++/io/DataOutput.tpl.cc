/**
 * @file DataOutput.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 *
 * @brief This is a template to output a VECTOR as hdf5/xdmf.
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

// DTM++ includes
#include <DTM++/io/DataOutput.tpl.hh>

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

// C++ includes
#include <memory>
#include <string>
#include <vector>
#include <fstream>


namespace DTM {

template<int dim>
DataOutput<dim>::
DataOutput() {
    set_MPI_Comm();
    set_setw();

    process_id = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
	
#ifdef DEAL_II_WITH_HDF5
    set_output_format(DataFormat::HDF5_XDMF);
#else
    set_output_format(DataFormat::invalid);
#endif
	
    set_data_output_patches();

    data_file_counter = 0;

    xdmf_entries_data.clear();
}


template<int dim>
void
DataOutput<dim>::
set_MPI_Comm(MPI_Comm _mpi_comm) {
    mpi_comm = _mpi_comm;
}


template<int dim>
void
DataOutput<dim>::
set_DoF_data(
    std::shared_ptr< dealii::DoFHandler<dim> > _dof) {
    dof = _dof;
}


template<int dim>
void
DataOutput<dim>::
set_output_format(DataFormat _format) {
    format = _format;
}


template<int dim>
void
DataOutput<dim>::
set_data_output_patches(unsigned int _data_output_patches) {
    data_output_patches = _data_output_patches;
}


template<int dim>
void
DataOutput<dim>::
set_setw(const unsigned int _setw_value) {
    setw_value = _setw_value;
}


template<int dim>
void
DataOutput<dim>::
set_data_field_names(
    std::vector<std::string> &_data_field_names) {
    data_field_names.resize(_data_field_names.size());

    auto it_data_field_names = _data_field_names.begin();
    for (auto &data_field_name : data_field_names) {
        data_field_name = *it_data_field_names;
        ++it_data_field_names;
    }

    data_field_names_process_id.resize(data_field_names.size());
    it_data_field_names = _data_field_names.begin();
    for (auto &data_field_name : data_field_names_process_id) {
        data_field_name = "process_id__" + *it_data_field_names;
        ++it_data_field_names;
    }
}


template<int dim>
void
DataOutput<dim>::
set_data_component_interpretation_field(
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> &_dci_field) {
    dci_field.resize(_dci_field.size());

    auto it_dci_field = _dci_field.begin();
    for (auto &dci : dci_field) {
        dci = *it_dci_field;
        ++it_dci_field;
    }
}


template<int dim>
void
DataOutput<dim>::
write_data(
    const std::string &solution_name,
    std::shared_ptr< VectorType > solution_vector,
    const double &time) {
    Assert(format == DataFormat::HDF5_XDMF, dealii::ExcNotImplemented());

    // prepare output filenames
    std::ostringstream stream;

    stream.str("");
    stream << solution_name << "_"
           << std::setw(setw_value) << std::setfill('0') << data_file_counter
           << ".h5";
    std::string data_filename = stream.str();

    stream.str("");
    stream << solution_name
           << ".xdmf";
    std::string xdmf_filename = stream.str();

    // create dealii::DataOut object and attach the data
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(*dof);

    Assert(
        (data_field_names.size()==2),
        dealii::ExcMessage("To use this write_data function, \
        you can only handle one solution file. \
        Please correct the number of data field names.")
    );

    data_out.add_data_vector(
        *solution_vector,
        data_field_names,
        dealii::DataOut<dim>::type_dof_data,
        dci_field
    );
    data_out.build_patches(data_output_patches);

    // Filter duplicated verticies = true, hdf5 = true
    dealii::DataOutBase::DataOutFilter data_filter(
        dealii::DataOutBase::DataOutFilterFlags(false, true)
    );

    // Filter the data and store it in data_filter
    data_out.write_filtered_data(data_filter);

    // Write the filtered data to HDF5
    data_out.write_hdf5_parallel(
        data_filter,
        data_filename.c_str(),
        mpi_comm
    );

    // Add XDMF entry
    xdmf_entries_data.push_back(
        data_out.create_xdmf_entry(
            data_filter,
            data_filename.c_str(),
            time,
            mpi_comm
        )
    );

    data_out.write_xdmf_file(
        xdmf_entries_data,
        xdmf_filename.c_str(),
        mpi_comm
    );

    data_file_counter++;
}


template<int dim>
void
DataOutput<dim>::
write_data(
    const std::string &solution_name,
    std::vector< std::shared_ptr< VectorType > > &solution_vectors,
    const double &time) {
    Assert(format == DataFormat::HDF5_XDMF, dealii::ExcNotImplemented());

    // prepare output filenames
    std::ostringstream stream;

    stream.str("");
    stream << solution_name << "_"
           << std::setw(setw_value) << std::setfill('0') << data_file_counter
           << ".h5";
    std::string data_filename = stream.str();

    stream.str("");
    stream << solution_name
            << ".xdmf";
    std::string xdmf_filename = stream.str();

    // create dealii::DataOut object and attach the data
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(*dof);
	
    {
        Assert(
            (solution_vectors.size() == data_field_names.size()),
            dealii::ExcMessage("You have given a different number of \
            solutions_vectors and data field names!")
        );
        auto data_field_name = data_field_names.begin();

        for (auto solution_vector : solution_vectors) {
            Assert(solution_vector.use_count(), dealii::ExcNotInitialized());

            data_out.add_data_vector(
                *solution_vector,
                *data_field_name
            );

            ++data_field_name;
        }
    }
	
    data_out.build_patches(data_output_patches);

    // Filter duplicated verticies = true, hdf5 = true
    dealii::DataOutBase::DataOutFilter data_filter(
        dealii::DataOutBase::DataOutFilterFlags(false, true)
    );

    // Filter the data and store it in data_filter
    data_out.write_filtered_data(data_filter);

    // Write the filtered data to HDF5
    data_out.write_hdf5_parallel(
        data_filter,
        data_filename.c_str(),
        mpi_comm
    );
	
    // Add XDMF entry
    xdmf_entries_data.push_back(
        data_out.create_xdmf_entry(
            data_filter,
            data_filename.c_str(),
            time,
            mpi_comm
        )
    );
	
    data_out.write_xdmf_file(
        xdmf_entries_data,
        xdmf_filename.c_str(),
        mpi_comm
    );

    data_file_counter++;
}


template<int dim>
void
DataOutput<dim>::
write_data(
    const std::string &solution_name,
    std::shared_ptr< VectorType > solution_vector,
    std::shared_ptr< dealii::DataPostprocessor<dim> > data_postprocessor,
    const double &time) {
    Assert(format == DataFormat::HDF5_XDMF, dealii::ExcNotImplemented());

    // prepare output filenames
    std::ostringstream stream;
	
    stream.str("");
    stream << solution_name << "_"
           << std::setw(setw_value) << std::setfill('0') << data_file_counter
           << ".h5";
    std::string data_filename = stream.str();

    stream.str("");
    stream << solution_name
           << ".xdmf";
    std::string xdmf_filename = stream.str();

    // create dealii::DataOut object and attach the data
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(*dof);

    data_out.add_data_vector(
        *solution_vector,
        *data_postprocessor
    );
    data_out.build_patches(data_output_patches);

    // Filter duplicated verticies = true, hdf5 = true
    dealii::DataOutBase::DataOutFilter data_filter(
        dealii::DataOutBase::DataOutFilterFlags(false, true)
    );

    // Filter the data and store it in data_filter
    data_out.write_filtered_data(data_filter);

    // Write the filtered data to HDF5
    data_out.write_hdf5_parallel(
        data_filter,
        data_filename.c_str(),
        mpi_comm
    );
	
    // Add XDMF entry
    xdmf_entries_data.push_back(
        data_out.create_xdmf_entry(
            data_filter,
            data_filename.c_str(),
            time,
            mpi_comm
        )
    );

    data_out.write_xdmf_file(
        xdmf_entries_data,
        xdmf_filename.c_str(),
        mpi_comm
    );

    data_file_counter++;
}

} // namespaces

#include "DataOutput.inst.in"
