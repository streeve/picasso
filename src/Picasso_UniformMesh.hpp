/****************************************************************************
 * Copyright (c) 2021 by the Picasso authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Picasso library. Picasso is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef PICASSO_UNIFORMMESH_HPP
#define PICASSO_UNIFORMMESH_HPP

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <boost/property_tree/ptree.hpp>

#include <cmath>
#include <memory>

#include <mpi.h>

namespace Picasso
{
//---------------------------------------------------------------------------//
/*!
  \class UniformMesh
  \brief Logically and spatially uniform Cartesian mesh.
 */
template <class MemorySpace>
class UniformMesh
{
  public:
    using cajita_mesh = Cajita::UniformMesh<double>;

    using memory_space = MemorySpace;

    using local_grid = Cajita::LocalGrid<cajita_mesh>;

    using node_array = Cajita::Array<double, Cajita::Node,
                                     Cajita::UniformMesh<double>, MemorySpace>;

    static constexpr std::size_t num_space_dim = 3;

    // Construct a mesh manager from the problem bounding box and a property
    // tree.
    template <class ExecutionSpace>
    UniformMesh( const boost::property_tree::ptree& ptree,
                 const Kokkos::Array<double, 6>& global_bounding_box,
                 const int minimum_halo_cell_width, MPI_Comm comm,
                 const ExecutionSpace& exec_space )
        : _global_bounding_box( global_bounding_box )
        , _minimum_halo_width( minimum_halo_cell_width )
    {
        setGlobalNumCellAndCellSize( ptree );
        setPeriodicity( ptree );
        setMinimumHaloWidth( ptree );

        auto partitioner = getPartitioner( ptree );
        build( comm, partitioner, exec_space );
    }

    // Constructor that uses the default ExecutionSpace for this MemorySpace.
    UniformMesh( const boost::property_tree::ptree& ptree,
                 const Kokkos::Array<double, 6>& global_bounding_box,
                 const int minimum_halo_cell_width, MPI_Comm comm )
        : _global_bounding_box( global_bounding_box )
        , _minimum_halo_width( minimum_halo_cell_width )

    {
        using exec_space = typename memory_space::execution_space;

        setGlobalNumCellAndCellSize( ptree );
        setPeriodicity( ptree );
        setMinimumHaloWidth( ptree );

        auto partitioner = getPartitioner( ptree );
        build( comm, partitioner, exec_space{} );
    }

    // Constructor from global number of cells (without using boost).
    UniformMesh( const Kokkos::Array<int, 3>& global_num_cell,
                 const Kokkos::Array<bool, 3>& periodic,
                 std::shared_ptr<Cajita::BlockPartitioner<3>> partitioner,
                 const Kokkos::Array<double, 6>& global_bounding_box,
                 const int minimum_halo_width, MPI_Comm comm )
        : _global_num_cell( global_num_cell )
        , _global_bounding_box( global_bounding_box )
        , _periodic( periodic )
        , _minimum_halo_width( minimum_halo_width )
    {
        using exec_space = typename memory_space::execution_space;

        setCellSize();
        build( comm, partitioner, exec_space{} );
    }

    // Constructor from global number of cells (without using boost).
    UniformMesh( const double cell_size, const Kokkos::Array<bool, 3>& periodic,
                 const std::shared_ptr<Cajita::BlockPartitioner<3>> partitioner,
                 const Kokkos::Array<double, 6>& global_bounding_box,
                 const int minimum_halo_width, MPI_Comm comm )
        : _global_bounding_box( global_bounding_box )
        , _cell_size( cell_size )
        , _periodic( periodic )
        , _minimum_halo_width( minimum_halo_width )
    {
        using exec_space = typename memory_space::execution_space;

        setGlobalNumCell();
        build( comm, partitioner, exec_space{} );
    }

  private:
    void setGlobalNumCellAndCellSize( const boost::property_tree::ptree& ptree )
    {
        const auto& mesh_params = ptree.get_child( "mesh" );

        // Get the global number of cells in each direction and the cell size.
        _cell_size = 0.0;
        if ( mesh_params.count( "cell_size" ) )
        {
            _cell_size = mesh_params.get<double>( "cell_size" );
            setGlobalNumCell();
        }
        else if ( mesh_params.count( "global_num_cell" ) )
        {
            if ( mesh_params.get_child( "global_num_cell" ).size() != 3 )
                throw std::runtime_error(
                    "3 entries required for mesh.global_num_cell" );

            int d = 0;
            for ( auto& element : mesh_params.get_child( "global_num_cell" ) )
            {
                _global_num_cell[d] = element.second.get_value<int>();
                ++d;
            }
            setCellSize();
        }
        else
        {
            throw std::runtime_error( "Invalid uniform mesh size parameters" );
        }
    }

    void setGlobalNumCell()
    {
        for ( int d = 0; d < 3; ++d )
        {
            _global_num_cell[d] = std::rint(
                ( _global_bounding_box[d + 3] - _global_bounding_box[d] ) /
                _cell_size );
        }
    }

    void setCellSize()
    {
        // Uniform cell size.
        _cell_size = ( _global_bounding_box[3] - _global_bounding_box[0] ) /
                     _global_num_cell[0];
    }

    void setPeriodicity( const boost::property_tree::ptree& ptree )
    {
        const auto& mesh_params = ptree.get_child( "mesh" );

        // Get the periodicity.
        if ( mesh_params.get_child( "periodic" ).size() != 3 )
            throw std::runtime_error( "3 entries required for mesh.periodic" );

        int d = 0;
        for ( auto& element : mesh_params.get_child( "periodic" ) )
        {
            _periodic[d] = element.second.get_value<bool>();
            ++d;
        }
    }

    void setMinimumHaloWidth( const boost::property_tree::ptree& ptree )
    {
        const auto& mesh_params = ptree.get_child( "mesh" );

        // Get the halo cell width. If the user does not assign one then it is
        // assumed the minimum halo cell width will be used.
        _minimum_halo_width = std::max(
            _minimum_halo_width, mesh_params.get<int>( "halo_cell_width", 0 ) );
    }

    auto getPartitioner( const boost::property_tree::ptree& ptree )
    {
        const auto& mesh_params = ptree.get_child( "mesh" );

        // Create the partitioner.
        const auto& part_params = mesh_params.get_child( "partitioner" );
        std::shared_ptr<Cajita::BlockPartitioner<3>> partitioner;
        if ( part_params.get<std::string>( "type" ).compare( "uniform_dim" ) ==
             0 )
        {
            partitioner = std::make_shared<Cajita::DimBlockPartitioner<3>>();
        }
        else if ( part_params.get<std::string>( "type" ).compare( "manual" ) ==
                  0 )
        {
            if ( part_params.get_child( "ranks_per_dim" ).size() != 3 )
                throw std::runtime_error(
                    "3 entries required for mesh.partitioner.ranks_per_dim " );

            std::array<int, 3> ranks_per_dim;
            int d = 0;
            for ( auto& element : part_params.get_child( "ranks_per_dim" ) )
            {
                ranks_per_dim[d] = element.second.get_value<int>();
                ++d;
            }
            partitioner = std::make_shared<Cajita::ManualBlockPartitioner<3>>(
                ranks_per_dim );
        }
        return partitioner;
    }

    template <class ExecutionSpace>
    void build( MPI_Comm comm,
                const std::shared_ptr<Cajita::BlockPartitioner<3>> partitioner,
                const ExecutionSpace& exec_space )
    {
        // Because the mesh is uniform check that the domain is evenly
        // divisible by the cell size in each dimension within round-off
        // error. This will let us do cheaper math for particle location.
        for ( int d = 0; d < 3; ++d )
        {
            double extent = _global_num_cell[d] * _cell_size;
            if ( std::abs( extent - ( _global_bounding_box[d + 3] -
                                      _global_bounding_box[d] ) ) >
                 std::numeric_limits<float>::epsilon() )
                throw std::logic_error(
                    "Extent not evenly divisible by uniform cell size" );
        }

        // Create global mesh bounds.
        std::array<double, 3> global_low_corner = { _global_bounding_box[0],
                                                    _global_bounding_box[1],
                                                    _global_bounding_box[2] };
        std::array<double, 3> global_high_corner = { _global_bounding_box[3],
                                                     _global_bounding_box[4],
                                                     _global_bounding_box[5] };

        // For dimensions that are not periodic we pad by the minimum halo
        // cell width to allow for projections outside of the domain.
        for ( int d = 0; d < 3; ++d )
        {
            if ( !_periodic[d] )
            {
                _global_num_cell[d] += 2 * _minimum_halo_width;
                global_low_corner[d] -= _cell_size * _minimum_halo_width;
                global_high_corner[d] += _cell_size * _minimum_halo_width;
            }
        }

        std::array<int, 3> gnc;
        std::array<bool, 3> p;
        for ( int d = 0; d < 3; ++d )
        {
            gnc[d] = _global_num_cell[d];
            p[d] = _periodic[d];
        }

        // Create the global mesh.
        auto global_mesh = Cajita::createUniformGlobalMesh(
            global_low_corner, global_high_corner, gnc );
        // Build the global grid.
        auto global_grid =
            Cajita::createGlobalGrid( comm, global_mesh, p, *partitioner );
        // Build the local grid.
        _local_grid =
            Cajita::createLocalGrid( global_grid, _minimum_halo_width );

        // Create the nodes.
        buildNodes( _cell_size, exec_space );
    }

  public:
    // Get the minimum required number of cells in the halo.
    int minimumHaloWidth() const { return _minimum_halo_width; }

    // Get the local grid.
    std::shared_ptr<local_grid> localGrid() const { return _local_grid; }

    // Get the mesh node coordinates.
    std::shared_ptr<node_array> nodes() const { return _nodes; }

    // Get the cell size.
    double cellSize() const
    {
        return _local_grid->globalGrid().globalMesh().cellSize( 0 );
    }

    // Build the mesh nodes.
    template <class ExecutionSpace>
    void buildNodes( const double cell_size, const ExecutionSpace& exec_space )
    {
        // Create both owned and ghosted nodes so we don't have to gather
        // initially.
        auto node_layout =
            Cajita::createArrayLayout( _local_grid, 3, Cajita::Node() );
        _nodes = Cajita::createArray<double, MemorySpace>( "mesh_nodes",
                                                           node_layout );
        auto node_view = _nodes->view();
        auto local_mesh =
            Cajita::createLocalMesh<ExecutionSpace>( *_local_grid );
        auto local_space = _local_grid->indexSpace(
            Cajita::Ghost(), Cajita::Node(), Cajita::Local() );
        Kokkos::parallel_for(
            "Picasso::UniformMesh::create_nodes",
            Cajita::createExecutionPolicy( local_space, exec_space ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                node_view( i, j, k, 0 ) =
                    local_mesh.lowCorner( Cajita::Ghost(), 0 ) + i * cell_size;
                node_view( i, j, k, 1 ) =
                    local_mesh.lowCorner( Cajita::Ghost(), 1 ) + j * cell_size;
                node_view( i, j, k, 2 ) =
                    local_mesh.lowCorner( Cajita::Ghost(), 2 ) + k * cell_size;
            } );
    }

    Kokkos::Array<int, 3> _global_num_cell;
    Kokkos::Array<double, 6> _global_bounding_box;
    double _cell_size;
    Kokkos::Array<bool, 3> _periodic;

  public:
    int _minimum_halo_width;
    std::shared_ptr<local_grid> _local_grid;
    std::shared_ptr<node_array> _nodes;
};

//---------------------------------------------------------------------------//
// Static type checker.
template <class>
struct is_uniform_mesh_impl : public std::false_type
{
};

template <class MemorySpace>
struct is_uniform_mesh_impl<UniformMesh<MemorySpace>> : public std::true_type
{
};

template <class T>
struct is_uniform_mesh
    : public is_uniform_mesh_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
// Creation functions.
template <class MemorySpace>
auto createUniformMesh(
    MemorySpace, const Kokkos::Array<int, 3>& global_num_cell,
    const Kokkos::Array<bool, 3>& periodic,
    std::shared_ptr<Cajita::BlockPartitioner<3>> partitioner,
    const Kokkos::Array<double, 6>& global_bounding_box,
    const int minimum_halo_cell_width, MPI_Comm comm )
{
    return std::make_shared<UniformMesh<MemorySpace>>(
        global_num_cell, periodic, partitioner, global_bounding_box,
        minimum_halo_cell_width, comm );
}

template <class MemorySpace>
auto createUniformMesh(
    MemorySpace, const double cell_size, const Kokkos::Array<bool, 3>& periodic,
    std::shared_ptr<Cajita::BlockPartitioner<3>> partitioner,
    const Kokkos::Array<double, 6>& global_bounding_box,
    const int minimum_halo_cell_width, MPI_Comm comm )
{
    return std::make_shared<UniformMesh<MemorySpace>>(
        cell_size, periodic, partitioner, global_bounding_box,
        minimum_halo_cell_width, comm );
}

template <class MemorySpace, class ExecSpace>
auto createUniformMesh( MemorySpace, const boost::property_tree::ptree& ptree,
                        const Kokkos::Array<double, 6>& global_bounding_box,
                        const int minimum_halo_cell_width, MPI_Comm comm,
                        ExecSpace exec_space )
{
    return std::make_shared<UniformMesh<MemorySpace>>(
        ptree, global_bounding_box, minimum_halo_cell_width, comm, exec_space );
}

template <class MemorySpace>
auto createUniformMesh( MemorySpace, const boost::property_tree::ptree& ptree,
                        const Kokkos::Array<double, 6>& global_bounding_box,
                        const int minimum_halo_cell_width, MPI_Comm comm )
{
    return std::make_shared<UniformMesh<MemorySpace>>(
        ptree, global_bounding_box, minimum_halo_cell_width, comm );
}

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_UNIFORMMESH_HPP
