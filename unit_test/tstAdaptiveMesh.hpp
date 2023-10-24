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

#include <Picasso_AdaptiveMesh.hpp>
#include <Picasso_InputParser.hpp>
#include <Picasso_Types.hpp>

#include <Picasso_ParticleInit.hpp>

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
void constructionTest()
{
    // Global parameters.
    Kokkos::Array<double, 6> global_box = { -10.0, -10.0, -10.0,
                                            10.0,  10.0,  10.0 };
    double cell_size = 0.5;
    int num_cell = 20.0 / cell_size;
    int minimum_halo_size = 1;

    // Get inputs for mesh.
    InputParser parser_1( "adaptive_mesh_test_1.json", "json" );
    auto pt1 = parser_1.propertyTree();

    // Make mesh 1.
    AdaptiveMesh<TEST_MEMSPACE> mesh_1( pt1, global_box, minimum_halo_size,
                                        MPI_COMM_WORLD );

    // Check grid 1.
    const auto& global_grid_1 = mesh_1.localGrid()->globalGrid();
    const auto& global_mesh_1 = global_grid_1.globalMesh();

    EXPECT_EQ( global_mesh_1.lowCorner( 0 ), 0.0 );
    EXPECT_EQ( global_mesh_1.lowCorner( 1 ), -1.0 );
    EXPECT_EQ( global_mesh_1.lowCorner( 2 ), 0.0 );

    EXPECT_EQ( global_mesh_1.highCorner( 0 ), num_cell );
    EXPECT_EQ( global_mesh_1.highCorner( 1 ), num_cell + 1 );
    EXPECT_EQ( global_mesh_1.highCorner( 2 ), num_cell );

    EXPECT_EQ( global_grid_1.globalNumEntity( Cabana::Grid::Cell(), 0 ),
               num_cell );
    EXPECT_EQ( global_grid_1.globalNumEntity( Cabana::Grid::Cell(), 1 ),
               num_cell + 2 );
    EXPECT_EQ( global_grid_1.globalNumEntity( Cabana::Grid::Cell(), 2 ),
               num_cell );

    EXPECT_TRUE( global_grid_1.isPeriodic( 0 ) );
    EXPECT_FALSE( global_grid_1.isPeriodic( 1 ) );
    EXPECT_TRUE( global_grid_1.isPeriodic( 2 ) );

    EXPECT_EQ( mesh_1.localGrid()->haloCellWidth(), 2 );

    // Check grid 1 nodes.
    const auto& nodes_1 = mesh_1.nodes();
    auto host_coords_1 = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), nodes_1->view() );
    auto local_space_1 = nodes_1->layout()->localGrid()->indexSpace(
        Cabana::Grid::Ghost(), Cabana::Grid::Node(), Cabana::Grid::Local() );
    auto local_mesh_1 = Cabana::Grid::createLocalMesh<TEST_EXECSPACE>(
        *( nodes_1->layout()->localGrid() ) );
    for ( int i = local_space_1.min( 0 ); i < local_space_1.max( 0 ); ++i )
        for ( int j = local_space_1.min( 1 ); j < local_space_1.max( 1 ); ++j )
            for ( int k = local_space_1.min( 2 ); k < local_space_1.max( 2 );
                  ++k )
            {
                EXPECT_EQ( host_coords_1( i, j, k, 0 ),
                           local_mesh_1.lowCorner( Cabana::Grid::Ghost(), 0 ) +
                               i * cell_size );
                EXPECT_EQ( host_coords_1( i, j, k, 1 ),
                           local_mesh_1.lowCorner( Cabana::Grid::Ghost(), 1 ) +
                               j * cell_size );
                EXPECT_EQ( host_coords_1( i, j, k, 2 ),
                           local_mesh_1.lowCorner( Cabana::Grid::Ghost(), 2 ) +
                               k * cell_size );
            }

    // Get inputs for mesh 2.
    InputParser parser_2( "adaptive_mesh_test_2.json", "json" );
    auto pt2 = parser_2.propertyTree();

    // Make mesh 2.
    AdaptiveMesh<TEST_MEMSPACE> mesh_2( pt2, global_box, minimum_halo_size,
                                        MPI_COMM_WORLD );

    // Check grid 2.
    const auto& global_grid_2 = mesh_2.localGrid()->globalGrid();
    const auto& global_mesh_2 = global_grid_2.globalMesh();

    EXPECT_EQ( global_mesh_2.lowCorner( 0 ), -1.0 );
    EXPECT_EQ( global_mesh_2.lowCorner( 1 ), 0.0 );
    EXPECT_EQ( global_mesh_2.lowCorner( 2 ), -1.0 );

    EXPECT_EQ( global_mesh_2.highCorner( 0 ), num_cell + 1 );
    EXPECT_EQ( global_mesh_2.highCorner( 1 ), num_cell );
    EXPECT_EQ( global_mesh_2.highCorner( 2 ), num_cell + 1 );

    EXPECT_EQ( global_grid_2.globalNumEntity( Cabana::Grid::Cell(), 0 ),
               num_cell + 2 );
    EXPECT_EQ( global_grid_2.globalNumEntity( Cabana::Grid::Cell(), 1 ),
               num_cell );
    EXPECT_EQ( global_grid_2.globalNumEntity( Cabana::Grid::Cell(), 2 ),
               num_cell + 2 );

    EXPECT_FALSE( global_grid_2.isPeriodic( 0 ) );
    EXPECT_TRUE( global_grid_2.isPeriodic( 1 ) );
    EXPECT_FALSE( global_grid_2.isPeriodic( 2 ) );

    EXPECT_EQ( mesh_2.localGrid()->haloCellWidth(), 1 );

    // Check grid 2 nodes.
    const auto& nodes_2 = mesh_2.nodes();
    auto host_coords_2 = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), nodes_2->view() );
    auto local_space_2 = nodes_2->layout()->localGrid()->indexSpace(
        Cabana::Grid::Ghost(), Cabana::Grid::Node(), Cabana::Grid::Local() );
    auto local_mesh_2 = Cabana::Grid::createLocalMesh<TEST_EXECSPACE>(
        *( nodes_2->layout()->localGrid() ) );
    for ( int i = local_space_2.min( 0 ); i < local_space_2.max( 0 ); ++i )
        for ( int j = local_space_2.min( 1 ); j < local_space_2.max( 1 ); ++j )
            for ( int k = local_space_2.min( 2 ); k < local_space_2.max( 2 );
                  ++k )
            {
                EXPECT_EQ( host_coords_2( i, j, k, 0 ),
                           local_mesh_2.lowCorner( Cabana::Grid::Ghost(), 0 ) +
                               i * cell_size );
                EXPECT_EQ( host_coords_2( i, j, k, 1 ),
                           local_mesh_2.lowCorner( Cabana::Grid::Ghost(), 1 ) +
                               j * cell_size );
                EXPECT_EQ( host_coords_2( i, j, k, 2 ),
                           local_mesh_2.lowCorner( Cabana::Grid::Ghost(), 2 ) +
                               k * cell_size );
            }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, construction_test ) { constructionTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
