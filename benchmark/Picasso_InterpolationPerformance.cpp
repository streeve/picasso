/****************************************************************************
 * Copyright (c) 2022 by the Picasso authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Picasso library. Picasso is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "Cabana_BenchmarkUtils.hpp"

#include <Picasso.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <mpi.h>

struct PolyPicTag
{
};
struct APicTag
{
};
struct FlipTag
{
};

using Node = Picasso::FieldLocation::Node;
using Particle = Picasso::FieldLocation::Particle;
using Position = Picasso::Field::LogicalPosition<3>;
struct Velocity : Picasso::Field::Vector<double, 3>
{
    static std::string label() { return "velocity"; }
};
struct PolyPicVelocity : Picasso::Field::Matrix<double, 8, 3>
{
    static std::string label() { return "velocity"; }
};
struct ApicVelocity : Picasso::Field::Matrix<double, 4, 3>
{
    static std::string label() { return "velocity"; }
};
struct PreviousVelocity : Picasso::Field::Vector<double, 3>
{
    static std::string label() { return "previous_velocity"; }
};
struct Mass : Picasso::Field::Scalar<double>
{
    static std::string label() { return "mass"; }
};

using mass_field = Picasso::FieldLayout<Node, Mass>;
using prev_vel_field = Picasso::FieldLayout<Node, PreviousVelocity>;
using scatter_deps = Picasso::ScatterDependencies<mass_field, prev_vel_field>;

// Particle indexing for PIC/FLIP vs APIC/PolyPIC vector field
template <class FieldType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<Picasso::LinearAlgebra::is_matrix<FieldType>::value, void>
    setField( FieldType& field, const int dir, const double value )
{
    field( 0, dir ) = value;
}

template <class FieldType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<Picasso::LinearAlgebra::is_vector<FieldType>::value, void>
    setField( FieldType& field, const int dir, const double value )
{
    field( dir ) = value;
}

template <int InterpolationOrder, class ParticleFieldType, class OldFieldType,
          class InterpolationType>
struct P2G;

//---------------------------------------------------------------------------//
// Project particle property/momentum to grid. PolyPIC variant
//---------------------------------------------------------------------------//
template <int InterpolationOrder, class ParticleFieldType, class OldFieldType>
struct P2G<InterpolationOrder, ParticleFieldType, OldFieldType, PolyPicTag>
{
    // Explicit time step.
    double dt;

    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh, const GatherDependencies&,
                const ScatterDependencies& scatter_deps,
                const LocalDependencies&, ParticleViewType& particle ) const
    {
        // Get particle data.
        auto f_p = Picasso::get( particle, ParticleFieldType() );
        auto v_p = Picasso::get( particle, PolyPicVelocity() );
        auto m_p = Picasso::get( particle, Mass() );
        auto x_p = Picasso::get( particle, Position() );

        // Get the scatter dependencies.
        auto m_i = scatter_deps.get( Node(), Mass() );
        auto f_i = scatter_deps.get( Node(), OldFieldType() );

        // Node interpolant.
        auto spline = Picasso::createSpline(
            Node(), Picasso::InterpolationOrder<InterpolationOrder>(),
            local_mesh, x_p, Picasso::SplineValue(),
            Picasso::SplineDistance() );

        // Interpolate mass and mass-weighted property/momentum to grid with
        // PolyPIC.
        Picasso::PolyPIC::p2g( m_p, v_p, f_p, f_i, m_i, dt, spline );
    }
};

//---------------------------------------------------------------------------//
// Project particle property/momentum to grid. APIC variant
//---------------------------------------------------------------------------//
template <int InterpolationOrder, class ParticleFieldType, class OldFieldType>
struct P2G<InterpolationOrder, ParticleFieldType, OldFieldType, APicTag>
{
    // Explicit time step.
    double dt;

    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh, const GatherDependencies&,
                const ScatterDependencies& scatter_deps,
                const LocalDependencies&, ParticleViewType& particle ) const
    {
        // Get particle data.
        auto f_p = Picasso::get( particle, ParticleFieldType() );
        auto m_p = Picasso::get( particle, Mass() );
        auto x_p = Picasso::get( particle, Position() );

        // Get the scatter dependencies.
        auto m_i = scatter_deps.get( Node(), Mass() );
        auto f_i = scatter_deps.get( Node(), OldFieldType() );

        // Node interpolant.
        auto spline = Picasso::createSpline(
            Node(), Picasso::InterpolationOrder<InterpolationOrder>(),
            local_mesh, x_p, Picasso::SplineValue(), Picasso::SplineDistance(),
            Picasso::SplineGradient() );

        // Interpolate mass and mass-weighted property/momentum to grid with
        // APIC.
        Picasso::APIC::p2g( m_p, f_p, m_i, f_i, spline );
    }
};

//---------------------------------------------------------------------------//
// Project particle property/momentum to grid. FLIP/PIC variant
//---------------------------------------------------------------------------//
template <int InterpolationOrder, class ParticleFieldType, class OldFieldType>
struct P2G<InterpolationOrder, ParticleFieldType, OldFieldType, FlipTag>
{
    // Explicit time step.
    double dt;

    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh, const GatherDependencies&,
                const ScatterDependencies& scatter_deps,
                const LocalDependencies&, ParticleViewType& particle ) const
    {
        // Get particle data.
        auto f_p = Picasso::get( particle, ParticleFieldType() );
        auto m_p = Picasso::get( particle, Mass() );
        auto x_p = Picasso::get( particle, Position() );

        // Get the scatter dependencies.
        auto m_i = scatter_deps.get( Node(), Mass() );
        auto f_i = scatter_deps.get( Node(), OldFieldType() );

        // Node interpolant.
        auto spline = Picasso::createSpline(
            Node(), Picasso::InterpolationOrder<InterpolationOrder>(),
            local_mesh, x_p, Picasso::SplineValue(),
            Picasso::SplineDistance() );

        // Interpolate mass and mass-weighted property/momentum to grid.
        Picasso::P2G::value( spline, m_p, m_i );
        Picasso::P2G::value( spline, m_p * f_p, f_i );
    }
};

//---------------------------------------------------------------------------//
// Performance test.
// Compare PIC/FLIP, APIC, and PolyPIC.
template <class Device, class ParticleVelocity, class InterpolationTag,
          int InterpolationOrder>
void performanceTest( std::ostream& stream, const std::string& test_prefix,
                      std::vector<int> cells_per_dim,
                      std::vector<int> particles_per_cell )
{
    using exec_space = typename Device::execution_space;
    using memory_space = typename Device::memory_space;

    // Ensemble size.
    int num_runs = 10;

    // Domain size setup
    std::array<double, 3> global_low_corner = { 0.0, 0.0, 0.0 };
    std::array<double, 3> global_high_corner = { 1.0, 1.0, 1.0 };
    std::array<bool, 3> is_dim_periodic = { true, true, true };

    int minimum_halo_width = 0;
    auto partitioner = std::make_shared<Cajita::DimBlockPartitioner<3>>();

    // System sizes
    int num_problem_size = cells_per_dim.size();
    int num_particles_per_cell = particles_per_cell.size();

    using mesh_type = Picasso::UniformMesh<memory_space>;
    using p2g_op = Picasso::GridOperator<mesh_type, scatter_deps>;
    using plist =
        Picasso::ParticleList<mesh_type, Position, ParticleVelocity, Mass>;

    const double pi = 4.0 * std::atan( 1.0 );
    double c = std::sqrt( 1.0 / 1.0 ); // E / rho
    const double length = global_high_corner[0] - global_low_corner[0];
    const double t = 1.0;

    auto init_func = KOKKOS_LAMBDA( const double x[3], const double,
                                    typename plist::particle_type& p )
    {
        auto u_p = Picasso::get( p, ParticleVelocity{} );
        setField( u_p, 0, std::cos( pi * ( x[0] / length - 0.5 ) ) );
    };

    for ( int ppc = 0; ppc < num_particles_per_cell; ++ppc )
    {
        // Create p2g value timers.
        std::stringstream p2g_scalar_value_time_name;
        p2g_scalar_value_time_name << test_prefix << "p2g_scalar_value_"
                                   << particles_per_cell[ppc];
        Cabana::Benchmark::Timer p2g_scalar_value_timer(
            p2g_scalar_value_time_name.str(), num_problem_size );

        std::stringstream p2g_vector_value_time_name;
        p2g_vector_value_time_name << test_prefix << "p2g_vector_value_"
                                   << particles_per_cell[ppc];
        Cabana::Benchmark::Timer p2g_vector_value_timer(
            p2g_vector_value_time_name.str(), num_problem_size );

        // Create p2g gradient timers.
        std::stringstream p2g_scalar_gradient_time_name;
        p2g_scalar_gradient_time_name << test_prefix << "p2g_scalar_gradient_"
                                      << particles_per_cell[ppc];
        Cabana::Benchmark::Timer p2g_scalar_gradient_timer(
            p2g_scalar_gradient_time_name.str(), num_problem_size );

        // Create p2g divergence timers.
        std::stringstream p2g_vector_divergence_time_name;
        p2g_vector_divergence_time_name << test_prefix
                                        << "p2g_vector_divergence_"
                                        << particles_per_cell[ppc];
        Cabana::Benchmark::Timer p2g_vector_divergence_timer(
            p2g_vector_divergence_time_name.str(), num_problem_size );

        std::stringstream p2g_tensor_divergence_time_name;
        p2g_tensor_divergence_time_name << test_prefix
                                        << "p2g_tensor_divergence_"
                                        << particles_per_cell[ppc];
        Cabana::Benchmark::Timer p2g_tensor_divergence_timer(
            p2g_tensor_divergence_time_name.str(), num_problem_size );

        for ( int n = 0; n < num_problem_size; ++n )
        {
            double cell_size = 1.0 / cells_per_dim[n];
            auto mesh = Picasso::createUniformMesh(
                memory_space(), cell_size, is_dim_periodic, partitioner,
                global_low_corner, global_high_corner, minimum_halo_width,
                MPI_COMM_WORLD );

            // Assign exact velocity
            auto exact_vel_array =
                Picasso::createArray( *mesh, Node{}, Velocity{} );
            auto exact_vel = exact_vel_array->view();
            Cajita::grid_parallel_for(
                "exact_vel", exec_space(),
                exact_vel_array->layout()->indexSpace( Cajita::Own{},
                                                       Cajita::Local{} ),
                KOKKOS_LAMBDA( const int i, const int j, const int k,
                               const int ) {
                    double x[3];
                    int index[3] = { i, j, k };
                    mesh->localGrid()->localMesh->coordinates( Cajita::Node(),
                                                               index, x );

                    exact_vel( i, j, k, 0 ) =
                        std::cos( pi * ( x[0] / length - 0.5 ) ) *
                        std::cos( pi * c * t / length );
                } );

            // Create the particles.
            int num_ppc = particles_per_cell[ppc];
            plist particles( "particles", mesh );
            Picasso::initializeParticles( Picasso::InitRandom{}, exec_space{},
                                          num_ppc, init_func, particles );

            // Now perform the p2g interpolations and time them.
            auto local_grid = mesh->localGrid();

            // Create a scalar field on the grid.
            auto scalar_layout =
                Cajita::createArrayLayout( local_grid, 1, Node() );
            auto scalar_grid_field = Cajita::createArray<double, memory_space>(
                "scalar_grid_field", scalar_layout );
            auto scalar_halo = Cajita::createHalo( Cajita::NodeHaloPattern<3>{},
                                                   -1, *scalar_grid_field );

            // Create a vector field on the grid.
            auto vector_layout =
                Cajita::createArrayLayout( local_grid, 3, Node() );
            auto vector_grid_field = Cajita::createArray<double, memory_space>(
                "vector_grid_field", vector_layout );
            auto vector_halo = Cajita::createHalo( Cajita::NodeHaloPattern<3>{},
                                                   -1, *vector_grid_field );

            // Interpolate a scalar point value to the grid.
            Cajita::ArrayOp::assign( *scalar_grid_field, 0.0, Cajita::Ghost() );

            // Run tests and time the ensemble.
            for ( int t = 0; t < num_runs; ++t )
            {
                // P2G scalar value
                auto scalar_value_p2g =
                    P2G<InterpolationOrder, ParticleVelocity, PreviousVelocity,
                        InterpolationTag>{ dt };
                p2g_scalar_value_timer.start( n );
                scalar_value_p2g.apply( "scalar_p2g", Particle{}, exec_space,
                                        fm, particles, scalar_value_p2g );
                p2g_scalar_value_timer.stop( n );

                // P2G vector value
                auto vector_value_p2g =
                    Cajita::createVectorValueP2G( vector, -0.5 );
                p2g_vector_value_timer.start( n );
                p2g( vector_value_p2g, position, position.size(),
                     Cajita::Spline<1>(), *vector_halo, *vector_grid_field );
                p2g_vector_value_timer.stop( n );

                // P2G scalar gradient
                auto scalar_grad_p2g =
                    Cajita::createScalarGradientP2G( scalar, -0.5 );
                p2g_scalar_gradient_timer.start( n );
                p2g( scalar_grad_p2g, position, position.size(),
                     Cajita::Spline<1>(), *vector_halo, *vector_grid_field );
                p2g_scalar_gradient_timer.stop( n );
            }
        }

        // Output results
        outputResults( stream, "grid_size_per_dim", cells_per_dim,
                       p2g_scalar_value_timer );
        outputResults( stream, "grid_size_per_dim", cells_per_dim,
                       p2g_vector_value_timer );
        outputResults( stream, "grid_size_per_dim", cells_per_dim,
                       p2g_scalar_gradient_timer );
        outputResults( stream, "grid_size_per_dim", cells_per_dim,
                       p2g_vector_divergence_timer );
        outputResults( stream, "grid_size_per_dim", cells_per_dim,
                       p2g_tensor_divergence_timer );
        stream << std::flush;
    }
}

//---------------------------------------------------------------------------//
// main
int main( int argc, char* argv[] )
{
    // Initialize environment
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    // Check arguments.
    if ( argc < 2 )
        throw std::runtime_error( "Incorrect number of arguments. \n \
             First argument -  file name for output \n \
             Optional second argument - run size (small or large) \n \
             \n \
             Example: \n \
             $/: ./InterpolationPerformance test_results.txt\n" );

    // Get the name of the output file.
    std::string filename = argv[1];

    // Define run sizes.
    std::string run_type = "";
    if ( argc > 2 )
        run_type = argv[2];
    std::vector<int> cells_per_dim = { 16, 32 };
    std::vector<int> particles_per_cell = { 1, 4 };
    if ( run_type == "large" )
    {
        cells_per_dim = { 16, 32, 64, 128 };
        particles_per_cell = { 1, 8, 32, 64 };
    }
    // Open the output file on rank 0.
    std::fstream file;
    file.open( filename, std::fstream::out );

    // Do everything on the default CPU.
    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_device_type = host_exec_space::device_type;
    // Do everything on the default device with default memory.
    using exec_space = Kokkos::DefaultExecutionSpace;
    using device_type = exec_space::device_type;

    // Don't run twice on the CPU if only host enabled.
    if ( !std::is_same<device_type, host_device_type>{} )
    {
        performanceTest<device_type, Velocity, FlipTag, 1>(
            file, "device_", cells_per_dim, particles_per_cell );
        performanceTest<device_type, ApicVelocity, APicTag, 1>(
            file, "device_", cells_per_dim, particles_per_cell );
        performanceTest<device_type, PolyPicVelocity, PolyPicTag, 1>(
            file, "device_", cells_per_dim, particles_per_cell );
    }
    performanceTest<host_device_type, Velocity, FlipTag, 1>(
        file, "host_", cells_per_dim, particles_per_cell );
    performanceTest<host_device_type, ApicVelocity, APicTag, 1>(
        file, "host_", cells_per_dim, particles_per_cell );
    performanceTest<host_device_type, PolyPicVelocity, PolyPicTag, 1>(
        file, "host_", cells_per_dim, particles_per_cell );

    // Close the output file on rank 0.
    file.close();

    // Finalize
    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
