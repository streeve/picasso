#ifndef PICASSO_BATCHEDLINEARALGEBRA_HPP
#define PICASSO_BATCHEDLINEARALGEBRA_HPP

#include <Kokkos_Core.hpp>

#include <Kokkos_ArithTraits.hpp>

#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_Copy_Impl.hpp>

#include <KokkosBatched_Set_Decl.hpp>
#include <KokkosBatched_Set_Impl.hpp>

#include <KokkosBatched_Scale_Decl.hpp>
#include <KokkosBatched_Scale_Impl.hpp>

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Serial_Impl.hpp>

#include <KokkosBatched_Gemv_Decl.hpp>
#include <KokkosBatched_Gemv_Serial_Impl.hpp>

#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_LU_Serial_Impl.hpp>

#include <KokkosBatched_SolveLU_Decl.hpp>

#include <type_traits>

namespace Picasso
{
namespace LinearAlgebra
{
//---------------------------------------------------------------------------//
// Transpose tags.
struct NoTranspose
{
    using type = KokkosBatched::Trans::NoTranspose;
};

struct Transpose
{
    using type = KokkosBatched::Trans::Transpose;
};

//---------------------------------------------------------------------------//
// Matrix
//---------------------------------------------------------------------------//
// Dense matrix in row-major order with a KokkosKernels compatible data
// interface.
template<class T, int M, int N, class TransposeType = NoTranspose>
struct Matrix;

// No transpose
template<class T, int M, int N>
struct Matrix<T,M,N,NoTranspose>
{
    T _d[M][N];
    int _extent[2] = {M,N};

    using value_type = T;
    using non_const_value_type = typename std::remove_const<T>::type;
    using pointer = T*;
    using reference = T&;
    using const_reference = typename std::add_const<T>::type&;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    Matrix() = default;

    // Initializer list constructor.
    KOKKOS_INLINE_FUNCTION
    Matrix( const std::initializer_list<std::initializer_list<T>> data )
    {
        int i = 0;
        int j = 0;
        for ( const auto& row : data )
        {
            j = 0;
            for ( const auto& value : row )
            {
                _d[i][j] = value;
                ++j;
            }
            ++i;
        }
    }

    // Deep copy constructor.
    KOKKOS_INLINE_FUNCTION
    Matrix( const Matrix<T,M,N,NoTranspose>& rhs )
    {
        KokkosBatched::SerialCopy<NoTranspose::type>::invoke(
            rhs, *this );
    }

    // Deep copy transpose constructor.
    KOKKOS_INLINE_FUNCTION
    Matrix( const Matrix<T,N,M,Transpose>& rhs )
    {
        KokkosBatched::SerialCopy<Transpose::type>::invoke(
            rhs, *this );
    }

    // Scalar constructor.
    KOKKOS_INLINE_FUNCTION
    Matrix( const T value )
    {
        KokkosBatched::SerialSet::invoke( value, *this );
    }

    // Deep copy assignment operator.
    KOKKOS_INLINE_FUNCTION
    Matrix& operator=( const Matrix<T,M,N,NoTranspose>& rhs )
    {
        KokkosBatched::SerialCopy<NoTranspose::type>::invoke(
            rhs, *this );
        return *this;
    }

    // Deep copy transpose assignment operator.
    KOKKOS_INLINE_FUNCTION
    Matrix& operator=( const Matrix<T,M,N,Transpose>& rhs )
    {
        KokkosBatched::SerialCopy<Transpose::type>::invoke(
            rhs, *this );
        return *this;
    }

    // Scalar value assignment.
    KOKKOS_INLINE_FUNCTION
    Matrix& operator=( const T value )
    {
        KokkosBatched::SerialSet::invoke( value, *this );
        return *this;
    }

    // Transpose operator.
    KOKKOS_INLINE_FUNCTION
    Matrix<T,M,N,Transpose> operator~()
    {
        return Matrix<T,M,N,Transpose>( this->data() );
    }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_0() const
    { return N; }

    KOKKOS_INLINE_FUNCTION
    int stride_1() const
    { return 1; }

    // Extent
    KOKKOS_INLINE_FUNCTION
    int extent( const int d ) const
    { return _extent[d]; }

    // Access an individual element.
    KOKKOS_INLINE_FUNCTION
    const_reference operator()( const int i, const int j ) const
    { return _d[i][j]; }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i, const int j )
    { return _d[i][j]; }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const
    { return const_cast<pointer>(&_d[0][0]); }
};

// Transpose. This class is essentially a shallow-copy placeholder to enable
// transpose matrix operations without copies.
template<class T, int M, int N>
struct Matrix<T,M,N,Transpose>
{
    T* _d;
    int _extent[2] = {M,N};

    using value_type = T;
    using non_const_value_type = typename std::remove_const<T>::type;
    using pointer = T*;
    using reference = T&;
    using const_reference = typename std::add_const<T>::type&;

    // Pointer constructor.
    KOKKOS_INLINE_FUNCTION
    Matrix( T* data )
        : _d( data )
    {}

    // Deep copy constructor.
    KOKKOS_INLINE_FUNCTION
    Matrix( const Matrix& rhs )
    {
        KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose>::invoke(
            rhs, *this );
    }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_0() const
    { return N; }

    KOKKOS_INLINE_FUNCTION
    int stride_1() const
    { return 1; }

    // Extent
    KOKKOS_INLINE_FUNCTION
    int extent( const int d ) const
    { return _extent[d]; }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const
    { return const_cast<pointer>(_d); }
};

//---------------------------------------------------------------------------//
// Vector
//---------------------------------------------------------------------------//
// Dense vector with a KokkosKernels compatible data interface.
template<class T, int N, class TransposeType = NoTranspose>
struct Vector;

// No transpose
template<class T, int N>
struct Vector<T,N,NoTranspose>
{
    T _d[N];
    int _extent[2] = {N,1};

    using value_type = T;
    using non_const_value_type = typename std::remove_const<T>::type;
    using pointer = T*;
    using reference = T&;
    using const_reference = typename std::add_const<T>::type&;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    Vector() = default;

    // Initializer list constructor.
    KOKKOS_INLINE_FUNCTION
    Vector( const std::initializer_list<T> data )
    {
        int i = 0;
        for ( const auto& value : data )
        {
            _d[i] = value;
            ++i;
        }
    }

    // Deep copy constructor.
    KOKKOS_INLINE_FUNCTION
    Vector( const Vector& rhs )
    {
        KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose>::invoke(
            rhs, *this );
    }

    // Scalar constructor.
    KOKKOS_INLINE_FUNCTION
    Vector( const T value )
    {
        KokkosBatched::SerialSet::invoke( value, *this );
    }

    // Deep copy assignment operator.
    KOKKOS_INLINE_FUNCTION
    Vector& operator=( const Vector& rhs )
    {
        KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose>::invoke(
            rhs, *this );
        return *this;
    }

    // Scalar value assignment.
    KOKKOS_INLINE_FUNCTION
    Vector& operator=( const T value )
    {
        KokkosBatched::SerialSet::invoke( value, *this );
        return *this;
    }

    // Transpose operator.
    KOKKOS_INLINE_FUNCTION
    Vector<T,N,Transpose> operator~()
    {
        return Vector<T,N,Transpose>( this->data() );
    }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_0() const
    { return 1; }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_1() const
    { return 0; }

    // Extent
    KOKKOS_INLINE_FUNCTION
    int extent( const int d ) const
    { return _extent[d]; }

    // Access an individual element.
    KOKKOS_INLINE_FUNCTION
    const_reference operator()( const int i ) const
    { return _d[i]; }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i )
    { return _d[i]; }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const
    { return const_cast<pointer>(&_d[0]); }
};

// Transpose. This class is essentially a shallow-copy placeholder to enable
// transpose vector operations without copies.
template<class T, int N>
struct Vector<T,N,Transpose>
{
    T* _d;
    int _extent[2] = {N,1};

    using value_type = T;
    using non_const_value_type = typename std::remove_const<T>::type;
    using pointer = T*;
    using reference = T&;
    using const_reference = typename std::add_const<T>::type&;

    // Pointer constructor.
    KOKKOS_INLINE_FUNCTION
    Vector( T* data )
        : _d(data)
    {}

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_0() const
    { return 1; }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_1() const
    { return 0; }

    // Extent
    KOKKOS_INLINE_FUNCTION
    int extent( const int d ) const
    { return _extent[d]; }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const
    { return const_cast<pointer>(&_d[0]); }
};

//---------------------------------------------------------------------------//
// Matrix-matrix multiplication
//---------------------------------------------------------------------------//
// NoTranspose case.
template<class T, int M, int N, int K>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose>
operator*( const Matrix<T,M,K,NoTranspose>& a, const Matrix<T,K,N,NoTranspose>& b )
{
    Matrix<T,M,N,NoTranspose> c;
    KokkosBatched::SerialGemm<NoTranspose::type,
                              NoTranspose::type,
                              KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  a, b, Kokkos::ArithTraits<T>::one(), c );
    return c;
}

//---------------------------------------------------------------------------//
// Transpose case.
template<class T, int M, int N, int K>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose>
operator*( const Matrix<T,K,M,Transpose>& a, const Matrix<T,N,K,Transpose>& b )
{
    Matrix<T,M,N,NoTranspose> c;
    KokkosBatched::SerialGemm<Transpose::type,
                              Transpose::type,
                              KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  a, b, Kokkos::ArithTraits<T>::one(), c );
    return c;
}

//---------------------------------------------------------------------------//
// NoTranspose-Transpose case.
template<class T, int M, int N, int K>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose>
operator*( const Matrix<T,M,K,NoTranspose>& a, const Matrix<T,N,K,Transpose>& b )
{
    Matrix<T,M,N,NoTranspose> c;
    KokkosBatched::SerialGemm<NoTranspose::type,
                              Transpose::type,
                              KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  a, b, Kokkos::ArithTraits<T>::one(), c );
    return c;
}

//---------------------------------------------------------------------------//
// Transpose-NoTranspose case.
template<class T, int M, int N, int K>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose>
operator*( const Matrix<T,K,M,Transpose>& a, const Matrix<T,K,N,NoTranspose>& b )
{
    Matrix<T,M,N,NoTranspose> c;
    KokkosBatched::SerialGemm<Transpose::type,
                              NoTranspose::type,
                              KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  a, b, Kokkos::ArithTraits<T>::one(), c );
    return c;
}

//---------------------------------------------------------------------------//
// Matrix-vector multiplication
//---------------------------------------------------------------------------//
// NoTranspose case.
template<class T, int M, int N>
KOKKOS_INLINE_FUNCTION
Vector<T,M,NoTranspose>
operator*( const Matrix<T,M,N,NoTranspose>& a, const Vector<T,N,NoTranspose>& x )
{
    Vector<T,M,NoTranspose> y;
    KokkosBatched::SerialGemv<NoTranspose::type,
                              KokkosBatched::Algo::Gemv::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  a, x, Kokkos::ArithTraits<T>::one(), y );
    return y;
}

//---------------------------------------------------------------------------//
// Transpose case.
template<class T, int M, int N>
KOKKOS_INLINE_FUNCTION
Vector<T,M,NoTranspose>
operator*( const Matrix<T,N,M,Transpose>& a, const Vector<T,N,NoTranspose>& x )
{
    Vector<T,M,NoTranspose> y;
    KokkosBatched::SerialGemv<Transpose::type,
                              KokkosBatched::Algo::Gemv::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  a, x, Kokkos::ArithTraits<T>::one(), y );
    return y;
}

//---------------------------------------------------------------------------//
// Vector-matrix multiplication
//---------------------------------------------------------------------------//
// NoTranspose case.
template<class T, int M, int N>
KOKKOS_INLINE_FUNCTION
Matrix<T,1,N,NoTranspose>
operator*( const Vector<T,M,Transpose>& x, const Matrix<T,M,N,NoTranspose>& a )
{
    Matrix<T,1,N,NoTranspose> c;
    KokkosBatched::SerialGemm<Transpose::type,
                              NoTranspose::type,
                              KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  x, a, Kokkos::ArithTraits<T>::one(), c );
    return c;
}

//---------------------------------------------------------------------------//
// Transpose case.
template<class T, int M, int N>
KOKKOS_INLINE_FUNCTION
Matrix<T,1,M,NoTranspose>
operator*( const Vector<T,N,Transpose>& x, const Matrix<T,M,N,Transpose>& a )
{
    Matrix<T,1,M,NoTranspose> c;
    KokkosBatched::SerialGemm<Transpose::type,
                              Transpose::type,
                              KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  x, a, Kokkos::ArithTraits<T>::one(), c );
    return c;
}

//---------------------------------------------------------------------------//
// Vector products.
//---------------------------------------------------------------------------//
// Dot product.
template<class T, int N>
KOKKOS_INLINE_FUNCTION
T operator*( const Vector<T,N,Transpose>& x, const Vector<T,N,NoTranspose>& y )
{
    auto v = Kokkos::ArithTraits<T>::zero();
    KokkosBatched::InnerMultipleDotProduct<1> dp( 0, 1, 1, 1 );
    dp.serial_invoke(  Kokkos::ArithTraits<T>::one(),
                       x.data(), y.data(), N, &v );
    return v;
}

//---------------------------------------------------------------------------//
// Inner product.
template<class T, int N>
KOKKOS_INLINE_FUNCTION
Matrix<T,N,N,NoTranspose>
operator*( const Vector<T,N,NoTranspose>& x, const Vector<T,N,Transpose>& y )
{
    Matrix<T,N,N,NoTranspose> c;
    KokkosBatched::SerialGemm<NoTranspose::type,
                              Transpose::type,
                              KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  x, y, Kokkos::ArithTraits<T>::one(), c );
    return c;
}

//---------------------------------------------------------------------------//
// Cross product
template<class T>
KOKKOS_INLINE_FUNCTION
Vector<T,3,NoTranspose>
operator%( const Vector<T,3,NoTranspose>& x, const Vector<T,3,NoTranspose>& y )
{
    Vector<T,3,NoTranspose> z = { x(1)*y(2) - x(2)*y(1),
                                  x(2)*y(0) - x(0)*y(2),
                                  x(0)*y(1) - x(1)*y(0) };
    return z;
}

//---------------------------------------------------------------------------//
// Scalar multiplication.
//---------------------------------------------------------------------------//
// Matrix. No Transpose.
template<class T, int M, int N>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose>
operator*( const T v, const Matrix<T,M,N,NoTranspose>& a )
{
    Matrix<T,M,N,NoTranspose> b = a;
    KokkosBatched::SerialScale::invoke( v, b );
    return b;
}

//---------------------------------------------------------------------------//
// Vector. No transpose.
template<class T, int N>
KOKKOS_INLINE_FUNCTION
Vector<T,N,NoTranspose>
operator*( const T v, const Vector<T,N,NoTranspose>& x )
{
    Vector<T,N,NoTranspose> y = x;
    KokkosBatched::SerialScale::invoke( v, y );
    return y;
}

//---------------------------------------------------------------------------//
// Linear solve.
//---------------------------------------------------------------------------//
// General case.
template<class T, int N, class Trans>
KOKKOS_INLINE_FUNCTION
Vector<T,N,NoTranspose>
operator^( const Matrix<T,N,N,Trans>& a, const Vector<T,N,NoTranspose>& b )
{
    const Matrix<T,N,N,NoTranspose> a_lu = a;
    KokkosBatched::SerialLU<KokkosBatched::Algo::LU::Unblocked>::invoke( a_lu );
    auto x = b;
    KokkosBatched::SerialSolveLU<
        NoTranspose::type,
        KokkosBatched::Algo::SolveLU::Unblocked>::invoke( a_lu, x );
    return x;
}

//---------------------------------------------------------------------------//
// 2x2 specialization. No transpose
template<class T>
KOKKOS_INLINE_FUNCTION
Vector<T,2,NoTranspose>
operator^( const Matrix<T,2,2,NoTranspose>& a, const Vector<T,2,NoTranspose>& b )
{
    auto a_det_inv = 1.0 / ( a(0,0) * a(1,1) - a(0,1) * a(1,0) );

    Matrix<T,2,2,NoTranspose> a_inv;

    a_inv(0,0) = a(1,1) * a_det_inv;
    a_inv(0,1) = -a(0,1) * a_det_inv;
    a_inv(1,0) = -a(1,0) * a_det_inv;
    a_inv(1,1) = a(0,0) * a_det_inv;

    return a_inv * b;
}

//---------------------------------------------------------------------------//
// 2x2 specialization. No transpose
template<class T>
KOKKOS_INLINE_FUNCTION
Vector<T,2,NoTranspose>
operator^( const Matrix<T,2,2,Transpose>& a, const Vector<T,2,NoTranspose>& b )
{
    Matrix<T,2,2,NoTranspose> a_cp = a;

    auto a_det_inv = 1.0 / ( a_cp(0,0) * a_cp(1,1) - a_cp(0,1) * a_cp(1,0) );

    Matrix<T,2,2,NoTranspose> a_inv;

    a_inv(0,0) = a_cp(1,1) * a_det_inv;
    a_inv(0,1) = -a_cp(0,1) * a_det_inv;
    a_inv(1,0) = -a_cp(1,0) * a_det_inv;
    a_inv(1,1) = a_cp(0,0) * a_det_inv;

    return a_inv * b;
}

//---------------------------------------------------------------------------//
// 3x3 specialization. No transpose
template<class T>
KOKKOS_INLINE_FUNCTION
Vector<T,3,NoTranspose>
operator^( const Matrix<T,3,3,NoTranspose>& a, const Vector<T,3,NoTranspose>& b )
{
    auto a_det_inv = 1.0 / (a(0,0) * a(1,1) * a(2,2) +
                            a(0,1) * a(1,2) * a(2,0) +
                            a(0,2) * a(1,0) * a(2,1) -
                            a(0,2) * a(1,1) * a(2,0) -
                            a(0,1) * a(1,0) * a(2,2) -
                            a(0,0) * a(1,2) * a(2,1) );

    Matrix<T,3,3,NoTranspose> a_inv;

    a_inv(0,0) = (a(1,1)*a(2,2) - a(1,2)*a(2,1)) * a_det_inv;
    a_inv(0,1) = (a(0,2)*a(2,1) - a(0,1)*a(2,2)) * a_det_inv;
    a_inv(0,2) = (a(0,1)*a(1,2) - a(0,2)*a(1,1)) * a_det_inv;

    a_inv(1,0) = (a(1,2)*a(2,0) - a(1,0)*a(2,2)) * a_det_inv;
    a_inv(1,1) = (a(0,0)*a(2,2) - a(0,2)*a(2,0)) * a_det_inv;
    a_inv(1,2) = (a(0,2)*a(1,0) - a(0,0)*a(1,2)) * a_det_inv;

    a_inv(2,0) = (a(1,0)*a(2,1) - a(1,1)*a(2,0)) * a_det_inv;
    a_inv(2,1) = (a(0,1)*a(2,0) - a(0,0)*a(2,1)) * a_det_inv;
    a_inv(2,2) = (a(0,0)*a(1,1) - a(0,1)*a(1,0)) * a_det_inv;

    return a_inv * b;
}

//---------------------------------------------------------------------------//
// 3x3 specialization. Transpose
template<class T>
KOKKOS_INLINE_FUNCTION
Vector<T,3,NoTranspose>
operator^( const Matrix<T,3,3,Transpose>& a, const Vector<T,3,NoTranspose>& b )
{
    Matrix<T,3,3,NoTranspose> a_cp = a;

    auto a_det_inv = 1.0 / (a_cp(0,0) * a_cp(1,1) * a_cp(2,2) +
                            a_cp(0,1) * a_cp(1,2) * a_cp(2,0) +
                            a_cp(0,2) * a_cp(1,0) * a_cp(2,1) -
                            a_cp(0,2) * a_cp(1,1) * a_cp(2,0) -
                            a_cp(0,1) * a_cp(1,0) * a_cp(2,2) -
                            a_cp(0,0) * a_cp(1,2) * a_cp(2,1) );

    Matrix<T,3,3,NoTranspose> a_inv;

    a_inv(0,0) = (a_cp(1,1)*a_cp(2,2) - a_cp(1,2)*a_cp(2,1)) * a_det_inv;
    a_inv(0,1) = (a_cp(0,2)*a_cp(2,1) - a_cp(0,1)*a_cp(2,2)) * a_det_inv;
    a_inv(0,2) = (a_cp(0,1)*a_cp(1,2) - a_cp(0,2)*a_cp(1,1)) * a_det_inv;

    a_inv(1,0) = (a_cp(1,2)*a_cp(2,0) - a_cp(1,0)*a_cp(2,2)) * a_det_inv;
    a_inv(1,1) = (a_cp(0,0)*a_cp(2,2) - a_cp(0,2)*a_cp(2,0)) * a_det_inv;
    a_inv(1,2) = (a_cp(0,2)*a_cp(1,0) - a_cp(0,0)*a_cp(1,2)) * a_det_inv;

    a_inv(2,0) = (a_cp(1,0)*a_cp(2,1) - a_cp(1,1)*a_cp(2,0)) * a_det_inv;
    a_inv(2,1) = (a_cp(0,1)*a_cp(2,0) - a_cp(0,0)*a_cp(2,1)) * a_det_inv;
    a_inv(2,2) = (a_cp(0,0)*a_cp(1,1) - a_cp(0,1)*a_cp(1,0)) * a_det_inv;

    return a_inv * b;
}

// //---------------------------------------------------------------------------//

} // end namespace LinearAlgebra
} // end namespace Picasso

#endif // end PICASSO_BATCHEDLINEARALGEBRA_HPP
