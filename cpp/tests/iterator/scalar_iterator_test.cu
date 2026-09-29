/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <tests/iterator/iterator_tests.cuh>

#include <cudf_test/random.hpp>

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/optional>
#include <cuda/std/utility>
#include <thrust/host_vector.h>
#include <thrust/logical.h>

#include <cstdint>
#include <type_traits>

using TestingTypes = cudf::test::FixedWidthTypes;

TYPED_TEST_SUITE(IteratorTest, TestingTypes);

TYPED_TEST(IteratorTest, scalar_iterator)
{
  using T = TypeParam;
  T init  = cudf::test::make_type_param_scalar<T>(
    cudf::test::UniformRandomGenerator<int>(-128, 128).generate());
  // data and valid arrays
  thrust::host_vector<T> host_values(100, init);
  std::vector<bool> host_bools(100, true);

  // create a scalar
  using ScalarType = cudf::scalar_type_t<T>;
  std::unique_ptr<cudf::scalar> s(new ScalarType{init, true});

  // calculate the expected value by CPU.
  thrust::host_vector<cuda::std::pair<T, bool>> value_and_validity(host_values.size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 value_and_validity.begin(),
                 [](auto v, auto b) { return cuda::std::pair<T, bool>{v, b}; });
  thrust::host_vector<cuda::std::optional<T>> optional_values(host_values.size());
  std::transform(host_values.begin(), host_values.end(), optional_values.begin(), [](auto v) {
    return cuda::std::optional<T>{v};
  });

  // GPU test
  auto it_dev = cudf::detail::make_scalar_iterator<T>(*s);
  this->iterator_test_thrust(host_values, it_dev, host_values.size());

  auto it_pair_dev = cudf::detail::make_pair_iterator<T>(*s);
  this->iterator_test_thrust(value_and_validity, it_pair_dev, host_values.size());

  auto it_optional_dev = cudf::detail::make_optional_iterator<T>(*s, cudf::nullate::DYNAMIC{true});
  this->iterator_test_thrust(optional_values, it_optional_dev, host_values.size());
}

TYPED_TEST(IteratorTest, null_scalar_iterator)
{
  using T = TypeParam;
  T init  = cudf::test::make_type_param_scalar<T>(
    cudf::test::UniformRandomGenerator<int>(-128, 128).generate());
  // data and valid arrays
  std::vector<T> host_values(100, init);
  std::vector<bool> host_bools(100, true);

  // create a scalar
  using ScalarType = cudf::scalar_type_t<T>;
  std::unique_ptr<cudf::scalar> s(new ScalarType{init, true});

  // calculate the expected value by CPU.
  thrust::host_vector<cuda::std::pair<T, bool>> value_and_validity(host_values.size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 value_and_validity.begin(),
                 [](auto v, auto b) { return cuda::std::pair<T, bool>{v, b}; });

  // GPU test
  auto it_pair_dev = cudf::detail::make_pair_iterator<T>(*s);
  this->iterator_test_thrust(value_and_validity, it_pair_dev, host_values.size());
}

template <typename T>
struct FixedPointScalarIteratorTest : cudf::test::BaseFixture {
  void check_nonzero_scale(typename T::rep rep, numeric::scale_type scale)
  {
    auto const policy = rmm::exec_policy_nosync(cudf::get_default_stream());
    cudf::fixed_point_scalar<T> scalar{rep, scale};

    auto const value_it = cudf::detail::make_scalar_iterator<T>(scalar);
    EXPECT_TRUE(
      thrust::all_of(policy, value_it, value_it + 100, [rep, scale] __device__(T value) -> bool {
        return value.value() == rep && value.scale() == scale;
      }));

    auto const pair_it = cudf::detail::make_pair_iterator<T>(scalar);
    EXPECT_TRUE(thrust::all_of(policy,
                               pair_it,
                               pair_it + 100,
                               [rep, scale] __device__(cuda::std::pair<T, bool> value) -> bool {
                                 return value.second && value.first.value() == rep &&
                                        value.first.scale() == scale;
                               }));

    auto const optional_it =
      cudf::detail::make_optional_iterator<T>(scalar, cudf::nullate::DYNAMIC{true});
    EXPECT_TRUE(thrust::all_of(policy,
                               optional_it,
                               optional_it + 100,
                               [rep, scale] __device__(cuda::std::optional<T> value) -> bool {
                                 return value.has_value() && value->value() == rep &&
                                        value->scale() == scale;
                               }));
  }

  void check_null_nonzero_scale(numeric::scale_type scale)
  {
    auto const policy = rmm::exec_policy_nosync(cudf::get_default_stream());
    cudf::fixed_point_scalar<T> scalar{typename T::rep{12'345}, scale, false};
    EXPECT_THROW(cudf::detail::make_scalar_iterator<T>(scalar), cudf::logic_error);

    auto const pair_it = cudf::detail::make_pair_iterator<T>(scalar);
    EXPECT_TRUE(thrust::all_of(
      policy, pair_it, pair_it + 100, [] __device__(cuda::std::pair<T, bool> value) -> bool {
        return !value.second;
      }));

    auto const optional_it =
      cudf::detail::make_optional_iterator<T>(scalar, cudf::nullate::DYNAMIC{true});
    EXPECT_TRUE(thrust::all_of(
      policy, optional_it, optional_it + 100, [] __device__(cuda::std::optional<T> value) -> bool {
        return !value.has_value();
      }));
  }
};

TYPED_TEST_SUITE(FixedPointScalarIteratorTest, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointScalarIteratorTest, NonzeroScale)
{
  using T        = TypeParam;
  using rep_type = typename T::rep;
  for (auto const scale : {numeric::scale_type{-2}, numeric::scale_type{2}}) {
    for (auto const rep : {rep_type{-12'345}, rep_type{12'345}}) {
      SCOPED_TRACE(static_cast<int32_t>(scale));
      SCOPED_TRACE(static_cast<int64_t>(rep));
      this->check_nonzero_scale(rep, scale);
    }
  }
}

TYPED_TEST(FixedPointScalarIteratorTest, NullNonzeroScale)
{
  for (auto const scale : {numeric::scale_type{-2}, numeric::scale_type{2}}) {
    SCOPED_TRACE(static_cast<int32_t>(scale));
    this->check_null_nonzero_scale(scale);
  }
}

TYPED_TEST(FixedPointScalarIteratorTest, MismatchedType)
{
  using T     = TypeParam;
  using Other = std::
    conditional_t<std::is_same_v<T, numeric::decimal32>, numeric::decimal64, numeric::decimal32>;
  cudf::fixed_point_scalar<T> scalar{typename T::rep{12'345}, numeric::scale_type{-2}};
  cudf::numeric_scalar<int32_t> representation{12'345};

  EXPECT_THROW(cudf::detail::make_scalar_iterator<int32_t>(scalar), cudf::logic_error);
  EXPECT_THROW(cudf::detail::make_scalar_iterator<Other>(scalar), cudf::logic_error);
  EXPECT_THROW(cudf::detail::make_scalar_iterator<T>(representation), cudf::logic_error);
}
