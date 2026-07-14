#include <cstddef>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laughableengine/linear_algebra.hpp"

namespace {

using laughableengine::DenseMatrix;
using laughableengine::GF;
using laughableengine::LinearSubspace;
using laughableengine::PrimeField;
using laughableengine::QQ;
using laughableengine::RationalField;
using laughableengine::horizontal_stack;
using laughableengine::matrix_from_columns;
using laughableengine::multiply;
using laughableengine::vertical_stack;

[[noreturn]] void fail(const std::string& message, int line) {
  throw std::runtime_error("line " + std::to_string(line) + ": " + message);
}

#define CHECK(expression)                                                     \
  do {                                                                        \
    if (!(expression)) {                                                      \
      fail("CHECK failed: " #expression, __LINE__);                          \
    }                                                                         \
  } while (false)

template <typename Exception, typename Function>
void expect_throw(Function&& function, int line) {
  try {
    std::invoke(std::forward<Function>(function));
  } catch (const Exception&) {
    return;
  } catch (const std::exception& error) {
    fail("unexpected exception type: " + std::string(error.what()), line);
  }
  fail("expected exception was not thrown", line);
}

#define EXPECT_THROW(exception, expression)                                   \
  expect_throw<exception>([&] { static_cast<void>(expression); }, __LINE__)

template <typename Field>
std::vector<typename Field::Element> integer_vector(
    const Field& field,
    std::initializer_list<long> values) {
  std::vector<typename Field::Element> result;
  result.reserve(values.size());
  for (const auto value : values) {
    result.push_back(field.from_integer(value));
  }
  return result;
}

template <typename Field>
DenseMatrix<Field> integer_matrix(
    const Field& field,
    std::size_t rows,
    std::size_t columns,
    std::initializer_list<long> entries) {
  if (entries.size() != rows * columns) {
    throw std::logic_error("invalid test matrix dimensions");
  }
  std::vector<typename Field::Element> converted;
  converted.reserve(entries.size());
  for (const auto value : entries) {
    converted.push_back(field.from_integer(value));
  }
  return DenseMatrix<Field>(field, rows, columns, std::move(converted));
}

template <typename Field>
bool is_zero_vector(
    const Field& field,
    const std::vector<typename Field::Element>& vector) {
  for (const auto& entry : vector) {
    if (!field.is_zero(entry)) {
      return false;
    }
  }
  return true;
}

void test_matrix_from_columns_and_shapes() {
  const auto field = QQ();
  const std::vector<std::vector<RationalField::Element>> columns{
      integer_vector(field, {1, 2}),
      integer_vector(field, {3, 4})};
  const auto matrix = matrix_from_columns(field, 2, columns);
  CHECK(matrix == integer_matrix(field, 2, 2, {1, 3, 2, 4}));

  const std::vector<std::vector<RationalField::Element>> no_columns;
  const auto empty = matrix_from_columns(field, 3, no_columns);
  CHECK(empty.row_count() == 3);
  CHECK(empty.column_count() == 0);

  const std::vector<std::vector<RationalField::Element>> wrong_length{
      integer_vector(field, {1, 2})};
  EXPECT_THROW(
      std::invalid_argument,
      matrix_from_columns(field, 3, wrong_length));

  const auto gf5 = GF(5);
  const auto gf7 = GF(7);
  const std::vector<std::vector<PrimeField::Element>> foreign_columns{
      {gf7.one()}};
  EXPECT_THROW(
      std::invalid_argument,
      matrix_from_columns(gf5, 1, foreign_columns));
}

void test_matrix_products() {
  const auto field = QQ();
  const auto left = integer_matrix(
      field, 2, 3, {1, 2, 3, 4, 5, 6});
  const auto right = integer_matrix(
      field, 3, 2, {7, 8, 9, 10, 11, 12});
  const auto product = multiply(left, right);
  CHECK(product == integer_matrix(field, 2, 2, {58, 64, 139, 154}));

  const auto third = integer_matrix(field, 2, 2, {1, 2, 0, 1});
  CHECK(multiply(product, third) ==
        multiply(left, multiply(right, third)));
  CHECK(multiply(left, right).transpose() ==
        multiply(right.transpose(), left.transpose()));

  const auto incompatible = integer_matrix(field, 4, 1, {1, 2, 3, 4});
  EXPECT_THROW(std::invalid_argument, multiply(left, incompatible));

  const DenseMatrix<RationalField> two_by_zero(field, 2, 0);
  const DenseMatrix<RationalField> zero_by_three(field, 0, 3);
  const auto zero_product = multiply(two_by_zero, zero_by_three);
  CHECK(zero_product.row_count() == 2);
  CHECK(zero_product.column_count() == 3);
  CHECK(zero_product.rank() == 0);

  const auto gf5 = GF(5);
  const auto gf7 = GF(7);
  const DenseMatrix<PrimeField> over_five(gf5, 1, 1, {gf5.one()});
  const DenseMatrix<PrimeField> over_seven(gf7, 1, 1, {gf7.one()});
  EXPECT_THROW(std::invalid_argument, multiply(over_five, over_seven));
}

void test_stacks_and_stack_identities() {
  const auto field = QQ();
  const auto top = integer_matrix(field, 1, 2, {1, 2});
  const auto bottom = integer_matrix(field, 2, 2, {3, 4, 5, 6});
  const auto vertical = vertical_stack({top, bottom});
  CHECK(vertical == integer_matrix(field, 3, 2, {1, 2, 3, 4, 5, 6}));

  const auto left = integer_matrix(field, 2, 1, {1, 2});
  const auto right = integer_matrix(field, 2, 2, {3, 4, 5, 6});
  const auto horizontal = horizontal_stack({left, right});
  CHECK(horizontal == integer_matrix(field, 2, 3, {1, 3, 4, 2, 5, 6}));

  const std::vector<DenseMatrix<RationalField>> no_blocks;
  const auto empty_vertical = vertical_stack(field, 4, no_blocks);
  CHECK(empty_vertical.row_count() == 0);
  CHECK(empty_vertical.column_count() == 4);
  const auto empty_horizontal = horizontal_stack(field, 3, no_blocks);
  CHECK(empty_horizontal.row_count() == 3);
  CHECK(empty_horizontal.column_count() == 0);
  EXPECT_THROW(std::invalid_argument, vertical_stack(no_blocks));
  EXPECT_THROW(std::invalid_argument, horizontal_stack(no_blocks));

  const auto post = integer_matrix(field, 2, 2, {1, 1, 0, 1});
  CHECK(multiply(vertical, post) ==
        vertical_stack({multiply(top, post), multiply(bottom, post)}));

  const auto pre = integer_matrix(field, 1, 2, {2, -1});
  CHECK(multiply(pre, horizontal) ==
        horizontal_stack({multiply(pre, left), multiply(pre, right)}));

  const auto wrong_columns = integer_matrix(field, 1, 3, {1, 2, 3});
  EXPECT_THROW(
      std::invalid_argument,
      vertical_stack(std::vector{top, wrong_columns}));
  const auto wrong_rows = integer_matrix(field, 3, 1, {1, 2, 3});
  EXPECT_THROW(
      std::invalid_argument,
      horizontal_stack(std::vector{left, wrong_rows}));

  const auto gf5 = GF(5);
  const auto gf7 = GF(7);
  const DenseMatrix<PrimeField> over_five(gf5, 1, 1, {gf5.one()});
  const DenseMatrix<PrimeField> over_seven(gf7, 1, 1, {gf7.one()});
  EXPECT_THROW(
      std::invalid_argument,
      vertical_stack(std::vector{over_five, over_seven}));
  EXPECT_THROW(
      std::invalid_argument,
      horizontal_stack(std::vector{over_five, over_seven}));
}

void test_subspace_span_coordinates_and_extremes() {
  const auto field = QQ();
  using Space = LinearSubspace<RationalField>;

  const auto e1 = integer_vector(field, {1, 0, 0});
  const auto e2 = integer_vector(field, {0, 1, 0});
  const auto e3 = integer_vector(field, {0, 0, 1});
  const auto zero_vector = integer_vector(field, {0, 0, 0});
  const auto plane = Space::span(
      field, 3,
      std::vector{
          e1, e2, integer_vector(field, {1, 2, 0}), zero_vector});
  const auto same_plane = Space::span(
      field, 3,
      std::vector{
          integer_vector(field, {0, 2, 0}),
          integer_vector(field, {-3, 0, 0})});
  CHECK(plane == same_plane);
  CHECK(plane.ambient_dimension() == 3);
  CHECK(plane.dimension() == 2);
  CHECK(plane.basis_matrix() ==
        integer_matrix(field, 3, 2, {1, 0, 0, 1, 0, 0}));

  const auto member = integer_vector(field, {3, 4, 0});
  CHECK(plane.contains(member));
  CHECK(!plane.contains(e3));
  CHECK(plane.coordinates(member) == integer_vector(field, {3, 4}));
  CHECK(plane.from_coordinates(integer_vector(field, {3, 4})) == member);
  EXPECT_THROW(std::domain_error, plane.coordinates(e3));
  const auto wrong_ambient = integer_vector(field, {1, 2});
  EXPECT_THROW(std::invalid_argument, plane.contains(wrong_ambient));
  EXPECT_THROW(
      std::invalid_argument,
      plane.from_coordinates(integer_vector(field, {1})));

  const auto independent = matrix_from_columns(field, 3, std::vector{e1, e2});
  CHECK(Space::from_independent_columns(independent) == plane);
  const auto dependent = matrix_from_columns(
      field, 3,
      std::vector{e1, integer_vector(field, {2, 0, 0})});
  EXPECT_THROW(
      std::invalid_argument,
      Space::from_independent_columns(dependent));

  const auto zero = Space::zero(field, 3);
  CHECK(zero.ambient_dimension() == 3);
  CHECK(zero.dimension() == 0);
  CHECK(zero.contains(zero_vector));
  CHECK(!zero.contains(e1));
  CHECK(zero.coordinates(zero_vector).empty());
  const std::vector<RationalField::Element> no_coordinates;
  CHECK(zero.from_coordinates(no_coordinates) == zero_vector);

  const auto whole = Space::whole(field, 3);
  CHECK(whole.dimension() == 3);
  const auto arbitrary = integer_vector(field, {3, -2, 5});
  CHECK(whole.contains(arbitrary));
  CHECK(whole.coordinates(arbitrary) == arbitrary);
  CHECK(whole.from_coordinates(arbitrary) == arbitrary);

  const auto zero_dimensional_zero = Space::zero(field, 0);
  const auto zero_dimensional_whole = Space::whole(field, 0);
  CHECK(zero_dimensional_zero == zero_dimensional_whole);
  CHECK(zero_dimensional_zero.ambient_dimension() == 0);
  CHECK(zero_dimensional_zero.dimension() == 0);
  const std::vector<RationalField::Element> empty_vector;
  CHECK(zero_dimensional_zero.contains(empty_vector));
  CHECK(zero_dimensional_zero.coordinates(empty_vector).empty());
}

void test_kernel_image_and_rank_nullity() {
  const auto field = QQ();
  using Space = LinearSubspace<RationalField>;

  const auto matrix = integer_matrix(
      field, 2, 3, {1, 2, 3, 2, 4, 6});
  const auto kernel = Space::kernel(matrix);
  const auto image = Space::image(matrix);
  CHECK(kernel.ambient_dimension() == 3);
  CHECK(kernel.dimension() == 2);
  CHECK(image.ambient_dimension() == 2);
  CHECK(image.dimension() == 1);
  CHECK(kernel.dimension() + matrix.rank() == matrix.column_count());
  for (const auto& vector : kernel.basis_vectors()) {
    CHECK(is_zero_vector(field, matrix.multiply_column(vector)));
  }
  for (std::size_t column = 0; column < matrix.column_count(); ++column) {
    const std::vector<RationalField::Element> basis_coordinate = [&] {
      std::vector<RationalField::Element> result(
          matrix.column_count(), field.zero());
      result[column] = field.one();
      return result;
    }();
    CHECK(image.contains(matrix.multiply_column(basis_coordinate)));
  }

  const auto annihilator = integer_matrix(field, 1, 2, {2, -1});
  CHECK(image == Space::kernel(annihilator));
  CHECK(multiply(annihilator, image.basis_matrix()).rank() == 0);

  const DenseMatrix<RationalField> no_equations(field, 0, 3);
  CHECK(Space::kernel(no_equations) == Space::whole(field, 3));
  const DenseMatrix<RationalField> no_columns(field, 3, 0);
  CHECK(Space::image(no_columns) == Space::zero(field, 3));
}

void test_intersections_and_dimension_identity() {
  const auto field = QQ();
  using Space = LinearSubspace<RationalField>;

  const auto e1 = integer_vector(field, {1, 0, 0});
  const auto e2 = integer_vector(field, {0, 1, 0});
  const auto e3 = integer_vector(field, {0, 0, 1});
  const auto first = Space::span(field, 3, std::vector{e1, e2});
  const auto second = Space::span(field, 3, std::vector{e2, e3});
  const auto common = Space::intersection(first, second);
  const auto expected = Space::span(field, 3, std::vector<Space::Vector>{e2});
  CHECK(common == expected);
  CHECK(Space::intersection(first, first) == first);
  CHECK(Space::intersection(
            Space::span(field, 3, std::vector<Space::Vector>{e1}),
            Space::span(field, 3, std::vector<Space::Vector>{e2})) ==
        Space::zero(field, 3));
  CHECK(Space::intersection(Space::whole(field, 3), first) == first);
  CHECK(Space::intersection(Space::zero(field, 3), first) ==
        Space::zero(field, 3));

  auto sum_generators = first.basis_vectors();
  const auto second_generators = second.basis_vectors();
  sum_generators.insert(
      sum_generators.end(), second_generators.begin(), second_generators.end());
  const auto sum = Space::span(field, 3, sum_generators);
  CHECK(sum.dimension() + common.dimension() ==
        first.dimension() + second.dimension());
  CHECK(sum == Space::whole(field, 3));

  const auto wrong_ambient = Space::whole(field, 2);
  EXPECT_THROW(
      std::invalid_argument,
      Space::intersection(first, wrong_ambient));
}

void test_prime_field_subspaces_and_foreign_tags() {
  const auto field = GF(5);
  using Space = LinearSubspace<PrimeField>;

  const auto generator = integer_vector(field, {1, 2, 3});
  const auto line = Space::span(
      field, 3,
      std::vector{
          generator,
          integer_vector(field, {2, 4, 6}),
          integer_vector(field, {4, 8, 12})});
  CHECK(line.dimension() == 1);
  CHECK(line.contains(integer_vector(field, {3, 6, 9})));
  CHECK(line.coordinates(integer_vector(field, {3, 6, 9})) ==
        integer_vector(field, {3}));

  std::size_t member_count = 0;
  for (long first = 0; first < 5; ++first) {
    for (long second = 0; second < 5; ++second) {
      for (long third = 0; third < 5; ++third) {
        if (line.contains(integer_vector(field, {first, second, third}))) {
          ++member_count;
        }
      }
    }
  }
  CHECK(member_count == 5);

  const auto left = integer_matrix(field, 2, 2, {1, 2, 3, 4});
  const auto right = integer_matrix(field, 2, 2, {4, 3, 2, 1});
  const auto third = integer_matrix(field, 2, 2, {2, 1, 1, 1});
  CHECK(multiply(multiply(left, right), third) ==
        multiply(left, multiply(right, third)));

  const auto foreign_field = GF(7);
  const auto foreign_space = Space::whole(foreign_field, 3);
  EXPECT_THROW(
      std::invalid_argument,
      Space::intersection(line, foreign_space));
  const auto foreign_vector = integer_vector(foreign_field, {1, 2, 3});
  EXPECT_THROW(std::invalid_argument, line.contains(foreign_vector));
  EXPECT_THROW(
      std::invalid_argument,
      line.from_coordinates(
          std::vector<PrimeField::Element>{foreign_field.one()}));
}

}  // namespace

int main() {
  try {
    test_matrix_from_columns_and_shapes();
    test_matrix_products();
    test_stacks_and_stack_identities();
    test_subspace_span_coordinates_and_extremes();
    test_kernel_image_and_rank_nullity();
    test_intersections_and_dimension_identity();
    test_prime_field_subspaces_and_foreign_tags();
    std::cout << "laughableengine: linear algebra composition tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "laughableengine linear algebra test failure: "
              << error.what() << '\n';
    return 1;
  }
}
