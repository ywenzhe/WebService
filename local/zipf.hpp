#pragma once

#include <numeric>   // for std::iota if needed, but not directly used in the simple pdf generation
#include <random>    // for std::mt19937 and std::discrete_distribution
#include <vector>
#include <cmath>     // for std::pow
#include <limits>    // for std::numeric_limits

namespace benchmark { // 使用一个更通用的命名空间，避免与 far_memory 混淆

/**
 * zipf_table_distribution(N, s)
 * Zipf distribution for `N` items, in the range `[0, N-1]` inclusive.
 * The probabilities follow the power-law 1/(k+1)^s with exponent `s`.
 * This uses a table-lookup approach with std::discrete_distribution,
 * which provides values efficiently after an initialization phase.
 * Note: Index k is 0-based, so for a range [0, N-1], the k-th item's probability
 * is based on (k+1).
 */
template <class IntType = unsigned long, class RealType = double>
class zipf_table_distribution {
public:
  typedef IntType result_type;

  static_assert(std::numeric_limits<IntType>::is_integer, "");
  static_assert(!std::numeric_limits<RealType>::is_integer, "");

  /// Constructor: Precomputes probabilities and initializes std::discrete_distribution.
  /// n: The number of elements (results will be in range [0, n-1]).
  /// q: The exponent for the power-law 1/(k+1)^q.
  zipf_table_distribution(const IntType n, const RealType q_);

  // No explicit reset needed for std::discrete_distribution in simple cases.
  void reset() { /* std::discrete_distribution has its own reset, if needed. */ }

  /// Generates a random number following the Zipf distribution (0-indexed).
  IntType operator()(std::mt19937 &rng);

  /// Returns the exponent parameter.
  RealType s() const { return q_; }

  /// Returns the minimum value potentially generated (0-indexed).
  result_type min() const { return 0; }

  /// Returns the maximum value potentially generated (0-indexed).
  result_type max() const { return n_ - 1; }

private:
  std::vector<RealType> probabilities_;       ///< Probabilities for each item
  IntType n_;                                 ///< Number of elements
  RealType q_;                                ///< Exponent
  std::discrete_distribution<IntType> dist_;  ///< Draw generator (internal)

  /// Helper to initialize the probability vector.
  void init(const IntType n, const RealType q);
};

// --- Implementation (equivalent to original .ipp file) ---
// Note: In modern C++, template implementations can often reside in the header file directly
// when they are small and non-polymorphic, or for simplicity. If performance is critical
// or compile times become an issue with large numbers of template instantiations,
// a separate .cpp/.ipp approach might be reconsidered. For this case, it's fine.

template <class IntType, class RealType>
zipf_table_distribution<IntType, RealType>::zipf_table_distribution(
    const IntType n, const RealType q)
    : n_(n), q_(q), dist_() { // Delay dist_ initialization
  init(n, q); // Call init to fill probabilities_
  dist_ = std::discrete_distribution<IntType>(probabilities_.begin(), probabilities_.end());
}

template <class IntType, class RealType>
IntType zipf_table_distribution<IntType, RealType>::operator()(std::mt19937 &rng) {
  return dist_(rng); // std::discrete_distribution naturally returns 0-indexed results
}

template <class IntType, class RealType>
void zipf_table_distribution<IntType, RealType>::init(const IntType n, const RealType q) {
  probabilities_.reserve(static_cast<std::size_t>(n)); // Cast n to size_t for reserve
  // For Zipf distribution, probability of item i is 1/i^q.
  // Since we want 0-indexed results, item at index 'k' corresponds to (k+1)-th item in Zipf.
  for (IntType i = 0; i < n; ++i) {
    // Probability for 0-th item means original 1st item (1/1^q), 1st item means original 2nd item (1/2^q), etc.
    probabilities_.push_back(std::pow(static_cast<RealType>(i + 1), -q));
  }
  // std::discrete_distribution will automatically normalize these weights so they sum to 1.
}


} // namespace benchmark

