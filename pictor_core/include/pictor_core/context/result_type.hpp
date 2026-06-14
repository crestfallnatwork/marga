#pragma once
#include <optional>
#include <variant>

namespace pictor {
template <typename T, typename E>
class Expected : protected std::variant<T, E> {
public:
  using std::variant<T, E>::variant;
  constexpr bool ok() const noexcept {
    return !std::holds_alternative<E>(*this);
  }
  constexpr E err() const noexcept { return std::get<E>(*this); }
  constexpr T &val() noexcept { return std::get<T>(*this); }
  constexpr T const &val() const noexcept { return std::get<T>(*this); }
  constexpr T *operator->() { return &this->val(); }
  constexpr T const *operator->() const { return &this->val(); }
  constexpr T &operator*() { return this->val(); }
  constexpr T const &operator*() const { return this->val(); }
};
template <typename T> class Optional : protected std::optional<T> {
public:
  using std::optional<T>::optional;
  constexpr bool ok() const noexcept { return this->has_value(); }
  constexpr T &val() noexcept { return this->value(); }
  constexpr T const &val() const noexcept { return this->value(); }
  constexpr T *operator->() { return &this->val(); }
  constexpr T const *operator->() const { return &this->val(); }
  constexpr T &operator*() { return this->val(); }
  constexpr T const &operator*() const { return this->val(); }
};
} // namespace pictor
