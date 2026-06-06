# Scribblez C++ Conventions

## Inline methods

- **Multiline methods** must be defined in the corresponding `.inl` file under
  `engine/inlines/scribblez/`, not in the class body in the header.
- **Single-line methods** (single-expression bodies) may be defined inline
  directly in the header.
- Every `.inl` file `#include`s its own header at the top and is `#include`d
  at the bottom of that header.
