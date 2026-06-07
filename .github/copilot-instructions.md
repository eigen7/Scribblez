# Doc

Please reference docs/Scribblez.pdf to understand the overall goal of this project.

# C++ Code

## Inline methods

- **Multiline methods** must be defined in the corresponding `.inl` file under
  `engine/inlines/scribblez/`, not in the class body in the header.
- **Single-line methods** (single-expression bodies) may be defined inline
  directly in the header.
- Every `.inl` file `#include`s its own header at the top and is `#include`d
  at the bottom of that header.

## Clang-format

After editing any file, make sure to sanitize it with clang-format

## Backwards compatibility

This project is completely self-contained, and there are thus no backwards-compatibility
requirements. Never compromise on interface for the sake of backwards-compatibility.
