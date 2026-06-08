# Doc

Please reference docs/Scribblez.pdf to understand the overall goal of this project.

# C++ Code

## Inline methods

- **Multiline methods** cannot be defined inside of a class definition. They must be moved into a
  corresponding `.cpp` file, or or into a corresponding `.inl` file that is `#include`'d at the
  bottom of the header file.
- **Single-line methods** (single-expression bodies) may be defined inline
  directly in the header.
- Every `.inl` file `#include`s its own header at the top and is `#include`d
  at the bottom of that header.

## Style

- Functions should be short. They should obey the "one thing and one thing only" principle. After
  writing a function, you should review it to see if it can be shortened by pulling some part of it
  out into a helper that has a clear semantic meaning with a clear API boundary. If so, do it.
- Classes similarly should have a clear purpose - no "god" classes.
- If code starts to look similar in multiple spots, determine whether it can be refactored
  so that common components can be shared. If this can be done reasonably, be aggressive in
  achieving it.
- Don't define structs or lambdas within functions, unless there is a very good reason. This is
  almost always a violation of the above principles. Instead, define them outside the function.
- Aim for high modularity. You want small, self-contained components with well-defined API
  boundaries, and you want higher-level components to built on top of them.

## Clang-format

After editing any file, make sure to sanitize it with clang-format

## Backwards compatibility

This project is completely self-contained, and there are thus no backwards-compatibility
requirements. Never compromise on interface for the sake of backwards-compatibility.
