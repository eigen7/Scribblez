# Doc

Please reference docs/Scribblez.pdf to understand the overall goal of this project.

# Comments and documentation

Write every comment and doc as a standalone description of the code as it currently is, for a reader
with no prior context. Do not reference past versions of the code or the change that produced it
("we replaced", "previously", "the old X", "now uses", "formerly"), and do not reference anything
that only makes sense from the current conversation or task. State what the code does and why, not
what it used to do or how it got here — that history belongs in commit messages, not the code.

# C++ Code

## Building

Build by running build.py

## Macondo repo

It is checked out at /workspace/mount/macondo/

If you are asked questions regarding Macondo, please look there.

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
  boundaries, which can be reasoned about and optimized in isolation. Then, you want higher-level
  components built on top of them. If you ever spot code that would permit this sort of separation,
  suggest a change.

## Clang-format

After editing any file, make sure to sanitize it with clang-format

## Backwards compatibility

This project is completely self-contained, and there are thus no backwards-compatibility
requirements. Never compromise on interface for the sake of backwards-compatibility.
