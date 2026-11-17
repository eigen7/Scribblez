"""Another tag's export that a workload reads, named by a pair of params,
`<name>_tag` and `<name>_generation`, the generation -1 for the source's
latest export. The workload pins the generation in its finalize hook, at task
creation, where the operator sees a missing export, and its roles read the
pinned file (a remote slot from the copy its RoleSpec.inputs stages).
"""

from scribblez import params as params_mod
from scribblez.paths import TagPaths


def export_generation(paths: TagPaths, generation: int, name: str, hint: str = "") -> int:
    """`generation` of the source at `paths`, or (-1) its latest export.
    Raises ParamsError if the tag is unset, the index invalid, or (-1) the tag
    has no export, `hint` appended to that message. A generation already
    pinned is taken as is: a remote worker's mount has none of the source's
    exports, only the staged input."""
    if not paths.tag:
        raise params_mod.ParamsError(f"{name}_tag is required")
    if generation < -1:
        raise params_mod.ParamsError(
            f"{name}_generation must be a generation index (>= 0) or -1 for the latest, "
            f"got {generation}"
        )
    if generation >= 0:
        return generation
    exported = paths.exported_generations()
    if not exported:
        raise params_mod.ParamsError(f"{paths.task} tag '{paths.tag}' has no export{hint}")
    return exported[-1]


def pinned_generation(paths: TagPaths, generation: int, name: str, hint: str = "") -> int:
    """export_generation, failing also if that generation has no export."""
    generation = export_generation(paths, generation, name, hint)
    if generation not in paths.exported_generations():
        raise params_mod.ParamsError(
            f"{paths.task} tag '{paths.tag}' generation {generation} has no export{hint}"
        )
    return generation
