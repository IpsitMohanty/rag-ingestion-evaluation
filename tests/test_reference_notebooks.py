"""Coverage: reference material integrity -- the 6 unexecuted IBM course
notebooks and the 1 previously-executed document-loaders lab notebook are
kept as reference (per the brief: read, not ported or run). This just
confirms they're intact, valid notebooks, not that their content is
correct -- matching cnn-vit-land-classification's notebook-integrity
convention (test_repo_integrity.py).
"""
import nbformat
import pytest

from conftest import REFERENCE_DIR

COURSE_NOTEBOOKS = sorted((REFERENCE_DIR / "course-notebooks").glob("*.ipynb"))
LAB_NOTEBOOKS = sorted((REFERENCE_DIR / "document-loaders-lab").glob("*.ipynb"))
ALL_REFERENCE_NOTEBOOKS = COURSE_NOTEBOOKS + LAB_NOTEBOOKS


def test_six_course_notebooks_are_present():
    assert len(COURSE_NOTEBOOKS) == 6, f"Expected 6 course notebooks, found {len(COURSE_NOTEBOOKS)}"


def test_course_notebooks_have_never_been_executed():
    """These are reference material only (see README) -- zero outputs is
    the signal that nothing here was run or ported."""
    for path in COURSE_NOTEBOOKS:
        nb = nbformat.read(path, as_version=4)
        for cell in nb.get("cells", []):
            assert not cell.get("outputs"), f"{path.name} has outputs -- expected untouched reference material"
            assert cell.get("execution_count") is None


@pytest.mark.parametrize("notebook_path", ALL_REFERENCE_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_readable_and_well_formed(notebook_path):
    # Not nbformat.validate(): these IBM/Skills-Network course exports
    # declare nbformat_minor=4 but include a cell "id" field (a 4.5+
    # feature) and an empty "outputs" list on markdown cells (never valid
    # for any minor version) -- a real quirk in the vendor's own export
    # tooling, not something we modify on reference-only material. Read
    # + basic structural checks is the honest bar for "intact file".
    nb = nbformat.read(notebook_path, as_version=4)
    assert nb["cells"], f"{notebook_path.name} has no cells"
    assert all("cell_type" in cell for cell in nb["cells"])
