"""
Guards for the declared public surface (:doc:`public_api`).

``abcmb/__init__.py`` re-exports lazily via PEP 562 ``__getattr__`` so that
``import abcmb`` stays JAX-free. That costs three hand-maintained lists of the
same names -- the ``_EXPORTS`` dict, the ``__all__`` literal, and the
``TYPE_CHECKING`` import mirror -- none of which a checker can cross-verify,
so they are guarded here. The comments in ``__init__.py`` name this module.
"""

import pathlib
import warnings

import pytest

import abcmb


def test_lazy_exports_all_resolve():
    # Every name in _EXPORTS must actually be reachable as an attribute: the
    # dict maps name -> submodule by hand, so a typo or a moved symbol only
    # shows up here (or for a user at `from abcmb import X`).
    for name in abcmb._EXPORTS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = getattr(abcmb, name)
        assert obj is not None, f"{name} resolved to None"


def test_all_matches_exports():
    # __all__ is a literal (so checkers can read it) rather than computed from
    # _EXPORTS; this is the sync check the __init__ comment promises.
    dunders = {"__version__", "__author__"}
    assert set(abcmb.__all__) - dunders == set(abcmb._EXPORTS)
    for name in dunders:
        assert hasattr(abcmb, name)


def test_unknown_attribute_raises_attribute_error():
    # The __getattr__ fallback must behave like a normal module: a plain
    # AttributeError naming the module, not a KeyError from the _EXPORTS dict.
    with pytest.raises(AttributeError, match="no attribute 'not_a_real_symbol'"):
        abcmb.not_a_real_symbol


def test_type_checking_mirror_covers_exports():
    # The third list: `if TYPE_CHECKING: from .main import Model as Model`.
    # It is what gives pyright (and IDEs) real types for the lazy attributes,
    # but it never executes, so nothing at runtime can see it -- parse it.
    #
    # Dropping a name here is SILENT: measured with a scratch package, pyright
    # reports 0 errors and quietly degrades the symbol from `type[Model]` to
    # `Any`. Since abcmb ships py.typed, that lands on downstream users as
    # missing type information with no signal anywhere.
    import ast

    src = pathlib.Path(abcmb.__file__).read_text()
    mirrored = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.If):
            continue
        # `if TYPE_CHECKING:` -- the block never runs, so match on the name.
        test_src = ast.unparse(node.test)
        if "TYPE_CHECKING" not in test_src:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.ImportFrom):
                for alias in sub.names:
                    mirrored.add(alias.asname or alias.name)
                    # PEP 484: only `X as X` marks a deliberate re-export.
                    assert alias.asname == alias.name, (
                        f"{alias.name} in the TYPE_CHECKING mirror must use the "
                        "redundant `as` form to count as a re-export"
                    )

    missing = set(abcmb._EXPORTS) - mirrored
    assert not missing, (
        f"names in _EXPORTS but absent from the TYPE_CHECKING mirror: "
        f"{sorted(missing)} -- they will type as Any for anyone using the "
        "installed package"
    )
    extra = mirrored - set(abcmb._EXPORTS)
    assert not extra, (
        f"names mirrored for type checkers but not lazily exported: "
        f"{sorted(extra)} -- they will fail at runtime"
    )
