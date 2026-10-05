"""
scripts/check_test_layout.py

Garde-fou contre la corruption silencieuse des fichiers de test.

pytest ne collected jamais :

1. un `def test_...` écrit À L'INTÉRIEUR d'un autre `def test_...` (une
   réindentation accidentelle le transforme en simple closure) ;
2. un `def test_...` placé dans une classe redéclarée plus bas dans le même
   fichier (la seconde définition écrase la première en Python) ;
3. un `def test_...` défini deux fois.

Dans les trois cas la suite passe au vert en ayant perdu des tests — c'est
exactement ce qui s'est produit : 5 tests imbriqués dans
`test_recommend_no_sold_articles` et une classe `TestMetrics` dupliquée
n'étaient jamais exécutés, sans la moindre alerte. pytest n'a aucun moyen de
le signaler : il ne voit que ce qu'il collecte.

Usage :
    python scripts/check_test_layout.py [tests/ ...]

Code de sortie 1 si une anomalie est détectée.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import Counter
from pathlib import Path


def _classes(tree: ast.Module) -> list[ast.ClassDef]:
    return [n for n in tree.body if isinstance(n, ast.ClassDef)]


def _tests(node: ast.AST, prefix: str = "") -> list[tuple[str, ast.FunctionDef]]:
    out: list[tuple[str, ast.FunctionDef]] = []
    for child in getattr(node, "body", []):
        if isinstance(child, ast.ClassDef):
            out.extend(_tests(child, f"{prefix}{child.name}."))
        elif isinstance(child, ast.FunctionDef) and child.name.startswith("test"):
            out.append((f"{prefix}{child.name}", child))
    return out


def _nested_tests(func: ast.FunctionDef) -> list[str]:
    """Tests définis dans un autre test : jamais collectés."""
    out: list[str] = []
    for node in ast.walk(func):
        if node is func or isinstance(node, ast.Lambda):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                out.append(node.name)
            out.extend(_nested_tests(node))  # searching deeper as well
    return out


def check(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    problems: list[str] = []

    # 1. classes redéclarées
    for name, count in Counter(c.name for c in _classes(tree)).items():
        if count > 1:
            problems.append(
                f"classe {name!r} déclarée {count} fois dans le même module : "
                f"la dernière définition écrase les tests de la précédente"
            )

    # 2 + 3. tests imbriqués et doublons
    names = Counter(qualname for qualname, _ in _tests(tree))
    for qualname, func in _tests(tree):
        line = getattr(func, "lineno", "?")
        for kid in _nested_tests(func):
            problems.append(
                f"{path}:{line} test {qualname!r} contient le test {kid!r} — "
                f"pytest ne le collecte jamais (vérifier l'indentation)"
            )
        if names[qualname] > 1:
            problems.append(
                f"{path}:{line} test {qualname!r} défini {names[qualname]} fois"
            )

    return problems


def main(argv: list[str] | None = None) -> int:
    # Même correctif que generate_fake_models.py : sur la console Windows
    # (cp1252) le caractère de l'emoji fait lever UnicodeEncodeError APRÈS
    # l'analyse — donc un garde-fou qui plante au pire moment.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets", nargs="*", type=Path, default=[Path("tests")],
        help="fichiers ou dossiers de tests à vérifier",
    )
    args = parser.parse_args(argv)

    files: list[Path] = []
    for target in args.targets:
        files.extend(sorted(target.rglob("test_*.py")) if target.is_dir() else [target])
    if not files:
        print("Aucun fichier de test trouvé", file=sys.stderr)
        return 1

    total = 0
    problems: list[str] = []
    for path in files:
        total += len(_tests(ast.parse(path.read_text(encoding="utf-8-sig"))))
        problems.extend(check(path))

    print(f"{len(files)} fichier(s), {total} tests détectés dans l'AST")
    if problems:
        print("\n::error title=Tests structurellement morts::"
              "Des tests ne seront jamais exécutés par pytest.")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("✅ Aucun test imbriqué, dupliqué ni classe écrasée")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())