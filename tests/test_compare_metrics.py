"""
Tests de scripts/compare_metrics.py — détection de régression entre deux
entraînements.

Enjeu : le ré-entraînement est automatique et sans garde-fou (« toujours
déployer »), donc une régression serait déployée silencieusement. Ce module est
le seul endroit où elle est détectée, et il est appelé par retrain.sh dans un
contexte où une erreur fait échouer le pipeline.

Le piège principal couvert ici : la clé NDCG n'est pas 'NDCG@10'. `eval_k` est
plafonné au nombre de catégories, soit 7 sur le jeu réel. Une clé en dur
renverrait None partout et ne détecterait jamais aucune régression.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_metrics.py"
sys.path.insert(0, str(SCRIPT.parent))

from compare_metrics import TOLERANCE, compare, extract_ndcg


def summary(ndcg: float | None, k: int | None = 7, **extra) -> dict:
    """Construit un results_summary.json minimal mais réaliste."""
    metrics: dict = {}
    if k is not None:
        metrics["eval_k"] = k
    if ndcg is not None and k is not None:
        metrics[f"NDCG@{k}"] = ndcg
    return {"metrics": metrics, "n_users": 34, "n_interactions": 51, **extra}


class TestExtractNdcg:
    def test_reads_ndcg_at_eval_k(self):
        """La clé suit eval_k, pas un @10 en dur."""
        assert extract_ndcg(summary(0.93, k=7)) == (0.93, 7)

    def test_missing_eval_k(self):
        assert extract_ndcg({"metrics": {}}) == (None, None)

    def test_missing_ndcg_value(self):
        assert extract_ndcg({"metrics": {"eval_k": 7}}) == (None, 7)

    def test_no_metrics_section(self):
        assert extract_ndcg({}) == (None, None)

    def test_non_numeric_ndcg(self):
        assert extract_ndcg({"metrics": {"eval_k": 7, "NDCG@7": "?"}}) == (None, 7)

    def test_hardcoded_at10_would_not_be_found(self):
        """
        Garde-fou explicite : avec 7 catégories, un résumé ne contient pas
        'NDCG@10'. C'est exactement le bug qui rendait la détection inerte.
        """
        s = summary(1.0, k=7)
        assert "NDCG@10" not in s["metrics"]
        assert extract_ndcg(s)[0] == 1.0


class TestCompare:
    def test_no_regression_returns_empty(self):
        assert compare(summary(1.0), summary(1.0)) == ""

    def test_improvement_returns_empty(self):
        assert compare(summary(0.8), summary(0.95)) == ""

    def test_degradation_is_reported(self):
        msg = compare(summary(1.0), summary(0.8))
        assert msg, "Une baisse de NDCG doit produire une alerte"
        assert "0.2000" in msg

    def test_degradation_message_contains_volume_context(self):
        """Sans le volume de données, on ne sait pas interpréter la baisse."""
        msg = compare(
            summary(1.0, n_users=34, n_interactions=51),
            summary(0.9, n_users=40, n_interactions=60),
        )
        assert "34" in msg and "40" in msg
        assert "51" in msg and "60" in msg

    def test_tiny_drop_below_tolerance_is_ignored(self):
        assert compare(summary(1.0), summary(1.0 - TOLERANCE / 10)) == ""

    def test_drop_above_tolerance_is_reported(self):
        assert compare(summary(1.0), summary(1.0 - TOLERANCE * 10))

    def test_eval_k_change_is_flagged_not_silent(self):
        """
        Changer eval_k change l'ensemble des paires évaluées : comparer des
        NDCG à k différents n'a pas de sens, mais ce n'est pas une régression.
        On le signale sans prétendre à une baisse.
        """
        msg = compare(summary(1.0, k=7), summary(0.5, k=6))
        assert "non comparable" in msg

    def test_first_run_without_previous_metrics(self):
        """Pas de métriques précédentes = premier run : on n'alerte pas."""
        assert compare({"n_users": 5}, summary(0.3)) == ""


class TestCli:
    def _write(self, tmp_path: Path, name: str, payload) -> Path:
        p = tmp_path / name
        p.write_text(
            payload if isinstance(payload, str) else json.dumps(payload),
            encoding="utf-8",
        )
        return p

    def _run(self, *args) -> subprocess.CompletedProcess:
        """
        Lance le script en UTF-8 explicite.

        Le sous-processus hérite de l'encodage de la console Windows (cp1252),
        et `text=True` décode ensuite sa sortie en UTF-8 : les accents de
        « dégradé » arrivent alors en « dÃ©gradÃ© ». Le test échouait selon que
        la variable PYTHONIOENCODING était héritée ou non — une loterie, pas un
        défaut du script.
        """
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            capture_output=True, text=True, check=False,
            encoding="utf-8", env=env,
        )

    def test_cli_emits_message_on_regression(self, tmp_path):
        old = self._write(tmp_path, "old.json", summary(1.0))
        new = self._write(tmp_path, "new.json", summary(0.5))
        r = self._run(str(old), str(new))
        assert r.returncode == 0, r.stderr
        assert "dégradé" in r.stdout

    def test_cli_silent_on_success(self, tmp_path):
        old = self._write(tmp_path, "old.json", summary(0.5))
        new = self._write(tmp_path, "new.json", summary(0.6))
        r = self._run(str(old), str(new))
        assert r.returncode == 0
        assert r.stdout.strip() == ""

    def test_cli_invalid_json_returns_error(self, tmp_path):
        bad = self._write(tmp_path, "bad.json", "{pas du json")
        new = self._write(tmp_path, "new.json", summary(0.5))
        r = self._run(str(bad), str(new))
        assert r.returncode == 1

    def test_cli_missing_file_returns_error(self, tmp_path):
        new = self._write(tmp_path, "new.json", summary(0.5))
        r = self._run(str(tmp_path / "absent.json"), str(new))
        assert r.returncode == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])