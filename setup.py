"""
Setup script — downloads base data required by the predictor.

Usage:
    python setup.py

Downloads:
  1. World Football Elo Ratings (eloratings.net)
  2. Kaggle international football results CSV
     (place data/results.csv manually OR provide KAGGLE_USERNAME + KAGGLE_KEY env vars)
  3. Trains the XGBoost model if the results CSV is present
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import json
import requests


def _banner(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def step_directories() -> None:
    _banner("1/4 — Criando estrutura de diretórios")
    for d in ["data", "data/cache", "src", "tests"]:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}/")


def step_elo_ratings() -> None:
    _banner("2/4 — Baixando ratings Elo (eloratings.net)")
    from config import ELO_CSV_PATH, ELO_DATA_URL, SCRAPING_HEADERS

    out = Path(ELO_CSV_PATH)
    try:
        print(f"  → GET {ELO_DATA_URL}")
        resp = requests.get(ELO_DATA_URL, headers=SCRAPING_HEADERS, timeout=20)
        resp.raise_for_status()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(resp.text, encoding="utf-8")
        lines = resp.text.strip().split("\n")
        print(f"  ✓ {len(lines)} linhas salvas em {out}")
    except Exception as exc:
        print(f"  ⚠️  Falha no download: {exc}")
        print("     Os ratings de fallback hardcoded serão usados.")


def step_kaggle_data() -> None:
    _banner("3/4 — Dataset Kaggle (resultados internacionais)")

    out = Path("data/results.csv")
    if out.exists():
        print(f"  ✓ {out} já existe ({out.stat().st_size // 1024} KB) — pulando download.")
        return

    # Try kaggle API
    username = os.getenv("KAGGLE_USERNAME")
    key      = os.getenv("KAGGLE_KEY")

    if username and key:
        try:
            import subprocess
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d",
                 "martj42/international-football-results-from-1872-to-2017",
                 "--path", "data", "--unzip", "--force"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                print("  ✓ Dataset Kaggle baixado via CLI.")
                return
            else:
                print(f"  ⚠️  kaggle CLI falhou: {result.stderr.strip()}")
        except FileNotFoundError:
            print("  ⚠️  kaggle CLI não encontrado.")

    print("""
  [AÇÃO NECESSÁRIA]
  O dataset de resultados históricos não foi encontrado.

  Para habilitá-lo:
    1. Faça login em https://www.kaggle.com/
    2. Baixe o arquivo:
       https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
    3. Salve como  data/results.csv  na pasta do projeto.

  Sem o dataset, o modelo XGBoost será desativado e a previsão
  usará somente Elo + Poisson (ainda funcional).
""")


def step_train_model() -> None:
    _banner("4/4 — Treinando modelo XGBoost")

    results_path = Path("data/results.csv")
    if not results_path.exists():
        print("  ⚠️  data/results.csv não encontrado — treinamento ignorado.")
        print("     Siga as instruções acima para habilitar o XGBoost.")
        return

    try:
        from src.ml_model import get_model
        model = get_model()
        if model.is_trained:
            print("  ✓ Modelo XGBoost treinado e salvo em data/xgb_model.pkl")
        else:
            print("  ⚠️  Treinamento falhou. Verifique se xgboost está instalado:")
            print("     pip install xgboost shap")
    except Exception as exc:
        print(f"  ⚠️  Erro no treinamento: {exc}")


def step_verify() -> None:
    _banner("Verificação final")
    checks = {
        "config.py":               Path("config.py").exists(),
        "app.py":                  Path("app.py").exists(),
        "data/league_strength.json": Path("data/league_strength.json").exists(),
        "data/demo_squads.json":   Path("data/demo_squads.json").exists(),
        "src/ensemble.py":         Path("src/ensemble.py").exists(),
        "data/results.csv":        Path("data/results.csv").exists(),
        "data/xgb_model.pkl":      Path("data/xgb_model.pkl").exists(),
    }
    all_ok = True
    for name, exists in checks.items():
        icon = "✓" if exists else "✗"
        if not exists and name not in ("data/results.csv", "data/xgb_model.pkl"):
            all_ok = False
        print(f"  {icon} {name}")

    print()
    if all_ok:
        print("✅ Setup completo! Execute:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Alguns arquivos obrigatórios estão faltando. Verifique os erros acima.")


if __name__ == "__main__":
    print("⚽ Copa do Mundo 2026 — Previsor de Resultados")
    print("   Setup script\n")
    step_directories()
    step_elo_ratings()
    step_kaggle_data()
    step_train_model()
    step_verify()
