"""Gera versão simplificada das previsões: data, hora, grupo, confronto e placar previsto."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

src  = ROOT / "data" / "group_predictions.json"
dest = ROOT / "data" / "previsoes_fase_grupos.json"

data = json.loads(src.read_text(encoding="utf-8"))

simple = []
for p in data["predictions"]:
    simple.append({
        "data":           p["date"],
        "horario_brt":    p["time_brt"],
        "grupo":          p["group"],
        "mandante":       p["home"],
        "visitante":      p["away"],
        "placar_previsto": f"{p['predicted_score_home']}-{p['predicted_score_away']}",
    })

output = {
    "_gerado_em": data["_generated_at"],
    "_modelo": data["_model_version"],
    "jogos": simple,
}

dest.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Salvo em {dest}  ({len(simple)} jogos)")
