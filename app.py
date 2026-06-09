"""
Copa do Mundo 2026 — Previsor de Resultados
Telas:
  📅 Cronograma        – jogos por dia, placar sugerido, análise detalhada
  📊 Resultado         – probabilidades, heatmap, breakdown por modelo
  🧩 Análise do Elenco – radar, tabela por jogador
  🏆 Mata-Mata         – chaveamento (TBD até fase de grupos terminar)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import TOURNAMENT_PHASES
from src.ensemble import predict_match
from src.schedule_loader import (
    SQUAD_SOURCE_LABELS,
    get_all_groups,
    get_matches_by_date,
    get_knockout_bracket,
    get_r32_slots,
    get_recent_form,
    get_squad,
    get_squad_source,
    get_squad_elo,
)
from src.team_builder import build_team_score, radar_dimensions
from config import LEAGUE_STRENGTH

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Copa do Mundo 2026 — Previsor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .metric-card{background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px;
    padding:18px;text-align:center;border:1px solid #0f3460;color:white;}
  .metric-card .value{font-size:2.2rem;font-weight:700;}
  .metric-card .label{font-size:.85rem;opacity:.75;margin-top:4px;}
  .win-a{border-top:4px solid #4ade80;}
  .draw{border-top:4px solid #fbbf24;}
  .win-b{border-top:4px solid #f87171;}
  .warn{background:#3b1f02;border-left:4px solid #f59e0b;
    padding:8px 12px;border-radius:4px;margin:4px 0;}
  .info{background:#172554;border-left:4px solid #3b82f6;
    padding:8px 12px;border-radius:4px;margin:4px 0;}
  .match-card{background:linear-gradient(135deg,#0f172a,#1e293b);border-radius:10px;
    padding:14px 18px;margin:6px 0;border:1px solid #334155;}
  .score-badge{font-size:1.5rem;font-weight:800;color:#f8fafc;
    background:#0f3460;border-radius:8px;padding:4px 14px;display:inline-block;}
  .day-header{font-size:1.1rem;font-weight:700;color:#94a3b8;
    padding:8px 0 4px;border-bottom:1px solid #334155;margin-bottom:8px;}
  .venue-text{font-size:.8rem;color:#64748b;}
  .time-text{font-size:.8rem;color:#94a3b8;}
  .prob-bar{display:flex;gap:2px;width:100%;height:6px;border-radius:3px;overflow:hidden;margin-top:4px;}
  .tbd-card{background:#1e293b;border:2px dashed #475569;border-radius:10px;
    padding:16px;text-align:center;color:#64748b;}
</style>
""", unsafe_allow_html=True)

# ── Session-state defaults ────────────────────────────────────────────────────

_DEFAULTS = dict(
    page="📅 Cronograma",
    selected_match=None,
    squad_a=[], squad_b=[],
    team_res_a=None, team_res_b=None, result=None,
    custom_scores={},        # {match_key: (score_a, score_b)}
    editing_match=None,      # match_key being edited inline
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Month names (Portuguese) ──────────────────────────────────────────────────

_MONTHS_PT = {
    6: "junho", 7: "julho",
}

def _fmt_date(date_str: str) -> str:
    """'2026-06-11' → '11 de junho de 2026'"""
    try:
        y, m, d = date_str.split("-")
        return f"{int(d)} de {_MONTHS_PT.get(int(m), m)} de {y}"
    except Exception:
        return date_str

def _day_number(all_dates: list[str], date_str: str) -> int:
    try:
        return all_dates.index(date_str) + 1
    except ValueError:
        return 0

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    try:
        st.image(
            "https://upload.wikimedia.org/wikipedia/en/thumb/a/a3/2026_FIFA_World_Cup.svg/200px-2026_FIFA_World_Cup.svg.png",
            width=130,
        )
    except Exception:
        pass
    st.title("⚽ Copa 2026")
    st.markdown("---")
    _PAGES = [
        "📅 Cronograma",
        "📊 Resultado da Previsão",
        "🧩 Análise do Elenco",
        "🏆 Mata-Mata",
        "🥇 Probabilidade de Título",
        "ℹ️ Como Funciona",
    ]
    page = st.radio(
        "Navegação",
        _PAGES,
        index=_PAGES.index(st.session_state["page"]) if st.session_state["page"] in _PAGES else 0,
    )
    st.session_state["page"] = page

    if st.session_state["selected_match"]:
        m = st.session_state["selected_match"]
        st.markdown("---")
        st.caption("**Partida selecionada**")
        st.caption(f"{m['home']} × {m['away']}")
        st.caption(f"📅 {m['date']}")

    st.markdown("---")
    st.caption("Modelos: Elo · Poisson · XGBoost")
    st.caption("Dados: FBref · eloratings.net · Transfermarkt")

# ── Bulk prediction cache ─────────────────────────────────────────────────────

@st.cache_data(ttl=7200, show_spinner=False)
def _bulk_predictions() -> dict:
    """Pre-compute fast predictions (Elo + Poisson) for all 72 group matches."""
    from src.poisson_model import get_match_probabilities as poisson_probs
    from src.elo_model import win_draw_loss
    from src.data_fetcher import load_elo_ratings

    elo_ratings = load_elo_ratings()
    results: dict[str, dict] = {}

    for date_matches in get_matches_by_date().values():
        for m in date_matches:
            key = f"{m['home']}_{m['away']}"
            sq_a, _, _ = get_squad(m["home"])
            sq_b, _, _ = get_squad(m["away"])
            ra = build_team_score(sq_a[:11])
            rb = build_team_score(sq_b[:11])

            elo_a = float(elo_ratings.get(m["home"], get_squad_elo(m["home"])))
            elo_b = float(elo_ratings.get(m["away"], get_squad_elo(m["away"])))

            form_a = get_recent_form(m["home"])
            form_b = get_recent_form(m["away"])

            elo_p = win_draw_loss(elo_a, elo_b)
            pois  = poisson_probs(
                m["home"], m["away"],
                ra["team_score"], rb["team_score"],
                form_a, form_b,
                pos_a=ra.get("position_scores"),
                pos_b=rb.get("position_scores"),
            )

            # Blend Elo 55 % + Poisson 45 % for the schedule preview
            w_e, w_p = 0.55, 0.45
            wa   = w_e * elo_p["win_a"]  + w_p * pois["win_a"]
            draw = w_e * elo_p["draw"]   + w_p * pois["draw"]
            wb   = w_e * elo_p["win_b"]  + w_p * pois["win_b"]
            tot  = wa + draw + wb

            results[key] = {
                "score":  pois["most_probable_score"],
                "win_a":  round(wa   / tot, 3),
                "draw":   round(draw / tot, 3),
                "win_b":  round(wb   / tot, 3),
            }
    return results


def _load_and_predict(team_a: str, team_b: str, phase: str = "Fase de Grupos") -> None:
    """Load squads + run full ensemble prediction, store in session_state."""
    sq_a, _, _ = get_squad(team_a)
    sq_b, _, _ = get_squad(team_b)
    with st.spinner("Calculando previsão..."):
        ra = build_team_score(sq_a[:11])
        rb = build_team_score(sq_b[:11])
        result = predict_match(team_a, team_b, ra, rb, phase)
    st.session_state.update({
        "squad_a": sq_a[:11], "squad_b": sq_b[:11],
        "team_res_a": ra, "team_res_b": rb,
        "result": result,
    })


# ── Monte Carlo: championship probabilities ───────────────────────────────────

@st.cache_data(ttl=7200, show_spinner=False)
def _simulate_championship(n_sims: int = 8000) -> dict:
    """
    Simula o torneio completo N vezes usando ratings Elo.
    Retorna {team: {"winner", "final", "semi", "quarter"}} em percentual.
    """
    import random as _rnd

    rng        = _rnd.Random(42)
    elo_all    = {}
    try:
        from src.data_fetcher import load_elo_ratings
        elo_all = load_elo_ratings()
    except Exception:
        pass
    all_groups = get_all_groups()
    all_teams  = [t for v in all_groups.values() for t in v]

    def _elo(t):
        return float(elo_all.get(t, get_squad_elo(t)))

    def _p_win(ta, tb):
        return 1.0 / (1.0 + 10 ** (-(_elo(ta) - _elo(tb)) / 400))

    def _group_match(ta, tb):
        pw = _p_win(ta, tb)
        p_draw = max(0.05, 0.28 - 0.40 * abs(pw - 0.5))
        p_home = pw * (1 - p_draw)
        p_away = (1 - pw) * (1 - p_draw)
        r = rng.random()
        if r < p_home:
            ga = rng.randint(1, 4); gb = rng.randint(0, ga - 1)
        elif r < p_home + p_draw:
            g = rng.randint(0, 2); ga = gb = g
        else:
            gb = rng.randint(1, 4); ga = rng.randint(0, gb - 1)
        return ga, gb

    def _ko_match(ta, tb):
        return ta if rng.random() < _p_win(ta, tb) else tb

    stats = {t: {"winner": 0, "final": 0, "semi": 0, "quarter": 0} for t in all_teams}

    for _ in range(n_sims):
        group_standings: dict = {}
        thirds_data: list = []

        for gid, teams in all_groups.items():
            pts = {t: 0 for t in teams}
            gd  = {t: 0 for t in teams}
            gf  = {t: 0 for t in teams}
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    ta, tb = teams[i], teams[j]
                    ga, gb = _group_match(ta, tb)
                    gf[ta] += ga; gf[tb] += gb
                    gd[ta] += ga - gb; gd[tb] += gb - ga
                    if ga > gb:   pts[ta] += 3
                    elif ga == gb: pts[ta] += 1; pts[tb] += 1
                    else:          pts[tb] += 3
            ranked = sorted(teams,
                            key=lambda t: (pts[t], gd[t], gf[t], rng.random()),
                            reverse=True)
            group_standings[gid] = ranked
            thirds_data.append((pts[ranked[2]], gd[ranked[2]], gf[ranked[2]], ranked[2]))

        first  = [group_standings[g][0] for g in all_groups]
        second = [group_standings[g][1] for g in all_groups]
        thirds_sorted = sorted(thirds_data,
                               key=lambda x: (x[0], x[1], x[2], rng.random()),
                               reverse=True)
        best8 = [t for _, _, _, t in thirds_sorted[:8]]

        r32 = (first + second + best8)[:32]
        rng.shuffle(r32)

        def _rnd_round(bracket: list) -> list:
            return [_ko_match(bracket[k], bracket[k + 1])
                    for k in range(0, len(bracket), 2)]

        r16      = _rnd_round(r32)
        qf_teams = _rnd_round(r16)
        for t in qf_teams: stats[t]["quarter"] += 1
        sf_teams = _rnd_round(qf_teams)
        for t in sf_teams:  stats[t]["semi"]   += 1
        finalists = _rnd_round(sf_teams)
        for t in finalists: stats[t]["final"]   += 1
        champion  = _rnd_round(finalists)[0]
        stats[champion]["winner"] += 1

    return {
        t: {k: round(v / n_sims * 100, 2) for k, v in s.items()}
        for t, s in stats.items()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TELA 1 — Cronograma por dia de jogo
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📅 Cronograma":
    st.title("📅 Copa do Mundo 2026 — Cronograma")
    st.caption(
        "Placar sugerido calculado por **Elo + Poisson**. "
        "Clique em **Análise** para ver a previsão completa do jogo."
    )

    with st.spinner("Carregando previsões..."):
        preds = _bulk_predictions()

    matches_by_date = get_matches_by_date()
    all_dates = sorted(matches_by_date.keys())
    custom_scores: dict = st.session_state.get("custom_scores", {})
    editing_match: str | None = st.session_state.get("editing_match")

    # ── Compact filter / navigation ───────────────────────────────────────────
    filter_col1, filter_col2 = st.columns([2, 3])
    search_team = filter_col1.text_input("🔍 Filtrar por seleção", placeholder="ex: Brasil, Argentina...").strip().lower()
    # Date range selector
    unique_dates_label = [f"Dia {i+1} — {_fmt_date(d)}" for i, d in enumerate(all_dates)]
    unique_dates_label.insert(0, "Todos os dias")
    selected_day_label = filter_col2.selectbox("Ir para o dia:", unique_dates_label, index=0)
    selected_date_filter = None
    if selected_day_label != "Todos os dias":
        idx = unique_dates_label.index(selected_day_label) - 1
        selected_date_filter = all_dates[idx] if 0 <= idx < len(all_dates) else None

    st.markdown("---")

    # ── Match listing ─────────────────────────────────────────────────────────
    for date_str, day_matches in matches_by_date.items():
        # Filter by selected date
        if selected_date_filter and date_str != selected_date_filter:
            continue
        # Filter by team search
        if search_team:
            day_matches = [
                m for m in day_matches
                if search_team in m["home"].lower() or search_team in m["away"].lower()
            ]
            if not day_matches:
                continue

        day_num = _day_number(all_dates, date_str)
        st.markdown(
            f'<div class="day-header">⚽ Dia {day_num} &nbsp;·&nbsp; {_fmt_date(date_str)}</div>',
            unsafe_allow_html=True,
        )

        for m in day_matches:
            home, away = m["home"], m["away"]
            match_key = f"{home}_{away}"

            # Get score: custom override > predicted
            pred = preds.get(match_key, {"score": (1, 1), "win_a": 0.33, "draw": 0.34, "win_b": 0.33})
            display_score = custom_scores.get(match_key, pred["score"])
            win_a  = pred["win_a"]
            draw   = pred["draw"]
            win_b  = pred["win_b"]

            with st.container():
                # Main row: team A | score | team B
                c_home, c_vs, c_score, c_vs2, c_away, c_btn1, c_btn2 = st.columns(
                    [3, 0.3, 1.4, 0.3, 3, 1.5, 1.5]
                )
                c_home.markdown(f"**{home}**")
                c_vs.markdown("<div style='text-align:center;padding-top:6px;color:#64748b;'>vs</div>",
                              unsafe_allow_html=True)

                # Score display (custom or predicted)
                s0, s1 = (display_score[0], display_score[1]) if isinstance(display_score, (tuple, list)) else (1, 1)
                is_custom = match_key in custom_scores
                score_color = "#f59e0b" if is_custom else "#f8fafc"
                c_score.markdown(
                    f"<div style='text-align:center;'>"
                    f"<span style='font-size:1.6rem;font-weight:800;color:{score_color};"
                    f"background:#0f3460;border-radius:8px;padding:3px 12px;'>{s0}–{s1}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                c_vs2.markdown("<div style='text-align:center;padding-top:6px;'></div>",
                               unsafe_allow_html=True)
                c_away.markdown(f"**{away}**")

                if c_btn1.button("🔍 Análise", key=f"detail_{match_key}", use_container_width=True):
                    st.session_state.update({
                        "selected_match": {**m, "phase": "Fase de Grupos"},
                        "result": None, "team_res_a": None, "team_res_b": None,
                        "page": "📊 Resultado da Previsão",
                    })
                    _load_and_predict(home, away, "Fase de Grupos")
                    st.rerun()

                # Editar placar toggle
                edit_active = editing_match == match_key
                edit_label = "✓ Fechar" if edit_active else "✏️ Editar"
                if c_btn2.button(edit_label, key=f"edit_{match_key}", use_container_width=True):
                    st.session_state["editing_match"] = None if edit_active else match_key
                    st.rerun()

                # Venue + time + probability bar
                info_c1, info_c2 = st.columns([4, 2])
                venue_str = m.get("venue", "")
                group_str = f"Grupo {m.get('group','')}"
                time_str  = m.get("time_brt", "—")
                info_c1.markdown(
                    f'<span class="venue-text">📍 {venue_str} &nbsp;·&nbsp; {group_str}</span>',
                    unsafe_allow_html=True,
                )
                info_c2.markdown(
                    f'<span class="time-text">⏰ {time_str} BRT &nbsp;·&nbsp; '
                    f'<span style="color:#4ade80">{win_a*100:.0f}%</span> / '
                    f'<span style="color:#fbbf24">{draw*100:.0f}%</span> / '
                    f'<span style="color:#f87171">{win_b*100:.0f}%</span></span>',
                    unsafe_allow_html=True,
                )

                # Inline score editor
                if edit_active:
                    cur_a = int(s0)
                    cur_b = int(s1)
                    ec1, ec2, ec3, ec4, ec5 = st.columns([1, 0.5, 1, 1.5, 2])
                    ec1.markdown(f"<div style='text-align:right;padding-top:8px;font-size:.9rem'>{home}</div>",
                                 unsafe_allow_html=True)
                    new_a = ec2.number_input("", 0, 20, cur_a, key=f"ns_a_{match_key}",
                                             label_visibility="collapsed")
                    new_b = ec3.number_input("", 0, 20, cur_b, key=f"ns_b_{match_key}",
                                             label_visibility="collapsed")
                    ec4.markdown(f"<div style='padding-top:8px;font-size:.9rem'>{away}</div>",
                                 unsafe_allow_html=True)
                    if ec5.button("✓ Confirmar placar", key=f"confirm_{match_key}"):
                        st.session_state["custom_scores"][match_key] = (int(new_a), int(new_b))
                        st.session_state["editing_match"] = None
                        st.rerun()
                    if match_key in custom_scores:
                        if ec5.button("↺ Restaurar previsão", key=f"reset_{match_key}"):
                            del st.session_state["custom_scores"][match_key]
                            st.session_state["editing_match"] = None
                            st.rerun()

            st.markdown("<hr style='border-color:#1e293b;margin:6px 0;'>", unsafe_allow_html=True)

    if search_team and not any(
        search_team in m["home"].lower() or search_team in m["away"].lower()
        for mlist in matches_by_date.values() for m in mlist
    ):
        st.info(f"Nenhuma partida encontrada com '{search_team}'.")


# ═══════════════════════════════════════════════════════════════════════════════
# TELA 2 — Resultado da Previsão
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Resultado da Previsão":
    match = st.session_state.get("selected_match")
    if not match:
        st.info("👈 Selecione uma partida no **Cronograma** para ver a previsão.")
        st.stop()

    team_a = match["home"]
    team_b = match["away"]

    # ── Back button ────────────────────────────────────────────────────────────
    if st.button("← Voltar ao Cronograma"):
        st.session_state["page"] = "📅 Cronograma"
        st.rerun()

    st.title(f"📊 {team_a} × {team_b}")
    st.markdown(
        f"📅 `{match['date']}` &nbsp;·&nbsp; {match.get('venue','')}",
        unsafe_allow_html=True,
    )

    # ── Auto-compute if no result ──────────────────────────────────────────────
    if st.session_state.get("result") is None:
        _load_and_predict(team_a, team_b, match.get("phase", "Fase de Grupos"))
        st.rerun()

    result = st.session_state["result"]
    res_a  = st.session_state["team_res_a"]
    res_b  = st.session_state["team_res_b"]

    # ── Source badges ──────────────────────────────────────────────────────────
    _, src_a, ann_a = get_squad(team_a)
    _, src_b, ann_b = get_squad(team_b)
    ba = SQUAD_SOURCE_LABELS.get(src_a, src_a) + (f" ({ann_a})" if ann_a else "")
    bb = SQUAD_SOURCE_LABELS.get(src_b, src_b) + (f" ({ann_b})" if ann_b else "")
    inf1, inf2 = st.columns(2)
    inf1.markdown(f'<div class="info">Elenco {team_a}: {ba}</div>', unsafe_allow_html=True)
    inf2.markdown(f'<div class="info">Elenco {team_b}: {bb}</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Probability cards ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="metric-card win-a"><div class="value">{result["win_a"]*100:.1f}%</div>'
        f'<div class="label">Vitória {team_a}</div></div>', unsafe_allow_html=True)
    c2.markdown(
        f'<div class="metric-card draw"><div class="value">{result["draw"]*100:.1f}%</div>'
        f'<div class="label">Empate</div></div>', unsafe_allow_html=True)
    c3.markdown(
        f'<div class="metric-card win-b"><div class="value">{result["win_b"]*100:.1f}%</div>'
        f'<div class="label">Vitória {team_b}</div></div>', unsafe_allow_html=True)

    sc = result["most_probable_score"]
    exp = result.get("expected_score", sc)
    st.markdown(
        f'<br><div class="info">⚽ Placar mais provável: <b>{sc[0]}–{sc[1]}</b>'
        f'&nbsp;&nbsp;({result["most_probable_prob"]*100:.1f}% de probabilidade)'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;Placar esperado: <b>{exp[0]}–{exp[1]}</b></div>',
        unsafe_allow_html=True,
    )

    # ── Top-5 scorelines ──────────────────────────────────────────────────────
    top_scores = result.get("top_scorelines", [])
    if top_scores:
        st.markdown("**🎯 Top 5 placares mais prováveis**")
        cols_top = st.columns(len(top_scores))
        for col, entry in zip(cols_top, top_scores):
            border = "#f59e0b" if entry.get("most_likely") else "#334155"
            col.markdown(
                f"<div style='text-align:center;background:#0f172a;border:2px solid {border};"
                f"border-radius:8px;padding:8px 4px;'>"
                f"<div style='font-size:1.3rem;font-weight:700;color:#f8fafc;'>"
                f"{entry['goals_a']}–{entry['goals_b']}</div>"
                f"<div style='font-size:.75rem;color:#94a3b8;'>"
                f"{entry['probability']*100:.1f}%</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Heatmap + model breakdown ──────────────────────────────────────────────
    col_h, col_p = st.columns([3, 2])

    with col_h:
        st.subheader("Distribuição de Placares")
        mat = np.array(result.get("poisson_score_matrix", []))
        if mat.size > 0:
            n = min(mat.shape[0], 7)
            z = mat[:n, :n] * 100
            fig = go.Figure(go.Heatmap(
                z=np.round(z, 2),
                x=[str(i) for i in range(n)],
                y=[str(i) for i in range(n)],
                colorscale="Blues",
                text=np.vectorize(lambda v: f"{v:.1f}%")(z),
                texttemplate="%{text}",
                hovertemplate=f"{team_a} %{{y}} × %{{x}} {team_b}: %{{z:.2f}}%<extra></extra>",
            ))
            fig.update_layout(
                xaxis_title=f"Gols {team_b}", yaxis_title=f"Gols {team_a}",
                height=370, margin=dict(l=40, r=10, t=10, b=40),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_p:
        st.subheader("Contribuição dos Modelos")
        bd = result.get("breakdown", {})
        palette = {"elo": "#3b82f6", "poisson": "#10b981", "xgboost": "#f59e0b"}
        labels, vals, colors = [], [], []
        for mn, data in bd.items():
            w = data.get("weight", 0)
            if w > 0:
                labels.append(mn.upper())
                vals.append(w)
                colors.append(palette.get(mn, "#9ca3af"))

        fig2 = go.Figure(go.Pie(
            labels=labels, values=vals, marker_colors=colors,
            hole=0.4, textinfo="label+percent",
            hovertemplate="%{label}: %{percent}<extra></extra>",
        ))
        fig2.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
            font=dict(color="white"),
        )
        st.plotly_chart(fig2, use_container_width=True)

        rows = []
        for mn, data in bd.items():
            rows.append({
                "Modelo":     mn.upper(),
                f"V {team_a}": f"{data.get('win_a',0)*100:.1f}%",
                "Empate":      f"{data.get('draw',0)*100:.1f}%",
                f"V {team_b}": f"{data.get('win_b',0)*100:.1f}%",
                "Peso":        f"{data.get('weight',0)*100:.0f}%",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Modelo"), use_container_width=True)

    st.markdown("---")

    # ── Top-5 player impact ────────────────────────────────────────────────────
    st.subheader("Top 5 Jogadores por Impacto")
    ia, ib = st.columns(2)

    def _impact_table(impact_list: list, team_name: str, col) -> None:
        if not impact_list:
            col.info("—")
            return
        rows = [
            {"Jogador": p.get("player_name", "—"), "Pos": p.get("position", "—"),
             "Score": f"{p.get('score', 0):.2f}", "Impacto": f"{p.get('impact', 0):+.4f}"}
            for p in impact_list[:5]
        ]
        col.markdown(f"**{team_name}**")
        col.dataframe(pd.DataFrame(rows).set_index("Jogador"), use_container_width=True)

    _impact_table(result.get("top5_impact_a", []), team_a, ia)
    _impact_table(result.get("top5_impact_b", []), team_b, ib)

    if not result.get("ml_trained", False):
        st.markdown(
            '<div class="warn">⚠️ XGBoost não treinado — execute <code>python setup.py</code> '
            'para habilitar. Previsão atual: Elo + Poisson.</div>',
            unsafe_allow_html=True,
        )

    with st.expander("📈 Detalhes técnicos"):
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric(f"Elo {team_a}", int(result.get("elo_a", 0)))
        dc2.metric(f"Elo {team_b}", int(result.get("elo_b", 0)))
        dc3.metric(f"λ {team_a}", f"{result.get('lambda_a', 0):.2f}")
        dc4.metric(f"λ {team_b}", f"{result.get('lambda_b', 0):.2f}")

    if st.button("🧩 Ver análise do elenco"):
        st.session_state["page"] = "🧩 Análise do Elenco"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TELA 3 — Análise do Elenco
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🧩 Análise do Elenco":
    st.title("🧩 Análise do Elenco")

    res_a  = st.session_state.get("team_res_a")
    res_b  = st.session_state.get("team_res_b")
    match  = st.session_state.get("selected_match")

    if res_a is None:
        st.info("👈 Selecione e preveja uma partida no **Cronograma** primeiro.")
        st.stop()

    team_a = match["home"] if match else "Time A"
    team_b = match["away"] if match else "Time B"

    if st.button("← Voltar"):
        st.session_state["page"] = "📊 Resultado da Previsão"
        st.rerun()

    ms1, ms2, ms3, ms4 = st.columns(4)
    ms1.metric(f"Score {team_a}", f"{res_a['team_score']:.3f}")
    ms2.metric(f"Score {team_b}", f"{res_b['team_score']:.3f}")
    ms3.metric(f"Mercado {team_a}", f"€{res_a['market_value_total_m']:.0f}M")
    ms4.metric(f"Mercado {team_b}", f"€{res_b['market_value_total_m']:.0f}M")

    if res_a.get("cohesion_club"):
        st.markdown(f'<div class="info">Coesão {team_a}: ≥6 jogadores do {res_a["cohesion_club"]}</div>',
                    unsafe_allow_html=True)
    if res_b.get("cohesion_club"):
        st.markdown(f'<div class="info">Coesão {team_b}: ≥6 jogadores do {res_b["cohesion_club"]}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # Radar
    st.subheader("Radar — 6 Dimensões")
    da, db = radar_dimensions(res_a), radar_dimensions(res_b)
    cats = list(da.keys()) + [list(da.keys())[0]]
    va   = [da[c] for c in list(da.keys())] + [da[list(da.keys())[0]]]
    vb   = [db[c] for c in list(db.keys())] + [db[list(db.keys())[0]]]

    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(r=va, theta=cats, fill="toself", name=team_a,
                                    line_color="#4ade80", fillcolor="rgba(74,222,128,.15)"))
    fig_r.add_trace(go.Scatterpolar(r=vb, theta=cats, fill="toself", name=team_b,
                                    line_color="#f87171", fillcolor="rgba(248,113,113,.15)"))
    fig_r.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True, height=400, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("---")
    st.subheader("Tabela de Jogadores")
    tab_a, tab_b = st.tabs([f"🟦 {team_a}", f"🟥 {team_b}"])

    def _player_table(res: dict, tab) -> None:
        rated = res.get("rated_players", [])
        if not rated:
            tab.info("—")
            return
        rows = [
            {"Jogador": p.get("player_name", "—"),
             "Pos":    p.get("position", "—"),
             "Clube":  p.get("club", "—"),
             "Liga":   p.get("league", "—"),
             "Coef.":  f"{p.get('league_coeff', 1.0):.2f}",
             "Score":  f"{p.get('score', 0):.2f}",
             "Jogos":  f"{p.get('n_national_games', 0)}{'⚠️' if p.get('low_sample') else ''}"}
            for p in rated
        ]
        tab.dataframe(pd.DataFrame(rows).set_index("Jogador"), use_container_width=True)
        for p in rated:
            if p.get("low_sample"):
                tab.markdown(
                    f'<div class="warn">⚠️ {p["player_name"]} — {p["n_national_games"]} jogo(s) '
                    f'pela seleção (últimos 2 anos).</div>',
                    unsafe_allow_html=True,
                )

    with tab_a:
        _player_table(res_a, tab_a)
    with tab_b:
        _player_table(res_b, tab_b)

    # Position bars
    st.markdown("---")
    st.subheader("Nota Média por Setor")
    pos_a = res_a.get("position_scores", {})
    pos_b = res_b.get("position_scores", {})
    pos_labels = {"GK": "Goleiro", "DEF": "Defesa", "MID": "Meio", "FWD": "Ataque"}
    positions  = list(pos_labels.keys())
    fig_b = go.Figure()
    fig_b.add_trace(go.Bar(
        name=team_a, x=[pos_labels[p] for p in positions],
        y=[pos_a.get(p, 0) for p in positions], marker_color="#4ade80",
    ))
    fig_b.add_trace(go.Bar(
        name=team_b, x=[pos_labels[p] for p in positions],
        y=[pos_b.get(p, 0) for p in positions], marker_color="#f87171",
    ))
    fig_b.update_layout(
        barmode="group", yaxis=dict(title="Nota média", range=[5.5, 9.5]),
        height=300, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_b, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TELA 4 — Mata-Mata
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🏆 Mata-Mata":
    st.title("🏆 Mata-Mata — Copa do Mundo 2026")
    st.caption(
        "Os confrontos do mata-mata serão definidos após o encerramento da fase de grupos "
        "(28/06/2026). Os chaveamentos abaixo mostram os critérios oficiais da FIFA."
    )
    st.markdown("---")

    # ── Round of 32 ───────────────────────────────────────────────────────────
    st.subheader("Oitavas de Final (Round of 32)")
    st.markdown(
        '<div class="info">Os 16 confrontos das oitavas seguem o chaveamento oficial da FIFA. '
        'Os classificados serão preenchidos automaticamente conforme os grupos terminam.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    r32_slots = get_r32_slots()

    # Two columns of matchups
    col_left, col_right = st.columns(2)
    for i, slot in enumerate(r32_slots):
        col = col_left if i % 2 == 0 else col_right
        with col:
            st.markdown(
                f'<div class="tbd-card">'
                f'<div style="font-size:.75rem;color:#64748b;margin-bottom:4px;">{slot["id"]} &nbsp;·&nbsp; {slot.get("date","")}</div>'
                f'<div style="font-size:1rem;font-weight:600;">'
                f'<span style="color:#94a3b8;">{slot["slot_a"]}</span>'
                f' &nbsp;×&nbsp; '
                f'<span style="color:#94a3b8;">{slot["slot_b"]}</span>'
                f'</div>'
                f'<div style="font-size:.7rem;margin-top:6px;color:#475569;">⏳ A definir após fase de grupos</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Legend ────────────────────────────────────────────────────────────────
    st.subheader("📋 Legenda do Chaveamento")
    leg_data = []
    for slot in r32_slots:
        leg_data.append({
            "Jogo":   slot["id"],
            "Slot A": slot["slot_a"],
            "Slot B": slot["slot_b"],
            "Data":   slot.get("date", ""),
        })
    st.dataframe(pd.DataFrame(leg_data).set_index("Jogo"), use_container_width=True)

    st.markdown("")
    st.markdown(
        "**Código dos slots:** `1X` = 1° do grupo X &nbsp;·&nbsp; "
        "`2X` = 2° do grupo X &nbsp;·&nbsp; "
        "`3XYZ...` = melhor 3° dos grupos indicados"
    )

    st.markdown("---")
    st.subheader("Próximas fases")
    for fase in ["Quartas de Final (Quartos de Final)", "Semifinal", "Final — 19 de julho de 2026"]:
        st.markdown(
            f'<div class="tbd-card" style="margin-bottom:8px;">'
            f'<b>{fase}</b><br><span style="font-size:.8rem;">⏳ A definir</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


elif page == "🥇 Probabilidade de Título":
    st.title("🥇 Probabilidade de Ser Campeão")
    st.caption(
        f"Simulação Monte Carlo com **8.000 torneios completos** — fase de grupos + eliminatórias. "
        "Modelo baseado em **ratings Elo**. Resultados em % de chance."
    )

    with st.spinner("Simulando 8.000 torneios... (pode levar ~20s na primeira vez)"):
        champ = _simulate_championship()

    # Ordena por chance de título
    df_champ = pd.DataFrame([
        {
            "Seleção":   team,
            "🏆 Campeão (%)":  data["winner"],
            "🥈 Final (%)":    data["final"],
            "🥉 Semi (%)":     data["semi"],
            "⚽ Quartas (%)":  data["quarter"],
        }
        for team, data in champ.items()
    ]).sort_values("🏆 Campeão (%)", ascending=False).reset_index(drop=True)

    df_champ.index += 1  # ranking começa em 1

    # ── Top 10 — gráfico de barras ────────────────────────────────────────────
    top10 = df_champ.head(10)

    fig = go.Figure(go.Bar(
        x=top10["Seleção"],
        y=top10["🏆 Campeão (%)"],
        marker=dict(
            color=top10["🏆 Campeão (%)"],
            colorscale="Blues",
            showscale=False,
        ),
        text=[f"{v:.1f}%" for v in top10["🏆 Campeão (%)"]],
        textposition="outside",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Top 10 favoritas ao título",
        yaxis=dict(title="Probabilidade (%)", range=[0, top10["🏆 Campeão (%)"].max() * 1.25]),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Tabela completa ───────────────────────────────────────────────────────
    st.subheader("Ranking completo — 48 seleções")

    # Adiciona barra de progresso visual via gradiente de cor
    def _color_champion(val):
        if val >= 15:   return "background-color:#1d4ed8; color:white"
        if val >= 8:    return "background-color:#2563eb; color:white"
        if val >= 3:    return "background-color:#3b82f6; color:white"
        if val >= 1:    return "background-color:#93c5fd; color:#1e3a5f"
        return ""

    styled = (
        df_champ.style
        .map(_color_champion, subset=["🏆 Campeão (%)"])
        .format({
            "🏆 Campeão (%)": "{:.2f}%",
            "🥈 Final (%)":   "{:.1f}%",
            "🥉 Semi (%)":    "{:.1f}%",
            "⚽ Quartas (%)": "{:.1f}%",
        })
    )
    st.dataframe(styled, use_container_width=True, height=420)

    st.markdown("---")
    st.caption(
        "**Metodologia:** cada simulação sorteia resultados com base na probabilidade "
        "derivada do Elo de cada seleção. A fase de grupos determina os 32 classificados "
        "(12 líderes + 12 vice-líderes + 8 melhores 3os colocados). "
        "No mata-mata não há empate — o vencedor é determinado por probabilidade Elo. "
        "Os percentuais convergem com o aumento de simulações (erro ≈ ±0.5pp a 8k sims)."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TELA 6 — Como Funciona
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ Como Funciona":
    st.title("ℹ️ Como a Previsão É Calculada")
    st.caption("Explicação do racional macro dos modelos utilizados para prever os resultados.")

    st.markdown("---")

    # ── Visão geral ────────────────────────────────────────────────────────────
    st.subheader("🔬 Arquitetura: Ensemble de 3 Modelos")
    st.markdown("""
A previsão de cada partida é resultado da **combinação ponderada de três modelos independentes**,
cada um capturando um aspecto diferente da qualidade das seleções:

| Modelo | Peso | O que mede |
|---|---|---|
| **Elo Rating** | ~53% | Força histórica relativa de cada seleção |
| **Poisson + Nota do Elenco** | ~47% | Qualidade atual dos jogadores convocados |
| **XGBoost (ML)** | 0% | Padrão estatístico de múltiplas variáveis *(requer treino)* |

O resultado final (% de vitória / empate / derrota) é a média ponderada dos três.
""")

    st.markdown("---")

    # ── Modelo 1: Elo ─────────────────────────────────────────────────────────
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("📊 Modelo 1 — Elo Rating")
        st.markdown("""
O sistema **Elo** foi criado para xadrez e adaptado para futebol pelo site
[eloratings.net](https://www.eloratings.net). Funciona como um "placar acumulado de reputação":

- Cada seleção tem uma pontuação (ex: Argentina ≈ 2142, Haiti ≈ 1560)
- **Vitória contra time forte** → ganha mais pontos
- **Derrota contra time fraco** → perde mais pontos
- A **diferença de Elo** é convertida em probabilidade via fórmula logística:

$$P(\\text{vitória A}) = \\frac{1}{1 + 10^{-(\\text{Elo}_A - \\text{Elo}_B) / 400}}$$

**Exemplo:** Argentina (2142) vs Argélia (1878):
- Diferença = +264 pontos
- P(vitória Argentina) ≈ **80%**

O modelo Elo é muito bom para capturar a **força histórica**, mas não considera o elenco atual.
        """)

    with col_b:
        st.subheader("⚽ Modelo 2 — Nota do Elenco + Poisson")
        st.markdown("""
Este modelo avalia a **qualidade real dos 11 titulares** convocados para a Copa e usa isso
para estimar quantos gols cada time tende a marcar.

**Passo 1 — Nota por jogador (0 a 10):**
- `rating_nacional × 65%` + `rating_clube × 35%`
- A nota do clube é ajustada pelo coeficiente da liga (Premier League = 1.0, MLS = 0.70, etc.)

**Passo 2 — Nota do time por setor:**
- `Ataque = 0.6 × média FWD + 0.4 × média MID`
- `Defesa = 0.7 × média DEF + 0.3 × nota GK`

**Passo 3 — Lambda (gols esperados):**
$$\\lambda_A = 1{,}35 \\times e^{\\, 0{,}60 \\times \\min(\\text{Ataque}_A - \\text{Defesa}_B,\\ 1{,}5)}$$

Quanto maior o ataque de A comparado à defesa de B, mais gols A tende a marcar.

**Passo 4 — Distribuição de Binomial Negativa:**
Com λ_A e λ_B calculados, gera a **matriz de probabilidade de cada placar possível** (0-0, 1-0, 2-1, etc.)
usando a Binomial Negativa, que dá mais peso a placares elásticos que o Poisson puro.

**Passo 5 — W/D/L:**
- P(vitória A) = soma de todos os placares onde A marca mais
- P(empate) = soma dos placares iguais
- P(vitória B) = o restante
        """)

    st.markdown("---")

    # ── Modelo 3: ML ──────────────────────────────────────────────────────────
    st.subheader("🤖 Modelo 3 — Machine Learning (XGBoost)")
    st.markdown("""
O XGBoost é um algoritmo de **gradient boosting** que aprende padrões a partir de dados históricos
de partidas internacionais. Ele recebe como entrada um vetor de features:

- Diferença de Elo entre as seleções
- Nota agregada do elenco (A e B)
- Valor de mercado total (€M)
- % de jogadores nas 5 maiores ligas
- Fase do torneio (grupo / oitavas / final, etc.)

e prevê diretamente as probabilidades de vitória/empate/derrota.

> ⚠️ **Status atual:** o modelo ML ainda não foi treinado (requer o dataset histórico completo).
> Enquanto isso, o peso do XGBoost é redistribuído entre Elo e Poisson automaticamente.
> Para ativar: execute `python setup.py` na pasta do projeto.
    """)

    st.markdown("---")

    # ── Placares ──────────────────────────────────────────────────────────────
    st.subheader("🎯 Como o Placar Sugerido é Calculado")
    st.markdown("""
O **placar mais provável** é simplesmente a célula com maior probabilidade na matriz de placares
gerada pelo modelo Poisson/NB (por exemplo, "2-0 com 12% de probabilidade").

O **placar esperado** é a média contínua: `E[gols_A] × E[gols_B]`, que tende a ser mais fracionado
(ex: "2.1 – 0.9") e representa o "valor central" da distribuição.

O cronograma exibe o placar modal (mais provável), e a página de análise mostra os **top 5 placares**
com suas respectivas probabilidades.
    """)

    st.markdown("---")

    # ── Probabilidade de título ────────────────────────────────────────────────
    st.subheader("🏆 Probabilidade de Título (Monte Carlo)")
    st.markdown("""
Para calcular a probabilidade de **cada seleção ser campeã**, a ferramenta executa
**8.000 simulações completas** do torneio:

1. **Fase de grupos** — simula todos os 72 jogos usando Elo, determina
   os 12 líderes + 12 vice-líderes + 8 melhores 3os colocados (32 classificados)
2. **Oitavas a Final** — 5 rodadas eliminatórias, sem empate
   (o time com maior probabilidade Elo vence, com aleatoriedade proporcional)
3. Ao final, conta-se **quantas vezes cada seleção ganhou** o torneio
4. Divide pelo total de simulações → % de chance de título

Quanto maior o Elo e mais favoráveis os confrontos do chaveamento,
maior a probabilidade acumulada de ser campeão.
    """)

    st.info(
        "💡 A margem de erro estimada é ±0.5 pontos percentuais para cada seleção "
        "com 8.000 simulações. Para maior precisão, aumente `n_sims` no código."
    )
