"""
Global constants and model weights for the World Cup Match Predictor.
"""

# ── Ensemble weights ────────────────────────────────────────────────────────
ENSEMBLE_WEIGHTS = {
    "elo":     0.40,
    "poisson": 0.35,
    "xgboost": 0.25,
}

# ── Player rating weights ────────────────────────────────────────────────────
WEIGHT_NATIONAL_TEAM = 0.65          # ≥5 national games in 24 months
WEIGHT_CLUB          = 0.35

WEIGHT_NATIONAL_TEAM_LOW_SAMPLE = 0.45   # <5 national games
WEIGHT_CLUB_LOW_SAMPLE          = 0.55

MIN_NATIONAL_GAMES_FOR_HIGH_WEIGHT = 5   # threshold to switch weight regime

# Exponential decay — recent games worth more; half-life = 180 days
DECAY_HALF_LIFE_DAYS = 180

# Penalty (%) when only one data source is available
MISSING_SOURCE_PENALTY = 0.05        # 5 %

# ── Position weights ─────────────────────────────────────────────────────────
POSITION_WEIGHTS = {
    "GK":  0.12,   # 1 goalkeeper
    "DEF": 0.26,   # 4 defenders
    "MID": 0.32,   # 3-4 midfielders
    "FWD": 0.30,   # 2-3 forwards
}

# Substitute weight (used only when a starter is injured/suspended)
SUBSTITUTE_WEIGHT_FACTOR = 0.10

# ── Squad bonuses ─────────────────────────────────────────────────────────────
COHESION_THRESHOLD         = 6        # players from the same club to earn bonus
COHESION_BONUS             = 0.03     # +3 % to team score

MIN_WORLD_CUP_GAMES        = 2        # games to count as "experienced"
EXPERIENCE_BONUS_PER_PLAYER = 0.005   # +0.5 % per experienced player

# ── Elo parameters ────────────────────────────────────────────────────────────
K_WORLD_CUP = 60
K_QUALIFIER = 40
K_FRIENDLY  = 20

GOAL_DIFF_MULTIPLIER = {1: 1.0, 2: 1.5, 3: 1.75}
GOAL_DIFF_MULTIPLIER_MAX = 2.0        # for 4+ goal differences

# ── Poisson / Dixon-Coles / Negative-Binomial ────────────────────────────────
N_SIMULATIONS = 50_000
MAX_GOALS     = 8

# Team score → lambda adjustment factor
TEAM_SCORE_LAMBDA_MIN = 0.85
TEAM_SCORE_LAMBDA_MAX = 1.15
TEAM_SCORE_NEUTRAL    = 7.0           # a "7.0/10" squad is baseline (factor=1.0)

# Minimum expected goals for any team — no team scales below this floor.
# Calibrated on WC 1966-2022: even the weakest side scored ~0.45 gpg on average.
LAMBDA_FLOOR = 0.45

# Dixon-Coles low-score correlation.
# Set to 0.0 = disabled (no correction applied).
# The original correction was artificially reducing draws; it is removed here.
DIXON_COLES_RHO = 0.0

# Overdispersion of the Negative Binomial that replaces pure Poisson.
# Var(X) = λ + λ²/r  where  r = λ / NB_DISPERSION.
# NB_DISPERSION = 0.15 adds ~10-20 % more weight on extreme scorelines (3-2, 4-1…).
NB_DISPERSION = 0.15

# Number of top scorelines shown in the detail page.
TOP_SCORELINES_DISPLAY = 5

# ── Attack vs Defence lambda model ───────────────────────────────────────────
# λ_A = GLOBAL_ATT × exp( ATK_DEF_SENSITIVITY × min(att_A − def_B, ATK_DEF_MAX_ADVANTAGE) )
# where  att_A = 0.6·FWD_A + 0.4·MID_A
#        def_B = 0.7·DEF_B + 0.3·GK_B
#        (all on the same 0-10 player-score scale, neutral = 7.0)
#
# ATK_DEF_SENSITIVITY: controls how fast λ grows with the att−def gap.
#   - gap +1.43 (Brazil vs Haiti FWD/DEF diff) → λ ≈ 3.2  (modal score k=3)
#   - gap ±0.0 (equal teams)                   → λ = 1.35 (neutral)
ATK_DEF_SENSITIVITY   = 0.60   # e-fold growth per unit of att-def gap
ATK_DEF_MAX_ADVANTAGE = 1.5    # cap: prevents unrealistic λ > 3.5 even for extreme mismatches

# ── XGBoost ───────────────────────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":    500,
    "max_depth":       4,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":     "mlogloss",
    "random_state":    42,
}
CALIBRATION_METHOD = "isotonic"       # "sigmoid" (Platt) or "isotonic"

# ── League strength coefficients ──────────────────────────────────────────────
LEAGUE_STRENGTH: dict[str, float] = {
    "Premier League":        1.00,
    "La Liga":               0.97,
    "Bundesliga":            0.95,
    "Serie A":               0.93,
    "Ligue 1":               0.90,
    "Eredivisie":            0.85,
    "Primeira Liga":         0.83,
    "Liga MX":               0.78,
    "Saudi Pro League":      0.72,
    "MLS":                   0.70,
    "Brasileirao":           0.75,
    "Argentine Primera":     0.76,
    "Championship (ENG)":    0.80,
    "Other European Top 5B": 0.82,
    "Other":                 0.65,
}

TOP_5_LEAGUES = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}

# ── Tournament phases ─────────────────────────────────────────────────────────
TOURNAMENT_PHASES = {
    "Fase de Grupos":    1,
    "Oitavas de Final":  2,
    "Quartas de Final":  3,
    "Semifinal":         4,
    "Final":             5,
}

# ── Data sources / caching ────────────────────────────────────────────────────
CACHE_TTL_HOURS = 24
CACHE_DIR       = "data/cache"

ELO_DATA_URL    = "https://www.eloratings.net/World.tsv"
KAGGLE_RESULTS_PATH = "data/results.csv"
ELO_CSV_PATH        = "data/elo_ratings.csv"
LEAGUE_STRENGTH_PATH = "data/league_strength.json"
MODEL_SAVE_PATH      = "data/xgb_model.pkl"

# Request delay between scraping calls (seconds)
SCRAPING_DELAY_SECONDS = 2.0

SCRAPING_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# ── National teams list ───────────────────────────────────────────────────────
NATIONAL_TEAMS = [
    "Argentina", "Brazil", "France", "England", "Spain", "Germany",
    "Netherlands", "Portugal", "Belgium", "Italy", "Croatia", "Uruguay",
    "Colombia", "Mexico", "USA", "Japan", "Morocco", "Senegal", "Ecuador",
    "Switzerland", "Denmark", "Poland", "Serbia", "Australia", "South Korea",
    "Ghana", "Cameroon", "Canada", "Qatar", "Wales", "Tunisia", "Saudi Arabia",
    "Iran", "Costa Rica", "Nigeria", "Egypt", "Algeria", "Chile", "Peru",
    "Bolivia", "Paraguay", "Venezuela", "Panama", "Honduras", "El Salvador",
    "New Zealand", "South Africa", "Ivory Coast", "Mali",
]

# ── Demo mode ─────────────────────────────────────────────────────────────────
DEMO_MODE = True    # Use built-in demo data; set False to enable live scraping
