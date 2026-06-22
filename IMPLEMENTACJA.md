# Implementacja — generowanie i przygotowanie danych

Dokumentacja decyzji implementacyjnych dotyczących zbioru danych do uczenia
reprezentacji stanu przekonań o ukrytych kartach development w Catan.

## Środowisko

- **Symulator:** `catanatron` zainstalowany **z repozytorium GitHub**, nie z PyPI.
  Wersja z pip (3.2.1) zawiera tylko słabe boty. Silne boty (`AlphaBetaPlayer`,
  `ValueFunctionPlayer`, `MCTSPlayer`, `GreedyPlayoutsPlayer`) są wyłącznie w repo.
  ```bash
  git clone https://github.com/bcollazo/catanatron.git
  cd catan-representation-learning
  uv pip install -e ../catanatron     # lub: source .venv/bin/activate && uv pip install -e ../catanatron
  uv add pandas pyarrow
  ```
- **Python:** wymagane 3.11+ (poprawić `requires-python` w `pyproject.toml` na `>=3.11`).
- **Menedżer pakietów:** `uv`.
- **Generowanie jest CPU-bound** — GPU nie pomaga. Zrównoleglone przez
  `multiprocessing` na wszystkich rdzeniach. 10k gier ≈ 1–2 h lokalnie zależnie
  od wag silnych botów. Chmura niepotrzebna.

## Skład stołu (BOT_POOL) i obserwowani gracze

- **Rozwiązanie A — losowy skład stołu** (realizm). Dla każdej gry losujemy 4 boty
  z powtórzeniami z ważonej puli.
- **Obserwowani** (gracze, których karty przewidujemy) = **tylko silne boty**:
  `ValueFunction`, `AlphaBeta`, `MCTS`. Słabe boty grają losowo → ich użycie kart
  to szum.
- **Wszystkie silne boty przy stole = osobne perspektywy** z jednej gry (więcej
  danych bez dodatkowej symulacji). Perspektywy tej samej gry trafiają do tego
  samego splitu.
- Win-rate (kontrola jakości, 50 gier): ValueFunction ~52%, AlphaBeta ~43%,
  MCTS ~14%, reszta <8% (poziom losowy dla 4 graczy = 25%).
- Czas/gra zależy DRASTYCZNIE od składu: 4×WeightedRandom ~0.04 s,
  4×ValueFunction ~1 s, stół z AlphaBeta ~11 s. Wagi MCTS/AlphaBeta strojone pod
  budżet czasowy.

## Struktura wyjściowa — dwie powiązane tabele

Generowane w chunkach (pliki `*_{start}_to_{end}.parquet`), scalane przez glob.

### `timesteps.parquet` — sekwencja per akcja
Wejście dla VAE-LSTM / RSSM. **Jeden wiersz = jeden krok (gra × perspektywa × akcja)**.
Zawiera WSZYSTKIE kroki, też te z 0 kartami na ręce (sekwencja musi być pełna).
Wiersz zapisywany po **każdej akcji w grze**, niezależnie od tego kto ją wykonał
(~27% kroków to tury obserwowanego, reszta to ruchy przeciwników).

### `card_samples.parquet` — próbki per-karta
Cel: **5-klasowa klasyfikacja typu karty** (softmax). **Jedna próbka = jedna
pojedyncza karta trzymana w danym kroku.** Łączy się z `timesteps` po kluczu
`(game_id, observed_color, action_index)`.

## Wektor obserwacji (cechy wejściowe)

Gracze ułożeni **względem obserwowanego**: `p0` = obserwowany, `p1..p3` = przeciwnicy.
Dla każdego z 4 graczy (cechy JAWNE, widoczne dla każdego):
- `p{i}_total_resources` — suma kart zasobów (liczba jawna, rozbicie ukryte)
- `p{i}_n_dev_in_hand` — suma kart development (liczba jawna, rozbicie = target)
- `p{i}_public_vp` — publiczne punkty (`VICTORY_POINTS`, **NIE** `ACTUAL_VICTORY_POINTS`)
- `p{i}_played_knight/monopoly/road_building/yop/vp` — zagrane karty (jawne)
- `p{i}_has_army`, `p{i}_has_longest_road`, `p{i}_longest_road_len`
- `p{i}_cities_built`, `p{i}_settlements_built`, `p{i}_roads_built`
- `p{i}_is_current` — czy jego tura

Globalne:
- `action_*` — one-hot typu wykonanej akcji (+ `action_OTHER`)
- `robber_on_observed` — czy złodziej blokuje obserwowanego (kluczowy sygnał)
- `num_turns` — postęp gry

Historia obserwowanego (sygnał stanowy):
- `obs_rounds_since_buy`, `obs_rounds_since_play` — ile akcji od ostatniego kupna/zagrania
- `obs_total_dev_bought`, `obs_total_dev_played`
- `n_hidden_cards` — liczba kart na ręce obserwowanego
- `is_observed_turn_start` — flaga początku tury obserwowanego (do filtrowania
  punktów ewaluacji)

## Maskowanie (brak wycieku targetu)

Do cech wejściowych **NIE wchodzi**:
- rozbicie kart development obserwowanego na typy (tylko suma `n_dev_in_hand`),
- rozbicie zasobów żadnego gracza (tylko sumy),
- `ACTUAL_VICTORY_POINTS` (zawiera ukryte karty VP — wyciek!).

Gwarancja maskowania: `p0_n_dev_in_hand == suma y_*` (sprawdzane w weryfikacji).

## Target — próbki per-karta

W `card_samples.parquet`:
- `label` — prawdziwy typ karty (5 klas: KNIGHT, VICTORY_POINT, ROAD_BUILDING,
  MONOPOLY, YEAR_OF_PLENTY)
- `rounds_held` — ile akcji karta jest trzymana (**najsilniejszy sygnał**: karty
  VP trzymane długo — mediana ~95 akcji — vs rycerze zagrywani szybko ~8 akcji)
- `bought_at_action`, `card_slot`, `n_hidden_cards`, `is_observed_turn_start`

Tabela `timesteps` ma też kolumny licznościowe `y_knight … y_year_of_plenty`
(liczba kart każdego typu) — przydatne do alternatywnych analiz.

## Mechanika catanatron — pułapki

- **Tasowanie graczy:** `P0..P3` w `player_state` NIE odpowiadają kolejności
  tworzenia graczy. Jedyne źródło prawdy to `state.color_to_index`
  (np. `{WHITE:0, RED:1, ORANGE:2, BLUE:3}`). Perspektywa identyfikowana
  **kolorem**, P-index wyznaczany przez `color_to_index[kolor]`. Pomylenie tego
  to cichy wyciek (czytanie kart innego gracza).
- **Typ kupionej karty:** w hooku `step()` na żywo `action.value=None`.
  Rozstrzygnięty typ jest dopiero w `state.action_records` PO grze
  (`BUY_DEVELOPMENT_CARD.value == 'KNIGHT'` itd.) — stąd dołączany po grze.
- **Zagrania:** typ zagranej karty wynika z nazwy akcji (`PLAY_KNIGHT_CARD` itd.).
  Karty VP nigdy nie są zagrywane (zostają na ręce do końca).
- **Złodziej:** `board.robber_coordinate` → `board.map.tiles[coord].nodes`
  (6 węzłów) → `board.buildings[node_id] = (kolor, typ)`. Blokada obserwowanego
  występuje w ~20–35% kroków i koreluje z targetem (zablokowany + nie zagrał
  rycerza → niższe P(rycerz): ~1.4% vs ~6.3%).
- **Powtarzalność:** boty używają GLOBALNEGO `random`, nie lokalnego. W każdej
  grze ustawiane `random.seed(game_id)` dla determinizmu.

## Gwarancja poprawności (reconciliation)

W każdym kroku śledzona ręka (rekonstruowana z zakupów/zagrań) MUSI zgadzać się
z `IN_HAND` z `player_state` (assert w generatorze). To gwarantuje poprawność
labeli per-karta — assert złapał błąd mapowania kolorów.

## Split train / val / test (bez wycieku)

**Asymetryczny, kryterium = obecność MCTS w SKŁADZIE STOŁU (kolumna `table`),
NIE `observed_type`:**
- **train / val** — WYŁĄCZNIE gry bez żadnego MCTS przy stole
- **test** — wszystkie gry z MCTS (niewidziany styl, `test_kind='unseen_mcts'`)
  + część czystych gier (widziany styl, `test_kind='seen'`)

Split **per gra** — obie tabele dzielone tym samym podziałem `game_id`, więc
skorelowane wiersze (perspektywy, karty, kolejne akcje) nie przeciekają między
zbiorami. Ewaluacja raportowana osobno dla widzianego i niewidzianego stylu.

> MCTS jako przeciwnik pojawia się dużo częściej niż jako obserwowany — przy
> izolacji znaczna część gier wylatuje z treningu do testu. Wagę MCTS w puli
> dobierać tak, by nie tracić zbyt wielu gier treningowych.

## Niezbalansowanie klas

Rozkład typów kart jest skrajny (VICTORY_POINT dominuje, MONOPOLY i
YEAR_OF_PLENTY <1%). Konsekwencje:
- metryka: **F1 per typ** (accuracy myli przy niezbalansowaniu),
- przy treningu: **balansowanie klas** (class weights / weighted sampling),
- dla rzadkich kart **uśredniać F1 po kilku seedach** (niestabilne).

## Pipeline — kolejność uruchamiania

```bash
# 1. Generowanie (chunki, równolegle)
uv run generate_dataset_v2.py --num 10000 --out-dir data --workers 16 --chunk-size 1000

# 2. Weryfikacja (15 checków: integralność, brak wycieku, spójność tabel, split)
uv run verify_dataset.py --data-dir data

# 3. Podział na train/val/test (izolacja MCTS, per gra)
uv run split_dataset.py --data-dir data --out-dir data/splits
# -> {train,val,test}_{timesteps,card_samples}.parquet
```

## Środowisko modelowe (osobne od generatora)

Część modelowa (analiza + uczenie reprezentacji) ma **własny venv `.venv-ml` na
Pythonie 3.12** — PyTorch nie ma jeszcze wheeli na zainstalowanego systemowo
Pythona 3.14. `catanatron` w tej części jest niepotrzebny (dane są gotowe), więc
środowisko jest lekkie (`requirements-ml.txt`: torch CPU, pandas, pyarrow,
scikit-learn, matplotlib, seaborn, jupyter):

```bash
python -m pip install uv
python -m uv venv .venv-ml --python 3.12
python -m uv pip install --python .venv-ml -r requirements-ml.txt
```

Pliki `verify_dataset.py` i `split_dataset.py` to czysty pandas/numpy — działają
w `.venv-ml`.

## Uczenie reprezentacji — implementacja (`src/`)

Wspólny backbone **Transformer** (`src/models.py`: `SeqTransformerEncoder`)
z **sinusoidalnym kodowaniem pozycji indeksowanym `action_index`** — model uczony
na oknach długości 256 działa na pełnych sekwencjach (do ~673 kroków) przy
ekstrakcji embeddingów. Jednostka sekwencji: `(game_id, observed_color)` sortowana
po `action_index`. Wejście = 82 cechy numeryczne (bez meta i bez `y_*`).
Standaryzacja cech ciągłych na statystykach TRAIN; kolumny binarne (one-hoty,
flagi) wykrywane automatycznie (`src/data.py: fit_feature_spec`).

Trzy **samonadzorowane** obiektywy na tym samym backbone:
- **InfoNCE / CPC** (`src/ssl_infonce.py`) — enkoder causal, predyktory liniowe
  przewidują reprezentacje przyszłych kroków `t+k`; strata InfoNCE z negatywami
  w batchu. Uczy **dynamiki** gry.
- **Barlow Twins** (`src/ssl_barlow.py`) — dwa augmentowane widoki okna
  (`src/augment.py`: feature-dropout, szum na cechach ciągłych, time-dropout),
  strata krzyżowej korelacji → macierz jednostkowa. Uczy **niezmienniczości**.
- **Transformer MAE** (`src/ssl_mae.py`) — maskowanie ~30% kroków i rekonstrukcja
  ich cech (MSE dla ciągłych + BCE dla binarnych). Uczy **struktury** kroku.

**Ewaluacja — linear probe** (`src/probe.py`): enkoder zamrożony → embedding kroku
w pozycji karty + jawne cechy per-karta → regresja logistyczna
(`class_weight='balanced'`). Metryka: **macro-F1 + F1 per klasa**, raportowane
osobno dla `test_kind ∈ {seen, unseen_mcts}`. Punkty odniesienia: `raw` (bez
enkodera), `random` (enkoder nieuczony), `supervised` (`src/supervised.py`, górna
granica end-to-end z etykietami).

Orkiestracja: `src/train_all.py` (CLI) — pretrening wszystkich metod + probe +
zapis `results/metrics.json`, `results/losses.json`, `results/encoder_*.pt`.
Hiperparametry CPU-friendly: `src/config.py`. Domyślnie podpróbkowanie gier
(`--subsample-games`) i okno 256 kroków, bo generowanie jest CPU-bound.

```bash
# pretrening 3 metod SSL + baseline'y + probe (CPU)
.venv-ml/Scripts/python -m src.train_all
#   smoke test: --subsample-games 300 --epochs 2 --probe-games 200
```

## Notebooki

- `notebooks/01_eda.ipynb` — analiza danych (rozkłady klas, `rounds_held`,
  robber↔rycerz, długości sekwencji, postęp gry) + baseline regresji logistycznej.
- `notebooks/02_representation_learning.ipynb` — porównanie InfoNCE / Barlow Twins
  / Transformer MAE: krzywe strat, macro-F1 (seen vs unseen), F1 per klasa, t-SNE
  embeddingów. Wymaga wcześniejszego `src/train_all.py`.

## Modele zespołu (równoległe ścieżki)

- **Baseline** (wspólny): reguły dziedzinowe + logistic regression na ręcznych
  cechach (`rounds_held`, kontekst surowcowy, `robber_on_observed`).
- **Osoba 1:** VAE z enkoderem LSTM (okno historii) + badanie hiperparametru β.
- **Osoba 2:** RSSM (Hafner et al. 2019, podstawowy z Dreamera, NIE HiP-RSSM)
  — inkrementalna agregacja stanu przez całą grę.
- **Ścieżka SSL** (powyżej): InfoNCE / Barlow Twins / Transformer — alternatywne,
  samonadzorowane reprezentacje oceniane tym samym protokołem probe.
- Wejście modeli: sekwencja z `timesteps` (grupować po `game_id, observed_color`,
  sortować po `action_index`). Predykcja: 5-klasowa per karta z `card_samples`.
