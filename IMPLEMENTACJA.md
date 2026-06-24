# Implementacja — dane, modele i ewaluacja

Dokumentacja decyzji implementacyjnych projektu uczenia reprezentacji stanu przekonań
o ukrytych kartach development w Catanie. Opis dotyczy aktualnego pipeline'u v2:
`generate_dataset_v2.py` → `verify_dataset.py` → `split_dataset.py` → trening i ewaluacja w `src/`.

## Środowisko

Całe środowisko jest zarządzane przez **uv**.

```bash
# catanatron musi leżeć obok repozytorium projektu
cd ..
git clone https://github.com/bcollazo/catanatron.git
cd catan-representation-learning

# instalacja zależności z pyproject.toml / uv.lock
uv sync

# opcjonalne zależności notebookowe
uv sync --extra notebooks
```

W `pyproject.toml` zależność `catanatron` jest wskazana jako lokalne źródło editable:
`../catanatron`. Korzystamy z wersji z GitHuba, ponieważ silne boty (`AlphaBetaPlayer`,
`ValueFunctionPlayer`, `MCTSPlayer`) nie są dostępne w pakiecie PyPI.

Generowanie danych jest CPU-bound. GPU jest potrzebne dopiero przy większych treningach
modeli, zwłaszcza finalnym `src.train_final`.

## Skład stołu i wybór perspektywy

Każda gra losuje czterech graczy z ważonej puli `BOT_POOL` w `generate_dataset_v2.py`:
`Random`, `WeightedRandom`, `VictoryPoint`, `ValueFunction`, `AlphaBeta`, `MCTS`.

Obserwowanym graczem może być tylko silny bot z `OBSERVABLE_TYPES`:
`ValueFunction`, `AlphaBeta`, `MCTS`. Jeśli wylosowany stół nie zawiera żadnego takiego
bota, gra jest pomijana.

Aktualna decyzja v2: **dokładnie jedna perspektywa na grę**. Spośród silnych botów przy
stole losowany jest jeden `observed_color`. Ogranicza to korelację między próbkami z tej
samej rozgrywki i upraszcza split per gra. Starszy wariant „wszystkie silne boty jako
osobne perspektywy” nie jest już używany w generatorze v2.

## Struktura wyjściowa danych

Generator zapisuje dane w chunkach parquet.

### `timesteps_*.parquet`

Jeden wiersz to jeden krok gry z perspektywy obserwowanego gracza:
`(game_id, observed_color, action_index)`. Tabela zawiera pełną sekwencję akcji, także
kroki, w których obserwowany gracz nie ma ukrytych kart development. Modele sekwencyjne
potrzebują pełnego kontekstu historii, nie tylko kroków z targetem.

### `card_samples_*.parquet`

Jeden wiersz to jedna konkretna karta development trzymana przez obserwowanego gracza w
danym kroku. Tabela łączy się z `timesteps` po `(game_id, observed_color, action_index)`.
Targetem jest `label`, czyli 5-klasowy typ karty:

- `KNIGHT`
- `VICTORY_POINT`
- `ROAD_BUILDING`
- `MONOPOLY`
- `YEAR_OF_PLENTY`

To świadome odejście od pierwotnego opisu 4-klasowego: finalny dataset rozdziela
`MONOPOLY` i `YEAR_OF_PLENTY`, bo są odrębnymi kartami i mają różne konsekwencje w grze.

## Wektor obserwacji

Gracze są indeksowani relatywnie do obserwowanego:

- `p0` — obserwowany gracz,
- `p1..p3` — pozostali gracze w stałej kolejności względem mapowania kolorów `catanatron`.

Dla każdego gracza zapisujemy wyłącznie publiczne cechy, m.in.:

- `p{i}_total_resources` — suma zasobów, bez rozbicia na typy,
- `p{i}_n_dev_in_hand` — liczba kart development na ręce, bez rozbicia na typy,
- `p{i}_public_vp` — publiczne punkty, nie `ACTUAL_VICTORY_POINTS`,
- zagrane typy kart development,
- największą armię, najdłuższą drogę, liczbę miast, osad i dróg,
- `p{i}_is_current` — informację, czy to aktualny gracz.

Cechy globalne i historyczne obejmują m.in.:

- `action_*` — one-hot typu wykonanej akcji,
- `robber_on_observed`,
- `num_turns`,
- `bank_*`,
- `played_*_total`,
- `dev_deck_remaining`,
- `obs_rounds_since_buy`, `obs_rounds_since_play`,
- `obs_total_dev_bought`, `obs_total_dev_played`,
- `n_hidden_cards`,
- `is_observed_turn_start`.

`src.data.feature_columns()` automatycznie wybiera numeryczne kolumny wejściowe, odrzucając
meta-kolumny i targety `y_*`.

## Brak wycieku targetu

Do cech wejściowych modeli nie trafiają:

- rozbicie ukrytych kart development obserwowanego na typy,
- `ACTUAL_VICTORY_POINTS`, które ujawnia ukryte VP,
- rozbicie zasobów żadnego gracza,
- targety pomocnicze `y_knight`, `y_victory_point`, `y_road_building`, `y_monopoly`, `y_year_of_plenty`.

Generator śledzi rękę obserwowanego gracza na podstawie zakupów i zagrań kart. W każdym
kroku porównuje zrekonstruowaną rękę z `IN_HAND` z `catanatron`; niezgodność kończy się
assertem. Ten mechanizm złapał wcześniej błędy mapowania kolorów.

## Mechanika catanatron — ważne pułapki

- `P0..P3` w `player_state` nie odpowiadają kolejności tworzenia graczy. Źródłem prawdy
  jest `state.color_to_index`.
- Typ kupionej karty nie jest dostępny bezpośrednio w hooku `step()`; jest odczytywany po
  grze z `state.action_records`.
- Typ zagranej karty wynika z nazwy akcji (`PLAY_KNIGHT_CARD`, `PLAY_MONOPOLY`, itd.).
- Karty VP nigdy nie są zagrywane, dlatego długi czas trzymania jest silnym sygnałem VP.
- Boty używają globalnego `random`, więc przed każdą grą ustawiamy `random.seed(game_id)`.

## Split train / val / test

Split jest wykonywany per `game_id`, wspólnie dla `timesteps` i `card_samples`.

Kryterium testu generalizacji to obecność MCTS w składzie stołu (`table`), a nie tylko
`observed_type`:

- **train / val** — gry bez żadnego MCTS przy stole,
- **test unseen_mcts** — wszystkie gry z MCTS przy stole,
- **test seen** — część gier bez MCTS, zostawiona jako znany styl w teście.

Dzięki temu test mierzy zarówno standardową generalizację, jak i odporność na niewidziany
styl gry.

## Niezbalansowanie klas

Rozkład kart development jest silnie niezbalansowany: `VICTORY_POINT` i `KNIGHT` dominują,
a `MONOPOLY` oraz `YEAR_OF_PLENTY` są rzadkie. Dlatego:

- główną metryką jest **macro-F1**,
- raportujemy F1 per klasa,
- probe i warianty nadzorowane używają mechanizmów ważenia klas,
- wyniki finalne uśredniamy po seedach, jeśli dostępne.

## Przygotowanie danych — komendy

```bash
# 1. Generowanie danych
uv run generate_dataset_v2.py --num 10000 --out-dir data --workers 16 --chunk-size 1000

# 2. Weryfikacja integralności i braku przecieku
uv run verify_dataset.py --data-dir data

# 3. Split train/val/test
uv run split_dataset.py --data-dir data --out-dir data/splits
```

Wynikiem splitu są pliki:

```text
data/splits/train_timesteps.parquet
data/splits/train_card_samples.parquet
data/splits/val_timesteps.parquet
data/splits/val_card_samples.parquet
data/splits/test_timesteps.parquet
data/splits/test_card_samples.parquet
```

## Implementacja modeli (`src/`)

### Warstwa danych

`src.data` odpowiada za:

- wczytywanie splitów,
- budowę sekwencji `(game_id, observed_color)`,
- standaryzację cech ciągłych na statystykach TRAIN,
- wykrywanie kolumn binarnych,
- filtrowanie próbek do startów tur (`compute_turn_starts`, `filter_to_turn_starts`),
- doklejanie jawnych cech per karta (`CARD_FEATS`) podczas probe.

### VAE-LSTM

`src.ssl_vae` implementuje `SeqVAE`:

- LSTM jako enkoder sekwencji,
- latent Gaussowski (`vae`) albo kategoryczny (`vae_cat`),
- dekoder MLP rekonstruujący wektor obserwacji,
- strata rekonstrukcji + `β * KL`,
- β-annealing przez pierwsze epoki.

W probe używany jest deterministyczny embedding: `mu_t` dla wariantu Gaussowskiego albo
prawdopodobieństwa klas latentnych dla wariantu kategorycznego.

### RSSM

`src.ssl_rssm` implementuje RSSM w stylu Dreamer/PlaNet:

- embedding obserwacji,
- deterministyczny stan `h_t` aktualizowany przez `GRUCell`,
- stochastyczny latent `z_t`,
- prior `p(z_t | h_t)` i posterior `q(z_t | h_t, x_t)`,
- dekoder rekonstruujący obserwację,
- KL posterior-prior.

Wariant kategoryczny używa KL balancing i free bits. Embedding do probe to konkatenacja
`[h_t || z_t]`.

### Transformer SSL

Dodatkowa ścieżka `src.train_all` korzysta z `SeqTransformerEncoder` i trzech obiektywów:

- `src.ssl_infonce` — InfoNCE / CPC,
- `src.ssl_barlow` — Barlow Twins,
- `src.ssl_mae` — Transformer MAE.

Ta część jest punktem odniesienia, ale finalna hipoteza projektu dotyczy głównie VAE-LSTM
kontra RSSM.

### Supervised warianty A/B/C

`src.supervised_seq` implementuje dodatkowe warianty nadzorowane dla VAE/RSSM:

- **A** — zamrożony encoder SSL + głowica MLP,
- **B** — end-to-end CE + KL,
- **C** — end-to-end CE + KL + rekonstrukcja.

Nie należy ich interpretować jako ten sam protokół co frozen encoder + linear probe.
Są dodatkowym punktem odniesienia dla pytania, ile daje wykorzystanie etykiet podczas treningu.

## Baseline

Finalny baseline to `src.baseline_heuristic`, czyli nieuczony system reguł. Startuje od
priora składu talii development i nakłada multiplikatywne korekty:

- długie trzymanie karty zwiększa prior `VICTORY_POINT`, bo VP nie da się zagrać,
- blokada złodziejem bez zagrania rycerza zmniejsza prior `KNIGHT`,
- nadmiar zasobów bez zagrania karty zmniejsza priory `ROAD_BUILDING` i `YEAR_OF_PLENTY`,
- niski stan banku zasobu jest proxy dla okazji do monopolu i wpływa na `MONOPOLY`.

Parametry są ręcznie ustawione w `HeuristicParams`; baseline nie uczy się na danych.
Wynik zapisywany jest do `results/baseline_metrics.json` w strukturze zgodnej z
`results/final_metrics.json`.

```bash
uv run python -m src.baseline_heuristic
```

Logistic regression z `notebooks/01_eda.ipynb` jest pomocniczym baseline'em analitycznym
na prostych cechach per karta, ale nie jest finalnym baseline'em projektu.

## Search i finalny trening

### Search na VAL

```bash
uv run python -m src.search --subsample-games 1500 --epochs 8 --probe-games 1200
```

`src.search` trenuje kuratorowaną przestrzeń konfiguracji VAE/RSSM i supervised A/B/C,
ocenia na VAL i zapisuje `results/search_results.json`.

### Final na TEST

```bash
uv run python -m src.train_final --families all --epochs 25 --seeds 0,1,2
```

`src.train_final` bierze najlepsze konfiguracje z searcha, trenuje je na pełniejszym
budżecie i zapisuje:

- `results/final_metrics.json`,
- `results/final_<family>_seed<n>.pt`.

## Notebooki i raportowanie

| Notebook | Rola |
|---|---|
| `01_eda.ipynb` | analiza danych, rozkłady, sygnały domenowe, pomocniczy baseline logistyczny |
| `02_representation_learning.ipynb` | Transformer SSL i porównania probe |
| `03_vae_rssm.ipynb` | VAE/RSSM, analiza niepewności, β-sweep |
| `04_final_report.ipynb` | agregacja wyników finalnych i figur |
| `05_project_summary.ipynb` | samodzielne sprawozdanie końcowe zgodne z poleceniem |

Uruchamianie przez `uv`:

```bash
uv run python -m ipykernel install --user --name catan-rl --display-name "catan-rl"
uv run jupyter lab

# albo batchowo
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_project_summary.ipynb --ExecutePreprocessor.timeout=7200
```

## Znane ograniczenia

- `data/` i `results/` nie są commitowane; notebooki wynikowe wymagają lokalnego odtworzenia artefaktów.
- Ewaluacja embeddingów jest ograniczona przez `eval_seq_len`; przy analizie późnej fazy gry trzeba kontrolować pokrycie sekwencji.
- Rzadkie klasy mają szeroką wariancję F1, więc pojedyncze uruchomienia mogą być mylące.
- Własny dataset nie ma zewnętrznego state of the art; porównanie odbywa się względem baseline'u heurystycznego i wewnętrznych punktów odniesienia.

