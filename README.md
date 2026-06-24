# Catan Representation Learning

Uczenie reprezentacji stanu przekonań o ukrytych kartach development w grze Catan.
Projekt na przedmiot **Uczenie Reprezentacji** (MSc AI, Politechnika Wrocławska).

Model ma wnioskować o typach ukrytych kart development obserwowanego gracza na
podstawie publicznie widocznego przebiegu gry. Główne porównanie obejmuje
**VAE-LSTM** i **RSSM** względem nieuczonego baseline'u heurystycznego. Repo zawiera
też wcześniejszą ścieżkę Transformer SSL (InfoNCE / Barlow Twins / MAE), traktowaną
jako dodatkowy punkt odniesienia.

## Wymagania

- **Python 3.11+**
- **[uv](https://github.com/astral-sh/uv)** jako jedyny menedżer środowiska i uruchamiania komend
- **catanatron sklonowany z GitHuba** obok tego repozytorium (`../catanatron`) — silne boty nie są dostępne w wersji z PyPI

## Instalacja

Repozytorium `catanatron` powinno znajdować się w katalogu nadrzędnym względem tego
projektu. `pyproject.toml` wskazuje je jako zależność editable przez `uv`.

```bash
# 1. Sklonuj catanatron obok tego projektu
cd ..
git clone https://github.com/bcollazo/catanatron.git
cd catan-representation-learning

# 2. Utwórz środowisko i zainstaluj zależności z uv.lock / pyproject.toml
uv sync

# 3. Opcjonalne zależności notebookowe (Jupyter, matplotlib, seaborn, SHAP, UMAP)
uv sync --extra notebooks
```

Oczekiwana struktura katalogów:

```text
sem1/ur/
├── catanatron/                      # sklonowane repo (silne boty)
└── catan-representation-learning/   # ten projekt
    ├── data/
    └── ...
```

## Struktura repozytorium

```text
catan-representation-learning/
├── generate_dataset_v2.py   # aktualny generator danych (catanatron -> parquet)
├── verify_dataset.py        # weryfikacja zbioru i braku przecieku
├── split_dataset.py         # podział train/val/test bez wycieku per gra
├── benchmark_parallel.py    # pomiar czasu generowania + diagnostyka win-rate
├── pyproject.toml           # zależności i źródło editable ../catanatron
├── uv.lock                  # lockfile środowiska uv
├── requirements-ml.txt      # pomocniczy eksport zależności; uv pozostaje źródłem prawdy
├── src/
│   ├── config.py            # hiperparametry i ścieżki
│   ├── data.py              # sekwencje, FeatureSpec, Dataset/collate, turn-start filtering
│   ├── models.py            # Transformer backbone dla ścieżki SSL
│   ├── augment.py           # augmentacje dla Barlow Twins
│   ├── ssl_infonce.py       # InfoNCE / CPC
│   ├── ssl_barlow.py        # Barlow Twins
│   ├── ssl_mae.py           # Transformer MAE
│   ├── ssl_vae.py           # SeqVAE: LSTM + β-annealing, wariant gauss/cat
│   ├── ssl_rssm.py          # RSSM: GRU + stochastyczny latent, wariant gauss/cat
│   ├── supervised.py        # supervised upper bound dla ścieżki Transformer
│   ├── supervised_seq.py    # supervised warianty A/B/C dla VAE/RSSM
│   ├── probe.py             # frozen encoder + linear probe / raw baseline
│   ├── baseline_heuristic.py # finalny nieuczony baseline regułowy
│   ├── search.py            # search konfiguracji na VAL
│   ├── train_final.py       # finalny trening multi-seed na TEST
│   └── train_all.py         # wcześniejsza orkiestracja Transformer SSL
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_representation_learning.ipynb
│   ├── 03_vae_rssm.ipynb
│   ├── 04_final_report.ipynb
│   └── 05_project_summary.ipynb
├── results/                 # metryki, checkpointy i figury (niecommitowane)
├── data/                    # wygenerowane dane (niecommitowane)
│   └── splits/
├── IMPLEMENTACJA.md         # szczegóły decyzji implementacyjnych
└── README.md
```

## Pipeline danych

### 1. Benchmark generowania (opcjonalnie)

```bash
uv run benchmark_parallel.py --num 50
```

### 2. Generowanie danych

```bash
uv run generate_dataset_v2.py --num 10000 --out-dir data --workers 16 --chunk-size 1000
```

Generator zapisuje chunki:

- `data/timesteps_*.parquet` — sekwencja publicznych obserwacji per akcja,
- `data/card_samples_*.parquet` — próbki per trzymana karta, z 5-klasowym targetem.

### 3. Weryfikacja

```bash
uv run verify_dataset.py --data-dir data
```

Weryfikacja sprawdza m.in. integralność dwóch tabel, brak target leakage, spójność
śledzonej ręki z `IN_HAND` i kompletność kolumn v2.

### 4. Split train/val/test

```bash
uv run split_dataset.py --data-dir data --out-dir data/splits
```

Split jest per `game_id`. Gry z MCTS przy stole trafiają wyłącznie do testu jako
`unseen_mcts`, a część gier bez MCTS tworzy test `seen`.

## Dane

Format danych to parquet. Tabele łączą się po `(game_id, observed_color, action_index)`.

| Tabela | Jednostka wiersza | Zastosowanie |
|---|---|---|
| `timesteps` | gra × obserwowany gracz × akcja | wejście sekwencyjne dla VAE-LSTM/RSSM/SSL |
| `card_samples` | pojedyncza trzymana karta w danym kroku | target 5-klasowy |

Cechy wejściowe zawierają wyłącznie informacje publiczne. Rozbicie ukrytych kart
development (`y_*`) i ukryte punkty VP nie trafiają do wejścia modeli.

## Modele i ewaluacja

### Główna ścieżka projektu

- **Baseline heurystyczny** (`src/baseline_heuristic.py`) — nieuczony system reguł.
  Startuje od priora składu talii i nakłada korekty wynikające z jawnych sygnałów gry:
  długiego trzymania karty, niezagrania rycerza pod blokadą złodzieja, nadmiaru zasobów
  oraz niskiego stanu banku zasobu.
- **VAE-LSTM** (`src/ssl_vae.py`) — sekwencyjny VAE z enkoderem LSTM, β-annealingiem
  i wariantem Gaussowskim lub kategorycznym.
- **RSSM** (`src/ssl_rssm.py`) — Dreamer/PlaNet-style Recurrent State Space Model:
  deterministyczny stan GRU + stochastyczny latent z uczonym priorem/posteriorem.

### Dodatkowa ścieżka Transformer SSL

`src/train_all.py` porównuje InfoNCE/CPC, Barlow Twins i Transformer MAE na wspólnym
Transformer backbone. To dodatkowy eksperyment porównawczy, nie główna hipoteza VAE vs RSSM.

### Protokół downstream

Najważniejszy protokół to frozen encoder + klasyfikator/probe dla 5-klasowej predykcji
typu ukrytej karty. Do embeddingu kroku doklejane są jawne cechy per karta, m.in.
`rounds_held`, `card_slot`, `n_hidden_cards`, `bought_at_action`, `current_rel_pos`.

Główna metryka to **macro-F1**, ponieważ klasy są silnie niezbalansowane. Raportujemy też
F1 per klasa, accuracy pomocniczo oraz wyniki osobno dla `seen`, `unseen_mcts` i
`observed_type`.

## Eksperymenty modelowe

### Search hiperparametrów na VAL

```bash
uv run python -m src.search --subsample-games 1500 --epochs 8 --probe-games 1200
```

Wynik: `results/search_results.json` z rankingiem konfiguracji i najlepszą konfiguracją
per rodzina.

### Finalny trening i ewaluacja TEST

```bash
uv run python -m src.train_final --families all --epochs 25 --seeds 0,1,2
```

Wynik: `results/final_metrics.json` oraz checkpointy `results/final_*_seed*.pt`.

### Baseline heurystyczny

```bash
uv run python -m src.baseline_heuristic
```

Wynik: `results/baseline_metrics.json`, w strukturze zgodnej z `final_metrics.json`.

### Wcześniejsza ścieżka Transformer SSL

```bash
uv run python -m src.train_all
# smoke test: uv run python -m src.train_all --subsample-games 300 --epochs 2 --probe-games 200
```

Wynik: `results/metrics.json`, `results/losses.json`, `results/encoder_*.pt`.

## Notebooki

Notebooki najlepiej uruchamiać z kernelu środowiska `uv`:

```bash
uv run python -m ipykernel install --user --name catan-rl --display-name "catan-rl"
uv run jupyter lab
```

| Notebook | Rola |
|---|---|
| `notebooks/01_eda.ipynb` | EDA: rozkład klas, `rounds_held`, sygnały robbera, pomocniczy baseline logistyczny |
| `notebooks/02_representation_learning.ipynb` | Transformer SSL: InfoNCE, Barlow Twins, MAE, probe, UMAP/SHAP |
| `notebooks/03_vae_rssm.ipynb` | VAE/RSSM, krzywe uczenia, niepewność, β-sweep |
| `notebooks/04_final_report.ipynb` | Agregacja finalnych wyników i figur porównawczych |
| `notebooks/05_project_summary.ipynb` | Samodzielne sprawozdanie końcowe zgodne z poleceniem projektu |

Przykład wykonania notebooka przez `uv`:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_project_summary.ipynb --ExecutePreprocessor.timeout=7200
```

## Uwagi metodologiczne

- W propozycji projektu zapisano zadanie 4-klasowe; finalna implementacja jest 5-klasowa,
  bo rozdziela `MONOPOLY` i `YEAR_OF_PLENTY` zgodnie z pełną talią Catan.
- `data/` i `results/` nie są commitowane do repozytorium; trzeba je wygenerować lokalnie
  albo współdzielić osobno.
- Ewaluacja modeli uczonych używa limitu `eval_seq_len` z konfiguracji, więc przy analizie
  późnej fazy gry trzeba kontrolować pokrycie sekwencji.
- Supervised warianty `*_supA/B/C` nie są tym samym protokołem co czysty frozen encoder + probe;
  w raportach należy je interpretować jako osobną oś porównania.

