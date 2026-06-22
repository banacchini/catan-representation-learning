# Catan Representation Learning

Uczenie reprezentacji stanu przekonań o ukrytych kartach development w grze Catan.
Projekt na przedmiot **Uczenie Reprezentacji** (MSc AI, Politechnika Wrocławska).

Model uczy się wnioskować o ukrytych kartach development obserwowanego gracza na
podstawie publicznie widocznego przebiegu gry. Porównujemy reprezentację z **VAE
(enkoder LSTM)** oraz **RSSM** względem baseline'u regułowego.

## Wymagania

- **Python 3.11+**
- **[uv](https://github.com/astral-sh/uv)** jako menedżer pakietów
- **catanatron sklonowany z GitHuba** — patrz niżej (silne boty nie są w wersji z PyPI)

## Instalacja

Repozytorium `catanatron` musi być **sklonowane w katalogu nadrzędnym** względem
tego projektu (`../catanatron`). Wersja z PyPI zawiera tylko słabe boty; silne
(`AlphaBetaPlayer`, `ValueFunctionPlayer`, `MCTSPlayer`) są wyłącznie w repo.

```bash
# 1. Sklonuj catanatron OBOK tego projektu
cd ..
git clone https://github.com/bcollazo/catanatron.git
cd catan-representation-learning

# 2. Środowisko i zależności
uv venv
uv pip install -e ../catanatron
uv add pandas pyarrow
```

Oczekiwana struktura katalogów:

```
sem1/ur/
├── catanatron/                      # sklonowane repo (silne boty)
└── catan-representation-learning/   # ten projekt
    ├── data/
    └── ...
```

> Jeśli `uv` zgłasza konflikt wersji Pythona, ustaw `requires-python = ">=3.11"`
> w `pyproject.toml`.

## Struktura repozytorium

```
catan-representation-learning/
├── generate_dataset_v2.py   # generator danych (catanatron -> parquet)
├── verify_dataset.py        # weryfikacja zbioru (15 checków)
├── split_dataset.py         # podział train/val/test bez wycieku
├── benchmark_parallel.py    # pomiar czasu generowania + diagnostyka win-rate
├── requirements-ml.txt      # zależności części modelowej (.venv-ml, CPU)
├── src/                     # uczenie reprezentacji (SSL) + ewaluacja
│   ├── config.py            #   hiperparametry (CPU-friendly)
│   ├── data.py              #   sekwencje, FeatureSpec, Dataset/collate
│   ├── models.py            #   wspólny enkoder Transformer + głowice
│   ├── augment.py           #   augmentacje okien (Barlow Twins)
│   ├── ssl_infonce.py       #   pretrening InfoNCE / CPC
│   ├── ssl_barlow.py        #   pretrening Barlow Twins
│   ├── ssl_mae.py           #   pretrening Transformer MAE
│   ├── supervised.py        #   górna granica (end-to-end)
│   ├── probe.py             #   linear-probe + baseline raw
│   └── train_all.py         #   orkiestracja: pretrening + probe -> results/
├── notebooks/
│   ├── 01_eda.ipynb         #   analiza danych, wykresy, baseline
│   └── 02_representation_learning.ipynb  # porównanie metod, t-SNE, F1
├── results/                 # metrics.json, losses.json, encoder_*.pt (nie commitowane)
├── data/                    # wygenerowane chunki parquet
│   └── splits/              # train/val/test
├── IMPLEMENTACJA.md         # szczegóły decyzji implementacyjnych
└── README.md
```

## Pipeline

### 1. (Opcjonalnie) Benchmark — ile potrwa generowanie

```bash
uv run benchmark_parallel.py --num 50
```
Mierzy czas na wszystkich rdzeniach i ekstrapoluje do 10k gier. Pokazuje też
win-rate botów. Generowanie jest **CPU-bound** — GPU nie pomaga.

### 2. Generowanie danych

```bash
uv run generate_dataset_v2.py --num 10000 --out-dir data --workers 16 --chunk-size 1000
```
Produkuje dwie powiązane tabele w chunkach:
- `data/timesteps_*.parquet` — sekwencja per akcja (wejście dla VAE/RSSM)
- `data/card_samples_*.parquet` — próbki per-karta (5-klasowy target)

Argumenty: `--num` liczba gier, `--workers` liczba procesów,
`--chunk-size` gier na plik, `--start-id` początkowe ID gry (do wznawiania).

Długi przebieg najlepiej puścić w tle:
```bash
nohup uv run generate_dataset_v2.py --num 10000 --out-dir data --chunk-size 1000 &
```

### 3. Weryfikacja

```bash
uv run verify_dataset.py --data-dir data
```
15 checków: integralność, brak wycieku targetu, spójność obu tabel, poprawność
splitu. Powinno wyjść `15/15`.

### 4. Podział na zbiory

```bash
uv run split_dataset.py --data-dir data --out-dir data/splits
```
Tworzy `{train,val,test}_{timesteps,card_samples}.parquet`. Split **per gra**,
z izolacją MCTS (gry z MCTS przy stole trafiają wyłącznie do testu jako
niewidziany styl).

## Dane

Format **parquet** (łatwy do współdzielenia, `pd.read_parquet(...)`).

| Tabela | Jednostka wiersza | Zastosowanie |
|---|---|---|
| `timesteps` | gra × perspektywa × akcja | wejście sekwencyjne (VAE-LSTM, RSSM) |
| `card_samples` | pojedyncza trzymana karta | target 5-klasowy (klasyfikacja typu) |

Łączenie tabel po kluczu `(game_id, observed_color, action_index)`.
Dla modeli sekwencyjnych: grupować `timesteps` po `(game_id, observed_color)`,
sortować po `action_index`.

Szczegóły wektora cech, maskowania, mechaniki catanatron i logiki splitu:
patrz **`IMPLEMENTACJA.md`**.

> `data/` nie jest commitowane do repo (rozmiar). Zbiór generuje się lokalnie
> albo współdzieli osobno (np. Dysk Google).

## Modele

- **Baseline** — reguły dziedzinowe + logistic regression na ręcznych cechach.
- **Osoba 1** — VAE z enkoderem LSTM (+ badanie hiperparametru β).
- **Osoba 2** — RSSM (Hafner et al. 2019).

Metryka: **F1 per typ karty** (zbiór jest niezbalansowany — accuracy myli).
Ewaluacja osobno dla widzianego i niewidzianego (MCTS) stylu gry.

## Analiza danych i uczenie reprezentacji (SSL)

Dodatkowy moduł: **samonadzorowane uczenie reprezentacji** sekwencji stanu gry na
wspólnym backbone **Transformer**, porównanie trzech obiektywów oceniane protokołem
**linear-probe** (F1 per typ, osobno seen vs `unseen_mcts`).

| Metoda | Typ | Plik |
|---|---|---|
| **InfoNCE / CPC** | kontrastywne, czasowe (predykcja przyszłych kroków) | `src/ssl_infonce.py` |
| **Barlow Twins** | nie-kontrastywne (niezmienniczość + dekorelacja) | `src/ssl_barlow.py` |
| **Transformer MAE** | masked modeling (rekonstrukcja zamaskowanych kroków) | `src/ssl_mae.py` |

Punkty odniesienia: `raw` (bez enkodera), `random` (enkoder nieuczony),
`supervised` (górna granica). Enkoder + protokół: `src/models.py`, `src/probe.py`.

### Środowisko (osobne, CPU-only)

PyTorch nie ma jeszcze wheeli na Pythona 3.14, więc analiza ma **własny venv 3.12**
(catanatron tu niepotrzebny — dane są gotowe):

```bash
python -m pip install uv
python -m uv venv .venv-ml --python 3.12
python -m uv pip install --python .venv-ml -r requirements-ml.txt
```

### Uruchomienie

```bash
# 1. podział na zbiory (jeśli nie zrobiony) — czysty pandas, działa w .venv-ml
.venv-ml/Scripts/python split_dataset.py --data-dir data --out-dir data/splits

# 2. pretrening 3 metod SSL + baseline'y + linear probe (CPU, kilkadziesiąt minut)
.venv-ml/Scripts/python -m src.train_all
#    smoke test:  --subsample-games 300 --epochs 2 --probe-games 200
#    wynik: results/metrics.json, results/losses.json, results/encoder_*.pt

# 3. notebooki (uruchom kernel z .venv-ml)
#    notebooks/01_eda.ipynb                  — analiza danych, wykresy, baseline
#    notebooks/02_representation_learning.ipynb — porównanie metod, t-SNE, F1
```

Główne parametry `src/train_all.py`: `--subsample-games`, `--epochs`, `--batch-size`,
`--probe-games`, `--methods` (np. `raw,random,infonce,barlow,mae,supervised`).
Pozostałe hiperparametry: `src/config.py`.

## Uwagi

- Generowanie jest deterministyczne (`random.seed(game_id)` per gra).
- Klasy są skrajnie niezbalansowane — przy treningu stosować class weights /
  weighted sampling, dla rzadkich kart uśredniać F1 po seedach.
