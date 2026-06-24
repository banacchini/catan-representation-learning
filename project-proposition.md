"Typ danych: sekwencje obserwacji z perspektywy jednego gracza (catanatron, self-play agentów heurystycznych). Każda obserwacja = wektor informacji widocznych dla gracza X o innych graczach w danej rundzie. Ground truth typów kart dostępny wyłącznie do ewaluacji.
Modele:

Osoba 1: VAE z encoderem LSTM — okno K ostatnich obserwacji → (μ, σ). Badanie hiperparametru β (β-VAE).
Osoba 2: RSSM (Hafner et al. 2019, https://arxiv.org/abs/1811.04551) — inkrementalne aktualizowanie stanu przez strumień deterministyczny (GRU) + stochastyczny.

Downstream task: zamrożony encoder + klasyfikator 4-klasowy → typ każdej trzymanej karty development (rycerz / drogi / VP / obfitość). Każda karta w danej rundzie = osobna próbka. Metryki: F1 per typ, accuracy łączne.
Hipoteza: RSSM agregując całą historię gry da lepsze reprezentacje niepewności niż VAE patrzący na stałe okno, szczególnie w późnych fazach gry.

Baseline: system reguł oparty na wiedzy dziedzinowej — (1) czas trzymania karty bez zagrania (karta trzymana ≥10 rund bez zagrania → rosnący prior na VP), (2) kontekst surowcowy gracza (gracz ma surowce na osadę i nie zagrał karty → malejący prior na drogi; graczowi brakuje dokładnie jednego zasobu do budowy → rosnący prior na obfitość), (3) historia interakcji ze złodziejem (gracz był blokowany i nie zagrał rycerza → malejący prior na rycerza). Reguły łączone przez logistic regression na ręcznie stworzonych cechach.

Podział zadań: obie osoby wspólnie przygotowują pipeline danych przez catanatron, reprezentację wektora obserwacji i implementację baseline regułowego. Osoba 1 implementuje VAE-LSTM . Osoba 2 implementuje RSSM. Wspólna ewaluacja porównawcza: baseline reguły vs VAE vs RSSM.

"