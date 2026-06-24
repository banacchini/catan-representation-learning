## Architektury badane w `src.search`

W searchu porównujemy cztery główne rodziny reprezentacji sekwencyjnych oraz ich warianty nadzorowane. Wszystkie modele dostają tę samą sekwencję publicznych obserwacji gry, a downstream task polega na klasyfikacji typu ukrytej karty development.

### VAE Gaussian (`vae`)

Sekwencyjny VAE z enkoderem LSTM. Dla każdego kroku sekwencji encoder produkuje parametry rozkładu Gaussa:

`q(z_t | x_{\le t}) = N(mu_t, sigma_t^2)`

Dekoder rekonstruuje wektor obserwacji z latentu `z_t`. Funkcja straty:

`reconstruction_loss + beta * KL(q(z_t) || N(0, I))`

Do klasyfikatora downstream trafia deterministyczna reprezentacja `mu_t`.

Badane warianty obejmują m.in. rozmiar latentu, liczbę warstw LSTM, wartość `beta` i learning rate.

### VAE Categorical (`vae_cat`)

Wariant VAE, w którym latent nie jest Gaussowski, tylko kategoryczny. Model produkuje zestaw rozkładów kategorycznych, np. `16 x 16`, a embeddingiem są prawdopodobieństwa klas latentnych.

Różnica względem RSSM categorical: VAE categorical używa prostego, stałego priora jednostajnego. Nie modeluje uczonego priora zależnego od historii.

### RSSM Gaussian (`rssm_gauss`)

Recurrent State Space Model inspirowany PlaNet/Dreamer. Stan składa się z dwóch części:

- `h_t` — deterministyczny stan rekurencyjny aktualizowany przez GRU,
- `z_t` — stochastyczny latent Gaussowski.

Model uczy zarówno prior:

`p(z_t | h_t)`

jak i posterior:

`q(z_t | h_t, x_t)`

Dekoder rekonstruuje obserwację z `[h_t || z_t]`. Do downstream klasyfikatora trafia właśnie konkatenacja `[h_t || z_t]`.

W porównaniu z VAE, RSSM jawnie modeluje dynamikę stanu i uczy prior zależny od historii.

### RSSM Categorical (`rssm_cat`)

Wariant RSSM ze stochastycznym latentem kategorycznym, podobny do DreamerV2. Zamiast Gaussa model używa wielu zmiennych kategorycznych. Nadal zachowuje strukturę RSSM:

- deterministyczny GRU `h_t`,
- uczony prior `p(z_t | h_t)`,
- posterior `q(z_t | h_t, x_t)`,
- rekonstrukcja obserwacji,
- KL posterior-prior.

Dodatkowo używa KL balancing i free bits, żeby stabilizować uczenie priora i posteriora.

## Warianty Supervised A/B/C

Warianty `*_supA`, `*_supB`, `*_supC` sprawdzają, co dzieje się, gdy do treningu reprezentacji dopuszczamy etykiety kart. Nie są więc czystym self-supervised learning, tylko dodatkową osią porównania.

### Supervised A

`A = frozen SSL encoder + supervised head`

Najpierw uczymy encoder dokładnie jak w self-supervised wariancie, czyli przez rekonstrukcję i KL. Potem zamrażamy encoder i uczymy tylko głowicę MLP na etykietach kart.

Schemat:

`sekwencja -> encoder SSL -> stop gradient -> [embedding | CARD_FEATS] -> MLP -> typ karty`

To wariant najbliższy klasycznemu linear/frozen probe, ale głowica jest MLP, a nie regresją logistyczną.

### Supervised B

`B = end-to-end CE + KL`

Encoder i głowica są uczone razem bezpośrednio na zadaniu klasyfikacji. Strata zawiera:

`cross_entropy + sup_kl_weight * KL`

Nie ma rekonstrukcji obserwacji. Model zachowuje wariacyjny bottleneck, ale cała reprezentacja jest optymalizowana pod predykcję typu karty.

Schemat:

`sekwencja -> encoder -> [embedding | CARD_FEATS] -> linear head -> CE`

plus regularyzacja KL.

To bardziej supervised representation learning niż SSL.

### Supervised C

`C = end-to-end CE + KL + reconstruction`

Najbardziej wielozadaniowy wariant. Encoder i głowica są uczone end-to-end, ale model nadal musi rekonstruować obserwację.

Strata:

`cross_entropy + sup_kl_weight * KL + sup_recon_weight * reconstruction_loss`

Czyli reprezentacja ma jednocześnie:
- pomagać w klasyfikacji typu karty,
- zachować strukturę potrzebną do rekonstrukcji stanu gry,
- pozostać regularyzowana przez KL.

Wariant C jest kompromisem między czystym supervised learning a self-supervised reconstruction.

## Intuicja porównania

- `vae`, `vae_cat`, `rssm_gauss`, `rssm_cat` mierzą jakość reprezentacji uczonej bez etykiet kart.
- `*_supA` mierzy, ile daje etykietowana głowica na zamrożonym encoderze.
- `*_supB` mierzy, ile daje pełne dostrojenie encodera pod klasyfikację.
- `*_supC` sprawdza, czy połączenie klasyfikacji i rekonstrukcji daje bardziej użyteczną reprezentację.