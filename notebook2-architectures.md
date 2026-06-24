## Architektury i metody z notebooka 02

Notebook `notebooks/02_representation_learning.ipynb` opisuje dodatkową ścieżkę eksperymentalną projektu: samonadzorowane uczenie reprezentacji na wspólnym backbone **Transformer**. Nie jest to główna ścieżka VAE/RSSM z finalnego searcha, ale dodatkowy punkt odniesienia wykonany na tym samym zadaniu downstream: 5-klasowej klasyfikacji ukrytej karty development.

Wspólny protokół głównej ewaluacji:

`sekwencja publicznych obserwacji -> encoder -> embedding kroku -> [embedding | CARD_FEATS] -> classifier/probe -> typ karty`

Główne metryki to `macro-F1`, F1 per klasa oraz rozbicie `seen` / `unseen_mcts`.

## Wspólny backbone: Transformer (`SeqTransformerEncoder`)

Wszystkie metody SSL z notebooka 02 korzystają z tego samego encodera `SeqTransformerEncoder` z `src/models.py`.

Model działa na sekwencji kroków gry:

`x_1, x_2, ..., x_T`

Każdy krok jest wektorem publicznych cech obserwacji. Najpierw cechy są rzutowane liniowo do wymiaru `d_model`, a następnie dodawane jest sinusoidalne kodowanie pozycji zależne od rzeczywistego `action_index`.

Schemat:

`features_t -> Linear(F, d_model) + positional_encoding(action_index_t) -> TransformerEncoder -> h_t`

Encoder zwraca embedding per krok:

`h_1, h_2, ..., h_T`

W zależności od metody działa w trybie:

- **bidirectional** — krok może widzieć kontekst z całego okna,
- **causal** — krok `t` widzi tylko przeszłość, bez przyszłych kroków.

Tryb causal jest używany w InfoNCE/CPC, bo ta metoda ma przewidywać przyszłość z przeszłości.

## Raw baseline (`raw`)

`raw` nie jest modelem reprezentacji. To baseline bez encodera.

Regresja logistyczna dostaje:

`[surowe cechy kroku | CARD_FEATS]`

gdzie `CARD_FEATS` to jawne cechy per karta, m.in.:

- `rounds_held`,
- `card_slot`,
- `n_hidden_cards`,
- `bought_at_action`,
- `current_rel_pos`.

Ten baseline odpowiada na pytanie: ile da się osiągnąć bez uczenia reprezentacji, używając tylko jawnych cech wejściowych i cech karty.

## Random encoder (`random`)

`random` używa tego samego Transformera co metody SSL, ale encoder nie jest trenowany.

Schemat:

`sekwencja -> losowy Transformer -> embedding -> linear probe`

Ten wariant jest ważnym punktem kontrolnym. Jeśli metoda SSL nie przebija `random`, oznacza to, że sam pretraining nie wnosi sygnału użytecznego dla downstream task ponad losową projekcję i cechy per karta.

W notebooku 02 `random` okazał się bardzo mocnym baseline'em, co sugeruje, że duża część sygnału jest już liniowo dostępna w cechach wejściowych i `CARD_FEATS`.

## InfoNCE / CPC (`infonce`)

InfoNCE/CPC to kontrastywna metoda samonadzorowana ucząca dynamiki sekwencji.

Encoder działa w trybie causal:

`c_t = encoder(x_{\le t})`

Następnie liniowe predyktory `W_k` próbują przewidzieć reprezentację przyszłego kroku:

`W_k c_t -> z_{t+k}`

Dla każdego kroku model ma rozpoznać prawdziwy przyszły krok spośród negatywów w batchu.

Intuicja:

- reprezentacja powinna zawierać informacje o tym, jak gra będzie ewoluować,
- model uczy dynamiki publicznego stanu gry,
- brak dostępu do przyszłości wymusza kauzalne kodowanie historii.

Strata:

`InfoNCE(predicted_future, true_future, batch_negatives)`

W notebooku 02 InfoNCE jest najbardziej „czasową” metodą, ale może tracić informacje przydatne do klasyfikacji karty, jeśli downstream korzysta z pełnego kontekstu lub cech silnie związanych z aktualnym stanem.

## Barlow Twins (`barlow`)

Barlow Twins to niekontrastywna metoda SSL bez negatywów.

Dla tego samego okna sekwencji tworzone są dwa zaugmentowane widoki:

`view_1, view_2 = augment(sequence)`

Augmentacje obejmują m.in. dropout cech, szum na cechach ciągłych i time dropout.

Oba widoki przechodzą przez ten sam encoder Transformer. Następnie embedding okna jest liczony przez masked mean pooling, a potem przechodzi przez projektor MLP.

Schemat:

`view_1 -> encoder -> pooling -> projector -> z_1`

`view_2 -> encoder -> pooling -> projector -> z_2`

Strata wymusza, aby macierz korelacji krzyżowej między `z_1` i `z_2` była bliska macierzy jednostkowej:

- przekątna bliska `1` — reprezentacje dwóch widoków mają nieść tę samą informację,
- elementy poza przekątną bliskie `0` — wymiary reprezentacji nie powinny być redundantne.

Intuicja:

- model uczy reprezentacji odpornej na drobne zakłócenia,
- nie potrzebuje negatywnych przykładów,
- promuje inwariancję i dekorelację cech latentnych.

## Transformer MAE (`mae`)

Transformer MAE to masked autoencoding dla sekwencji obserwacji.

Losowo maskujemy część kroków sekwencji, domyślnie około `30%`:

`x_t -> mask_token`

Encoder Transformer widzi uszkodzoną sekwencję i ma odtworzyć oryginalne cechy zamaskowanych kroków.

Schemat:

`masked sequence -> Transformer encoder -> MAE decoder -> reconstructed features`

Strata rekonstrukcji jest mieszana:

- MSE dla cech ciągłych,
- BCE dla cech binarnych, np. one-hotów akcji i flag.

Intuicja:

- model uczy struktury pojedynczego kroku i lokalnego kontekstu,
- reprezentacja powinna pomagać w uzupełnianiu brakujących informacji o stanie gry,
- metoda jest podobna ideowo do BERT/MAE, ale zastosowana do sekwencji tabularnych obserwacji gry.

## Supervised upper bound (`supervised`)

`supervised` w notebooku 02 to nadzorowany punkt odniesienia dla architektury Transformer.

W przeciwieństwie do metod SSL, encoder widzi etykiety kart podczas treningu. Jest uczony end-to-end razem z liniową głowicą klasyfikacyjną.

Schemat:

`sekwencja -> Transformer encoder -> [embedding | CARD_FEATS] -> linear head -> typ karty`

Strata:

`cross_entropy`

z wagami klas, żeby częściowo kompensować niezbalansowanie targetu.

Ten model nie jest metodą samonadzorowaną. Pokazuje raczej, ile może osiągnąć podobna architektura, jeśli dostanie etykiety w trakcie treningu.

Dlatego w raportowaniu najlepiej opisywać go jako **upper bound** albo dodatkowy punkt odniesienia, a nie jako bezpośredniego konkurenta metod SSL.

## Dodatkowe analizy w notebooku 02

Notebook 02 zawiera także analizy przestrzeni reprezentacji i zachowania probe. One nie są osobnymi architekturami, ale pomagają interpretować wyniki.

### UMAP

UMAP rzutuje embeddingi kroków na 2D, żeby sprawdzić, czy klasy kart tworzą widoczne skupiska.

Wynik negatywny, np. brak wyraźnych klastrów, oznacza, że klasy nie są geometrycznie dobrze odseparowane w przestrzeni embeddingów, nawet jeśli linear probe potrafi coś z nich odczytać.

### kNN-probe

kNN-probe mierzy jakość reprezentacji bez uczenia parametrycznej głowicy.

Zamiast regresji logistycznej klasyfikujemy próbkę przez najbliższych sąsiadów w przestrzeni embeddingów.

To odpowiada na pytanie: czy reprezentacja sama z siebie układa próbki tego samego typu blisko siebie?

### Silhouette

Silhouette mierzy separowalność klas w przestrzeni embeddingów.

Wysoki wynik oznacza zwarte, dobrze rozdzielone klastry. Wynik bliski zera lub ujemny oznacza, że klasy nachodzą na siebie geometrycznie.

### SHAP dla probe

SHAP jest liczony dla regresji logistycznej na:

`[embedding | CARD_FEATS]`

Analiza pokazuje, czy predykcję napędzają głównie wymiary embeddingu, czy jawne cechy per karta, takie jak `rounds_held`.

To jest ważne, bo jeśli `CARD_FEATS` dominują, to różnice między encoderami mogą być małe niezależnie od jakości pretrainingu.

### Label efficiency

Label efficiency sprawdza, jak wynik probe zmienia się przy ograniczonej liczbie etykiet treningowych, np. `1%`, `3%`, `10%`, `30%`, `100%`.

Dobra metoda SSL powinna szczególnie pomagać przy małej liczbie etykiet. Jeśli `random` zachowuje się podobnie lub lepiej, oznacza to, że pretraining nie daje dużej przewagi w tym zadaniu.

## Porównywalność z finalnymi VAE/RSSM

Główna ewaluacja z notebooka 02 jest częściowo porównywalna z finalnymi VAE/RSSM, bo używa:

- tego samego datasetu,
- tego samego targetu 5-klasowego,
- tego samego splitu `seen` / `unseen_mcts`,
- analogicznego protokołu frozen encoder + downstream classifier,
- tych samych `CARD_FEATS`,
- tych samych metryk `macro-F1` i F1 per klasa.

Różnice:

- architektura to Transformer, a nie VAE-LSTM/RSSM,
- objective'y to InfoNCE, Barlow Twins i MAE, a nie rekonstrukcja + KL w modelu wariacyjnym,
- notebook 02 był eksperymentem dodatkowym, zwykle z mniejszym budżetem i single seed,
- finalne VAE/RSSM przechodzą przez `src.search` i `src.train_final`, czyli selekcję konfiguracji na VAL i finalną ewaluację po kilku seedach.

Najbezpieczniejsza interpretacja:

> Notebook 02 pokazuje dodatkowe metody uczenia reprezentacji na Transformerze, ocenione na tym samym downstream tasku. Wyniki są użyteczne jako punkt odniesienia, ale nie powinny być mieszane z finalnym rankingiem VAE/RSSM bez zastrzeżeń dotyczących innej architektury, innego budżetu treningu i innej procedury selekcji hiperparametrów.

## Intuicja porównania metod

- `raw` mówi, ile daje sam ręczny wektor cech bez encodera.
- `random` mówi, czy losowa projekcja Transformera jest już wystarczająco silna.
- `infonce` sprawdza, czy predykcja przyszłości uczy użytecznej dynamiki.
- `barlow` sprawdza, czy inwariancja na augmentacje daje stabilną reprezentację.
- `mae` sprawdza, czy rekonstrukcja zamaskowanych kroków uczy struktury stanu gry.
- `supervised` pokazuje górną granicę architektury Transformer przy dostępie do etykiet.
