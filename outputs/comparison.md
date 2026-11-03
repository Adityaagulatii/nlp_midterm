# Three-Model Comparison

Task: character-level next-character prediction on Tiny Shakespeare (vocabulary = 65 characters).

Random-guessing baseline loss = ln(65) = **4.17**. Every model should beat this; how far below it they get is the story.

## Results table

| Model | Train Loss | Validation Loss | Training Time | Throughput (tok/s) | Peak GPU Mem | Parameters | Generated Text Quality |
|---|---|---|---|---|---|---|---|
| Bigram | 2.4581 | 2.4892 | 26.1s | 48950 | 0 MB | 4,225 | Low |
| LSTM | 1.2774 | 1.5239 | 101.7s | 120858 | 128 MB | 946,625 | Medium |
| Transformer | 1.2091 | 1.5095 | 750.0s | 54612 | 901 MB | 2,689,985 | High |

## Loss comparison

- **Bigram**: val loss 2.4892 (1.69 below the 4.17 baseline).
- **LSTM**: val loss 1.5239 (2.65 below the 4.17 baseline).
- **Transformer**: val loss 1.5095 (2.66 below the 4.17 baseline).

Best (lowest validation loss): **Transformer**. Fastest to train: **Bigram**.

## Generated text samples

### Bigram

```
CExthy brid owindakis by bth

HAPet bobe d e.
S:
O:
IS:
Falatanss:
Wanthar u qur, vet?
F dilasoate awice my.

Hastarom oroup
Yowhthetof isth ble mil ndill, ath ireeesenghin lat Heriliovets, and Win nghirileranousel lind me l.
MAshe ce hiry:
Supr aisspllw y.
Hentous n Boopetelaves
MPOLI s, d mothakleo Windo whth eisbyo wie m dourive we higend t so mower; te

AN ad nterupirf s ar igr t m:

Thiny aleronth,
Mad
RD:

WISo myrangoube!
KENoby ak
Sadsal thes ghesthidin couk ay aney RDUSts I fr t ce.
J
```

### LSTM

```
Clown:
Read cowful, O lo, being it was been evaligence?

First Shepard:
The usure her by their caues with my father comzong hence,
Fromful thy covering will, even well, I intend Hereful by Saint
Will you go want well: now, all thus, court:
These baise have denevingure Bona tell
Is not the odligenter of your whit evils,
Some of this sweet precespion of his butterandness,
Which armiored my mother fled at this Prince, Somerry
To be aslied in a rosely, is a formining stray.

ANGELO:
The hand for his
```

### Transformer

```
WARWICK:
Title, March tills he with so trial,
And take a got of the cage noble hour.
Away, adily, futt shards o' the dark.

QUEEN ELIZABETH:
And all these interchanges whiles breath shall
seemst cort of a sin.

FRORIZUS:
Ever execution would I achieful affect wind.

Fourth Ghost time, Aumerle's stay.

MISTRESSS:
O is hand!

LUCIO:
O, as it that yet to follow the kind mock'd,
It bushin it relignly to dish though the light,
Pursuius my bust of blooding.

Citizens:
Too, brother!

Clown:
Mords: or 
```

## Observations

**Which model performed best?** The Transformer achieved the lowest validation loss (1.5095), so it models the text best. Loss falls monotonically as the models gain access to more context: the bigram sees one character, the LSTM compresses the whole prefix into a hidden state, and the transformer attends directly over its full window.

**Which model trained fastest?** The Bigram trained fastest (26.1s). The bigram is cheapest per step (a single table lookup) but learns almost nothing; the transformer costs the most compute per step because attention and the feed-forward sub-layers do far more work.

**Which model generated the most realistic text?** The transformer. The bigram produces plausible letter pairs but no real words; the LSTM produces real words and keeps the SPEAKER:/line structure; the transformer additionally holds longer-range coherence — balanced dialogue turns and more consistent sentence shape.

**What did the transformer improve over the other models?** It lowered validation loss from 2.4892 (bigram) and 1.5239 (LSTM) to 1.5095. The gain comes from self-attention: any position can look directly at any earlier position in one step, instead of relying on a one-character window (bigram) or a single fixed-size memory vector squeezed through many sequential updates (LSTM).

**Was the improvement worth the added complexity?** On this dataset, yes. The transformer has more parameters (2,689,985 vs the LSTM's 946,625) and takes longer to train (750.0s vs 101.7s), but it reaches a lower loss and more coherent samples. For a tiny corpus the margin is modest; the architecture's real advantage grows with more data and longer context.

**What did I learn about attention?** Attention replaces sequential memory with direct, content-based lookup. Each position computes how relevant every other position is and reads a weighted mixture of them, so long-range dependencies no longer have to survive a long chain of recurrent updates. Combined with positional encoding (attention alone is order-blind), this is what lets the transformer beat the LSTM at the very same task.