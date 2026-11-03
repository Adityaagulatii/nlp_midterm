# Same-Prompt Test

Every model is primed with the **same real line** from the dataset and asked to generate the next 220 characters, with the same random seed.

**Prompt (the >> marks where the model takes over):**

```
First Citizen:
Before we proceed any further, >>
```

Ground truth: in the dataset this line actually continues `hear me speak.`

## Bigram

```
First Citizen:
Before we proceed any further, fu ay IIn eall Clloimeabs bind mechithisur ig wen. n;
Ag's,
Swant
Nurninorak' ht ste ithunghe You l tou wim thethesan bur sththor ther m a ile, ans f wo in, w ave NGon tathtll d amarid ld



Whil be wnsoft I'eeck y y mof
```

## LSTM

```
First Citizen:
Before we proceed any further, our cheek earth?

Pire:
Ay, by me?

HASTINGS:
Fellow; he's rewards controud, for sister.

MENENIUS:
This is the shear but stopp;
Which be silver high woes upon what knots.

STANLEY:
My lord!

Citizens:
Matter, consort.


```

## Transformer

```
First Citizen:
Before we proceed any further, full well.
Gentlemen, God's brother! how is giner?

NORTHUMBERLAND:
And read for steward, the branch's death!

LADY CAPULET:
How now?

LEONTES:
In good sin, what death nature?

MERCUTIO:
Was this be?

JULIET:
If it frien
```
