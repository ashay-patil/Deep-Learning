# Word Embeddings in Keras — Step-by-Step Explanation

This note explains the code in `word_embedding(1).ipynb` from beginning to end. The goal is to understand how text is converted into integer IDs, how sentences are made the same length, and how an Embedding layer converts each integer into a vector of numbers.

## 1. The complete code from the notebook

```python
from tensorflow.keras.preprocessing.text import one_hot

sentences = [
    'the glass of milk',
    'the glass of juice',
    'the cup of tea',
    'I am a good boy',
    'I am a good developer',
    'understand the meaning of words',
    'your videos are good'
]

# Define the vocabulary size
vocabulary_size = 10000

# One Hot Representation
one_hot_repr = [
    one_hot(sentence, vocabulary_size)
    for sentence in sentences
]
one_hot_repr

from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
import numpy as np

sentence_length = 8
# All sentences in the dataset must be of 8 words

padded_sentences = pad_sequences(
    one_hot_repr,
    padding='pre',
    maxlen=sentence_length
)
padded_sentences

dimension = 10

model = Sequential()
model.add(
    Embedding(
        vocabulary_size,
        dimension,
        input_length=sentence_length
    )
)

model.compile('adam', 'mse')

model.summary()

model.predict(padded_sentences)
```

`numpy` is imported in the notebook, although the shown code does not directly use `np`.

---

## 2. What are we trying to do?

A neural network works with numbers, not raw words such as `"milk"` or `"developer"`. This notebook performs two main conversions:

1. **Text → integer IDs:** each word is represented by an integer.
2. **Integer IDs → vectors:** the Embedding layer looks up a vector of 10 numbers for each integer ID.

The full pipeline is:

```text
Sentences (text)
      |
      v
one_hot()
      |
      v
A list of integer IDs per sentence
      |
      v
pad_sequences()
      |
      v
A rectangular array: 7 sentences × 8 integer positions
      |
      v
Embedding layer
      |
      v
A 3D array: 7 sentences × 8 positions × 10 numbers per position
```

Important distinction: in this notebook, Keras's `one_hot()` function returns **integer indices**. It does not directly return a long binary vector such as `[0, 0, 1, 0, ...]`. Those integer indices are then supplied to the Embedding layer.

## 3. Step 1 — Store the sentences

```python
sentences = [
    'the glass of milk',
    'the glass of juice',
    'the cup of tea',
    'I am a good boy',
    'I am a good developer',
    'understand the meaning of words',
    'your videos are good'
]
```

There are **7 sentences**. Each sentence contains a different number of words:

| Sentence | Number of words |
|---|---:|
| `the glass of milk` | 4 |
| `the glass of juice` | 4 |
| `the cup of tea` | 4 |
| `I am a good boy` | 5 |
| `I am a good developer` | 5 |
| `understand the meaning of words` | 5 |
| `your videos are good` | 4 |

The sentences do not all have the same length. We will fix that later.

## 4. Step 2 — Choose a vocabulary size

```python
vocabulary_size = 10000
```

This sets the size of the range from which `one_hot()` generates integer IDs. The Embedding layer will also be configured to accept indices in this vocabulary range.

Think of the IDs as labels used to look up words. An ID is **not** a measure of a word's meaning. For example, ID `9817` does not mean a word is more important than ID `565`.

The value `10000` is a chosen configuration; it does not mean the notebook contains 10,000 unique words. Also, because `one_hot()` uses hashing, different words can potentially map to the same index (a hash collision).

## 5. Step 3 — Convert each sentence to integer IDs

```python
one_hot_repr = [
    one_hot(sentence, vocabulary_size)
    for sentence in sentences
]
```

Let's break down the expression:

- `for sentence in sentences` visits the sentences one by one.
- `one_hot(sentence, vocabulary_size)` converts the words in that sentence into integer IDs.
- The outer square brackets collect the results into a list.

The notebook produced this output:

```python
[
    [9817, 3170, 565, 6200],
    [9817, 3170, 565, 5932],
    [9817, 7906, 565, 4419],
    [3195, 1986, 1701, 5486, 2368],
    [3195, 1986, 1701, 5486, 428],
    [5624, 9817, 9440, 565, 6209],
    [6563, 8500, 1285, 5486]
]
```

These are the integer IDs shown by this notebook's run. Exact IDs may differ across implementations or changes to the tokenizer/hash behavior, so do not memorize the particular numbers.

### Dry run: first sentence

Input:

```text
the glass of milk
```

The notebook's output for this sentence is:

```python
[9817, 3170, 565, 6200]
```

Conceptually, the conversion is:

| Word position | Word | Integer ID in this notebook |
|---:|---|---:|
| 1 | `the` | 9817 |
| 2 | `glass` | 3170 |
| 3 | `of` | 565 |
| 4 | `milk` | 6200 |

So the sentence becomes a list of four integers.

Notice that `the` has the same ID `9817` in other sentences in this output. Likewise, `of` has ID `565` wherever it appears. The integer IDs allow the model to identify the token positions; the IDs themselves do not yet contain a learned semantic representation.

### Is this true One-Hot Encoding?

In the strict mathematical sense, a one-hot vector has one `1` and all other entries `0`, for example:

```text
ID 2 in a vocabulary of 5 -> [0, 0, 1, 0, 0]
```

However, `tensorflow.keras.preprocessing.text.one_hot()` returns an **integer index for each token**, not that full binary vector. Its name can be confusing. In this notebook, those indices are useful because an Embedding layer expects integer indices.

## 6. Step 4 — Set a fixed sentence length

```python
sentence_length = 8
```

The notebook chooses to represent every sentence using exactly **8 integer positions**.

Why? Neural-network batches are easier to process when all sequences in a batch have the same length. At the moment, the sentences have 4 or 5 words, so we need padding.

The chosen length `8` is a configuration decision. It is not automatically inferred from the data.

## 7. Step 5 — Pad the sequences

```python
padded_sentences = pad_sequences(
    one_hot_repr,
    padding='pre',
    maxlen=sentence_length
)
```

Here:

- `one_hot_repr` is the list of integer sequences.
- `maxlen=8` requests a final sequence length of 8.
- `padding='pre'` adds padding zeros **at the beginning** of shorter sequences.

### Dry run: first sentence

Before padding:

```python
[9817, 3170, 565, 6200]
```

It has 4 numbers. We need 8, so we add `8 - 4 = 4` zeros at the beginning:

```python
[0, 0, 0, 0, 9817, 3170, 565, 6200]
```

For a sentence with 5 numbers, we add `8 - 5 = 3` zeros.

The notebook's complete padded array is:

```python
array([
    [   0,    0,    0,    0, 9817, 3170,  565, 6200],
    [   0,    0,    0,    0, 9817, 3170,  565, 5932],
    [   0,    0,    0,    0, 9817, 7906,  565, 4419],
    [   0,    0,    0, 3195, 1986, 1701, 5486, 2368],
    [   0,    0,    0, 3195, 1986, 1701, 5486,  428],
    [   0,    0,    0, 5624, 9817, 9440,  565, 6209],
    [   0,    0,    0,    0, 6563, 8500, 1285, 5486]
], dtype=int32)
```

The array has **7 rows and 8 columns**:

- 7 rows because there are 7 sentences.
- 8 columns because each sentence has been made 8 positions long.

Therefore, its shape is:

```text
(7, 8)
```

### What do the padding zeros mean?

The zeros are placeholders added to make the sequence lengths equal. They are not words from the original sentences.

One detail to remember: an Embedding layer can still look up a vector for index `0`. In this notebook, `mask_zero=True` was **not** set, so do not assume that the Embedding layer automatically ignores the padding positions. If you want the model to mask padding tokens, you can configure an embedding layer with `mask_zero=True` (and ensure the rest of the model supports masking).

## 8. Step 6 — Choose the embedding dimension

```python
dimension = 10
```

This means that each integer ID will be represented by a vector containing **10 numbers**.

For example, an embedding vector might look like this:

```text
[0.12, -0.03, 0.41, 0.08, -0.22,
 0.17, 0.05, -0.11, 0.30, 0.02]
```

This is only an illustrative vector, not a value copied from the notebook. The notebook initializes its own values.

The two important settings are now:

- `vocabulary_size = 10000`: the embedding table has 10,000 possible rows.
- `dimension = 10`: each row contains 10 numbers.

## 9. Step 7 — Create the Embedding layer

```python
model = Sequential()
model.add(
    Embedding(
        vocabulary_size,
        dimension,
        input_length=sentence_length
    )
)
```

`Sequential()` creates a model in which layers are arranged in order. The notebook adds one layer: `Embedding`.

### What does an Embedding layer do?

Imagine a table with 10,000 rows and 10 columns:

```text
                  10 numbers per row
             <------------------------->
ID 0       [ 0.01, -0.02, ... 10 values ]
ID 1       [ 0.03,  0.04, ... 10 values ]
ID 2       [-0.01,  0.05, ... 10 values ]
...
ID 9817    [ 10 numbers associated with ID 9817 ]
...
ID 9999    [ 10 numbers associated with ID 9999 ]
```

This table is the embedding matrix. Its shape is:

```text
(10000, 10)
```

When the input contains the integer `9817`, the Embedding layer retrieves row `9817` from the table. It returns the 10 numbers in that row.

It does **not** need to create a 10,000-element one-hot vector and multiply it by the matrix explicitly. It can perform the equivalent operation by looking up the row directly.

### Dry run: one padded sentence

Input:

```python
[0, 0, 0, 0, 9817, 3170, 565, 6200]
```

The Embedding layer looks up one row for each of the 8 integer positions:

| Input integer | What the layer does | Number of output values |
|---:|---|---:|
| 0 | Look up embedding row 0 | 10 |
| 0 | Look up embedding row 0 again | 10 |
| 0 | Look up embedding row 0 again | 10 |
| 0 | Look up embedding row 0 again | 10 |
| 9817 | Look up embedding row 9817 | 10 |
| 3170 | Look up embedding row 3170 | 10 |
| 565 | Look up embedding row 565 | 10 |
| 6200 | Look up embedding row 6200 | 10 |

Eight positions × ten values per position gives:

```text
8 × 10 = 80 numbers for one sentence
```

The output shape for one sentence, with a batch dimension included, is `(1, 8, 10)`. The output for the full set of 7 sentences is `(7, 8, 10)`.

## 10. Step 8 — Compile the model

```python
model.compile('adam', 'mse')
```

This is shorthand for specifying:

```python
model.compile(
    optimizer='adam',
    loss='mse'
)
```

- **Optimizer (`adam`)**: the algorithm Keras would use to update trainable parameters during training.
- **Loss (`mse`)**: mean squared error, a loss function that measures squared differences between predictions and target values.

The notebook does not show target values for a training task, and it does not call `model.fit()`. Therefore, this code **does not train the embeddings**. Compiling a model configures training; it does not itself perform training.

Also, `mse` is not automatically the correct loss for every text or NLP task. The appropriate loss depends on what the model is supposed to predict.

## 11. Step 9 — View the model summary

```python
model.summary()
```

The notebook's saved output showed the Embedding layer as **unbuilt**, with an unknown output shape and zero parameters:

```text
embedding (Embedding)    Output Shape: ?    Param #: 0 (unbuilt)
Total params: 0
```

That happens because, in the notebook's Keras version, the layer had not yet been built when `summary()` was called. The later `model.predict(padded_sentences)` call supplies input data and builds the layer.

Once the layer is built, the expected dimensions are:

```text
Input shape:  (None, 8)
Output shape: (None, 8, 10)
Parameters:   10000 × 10 = 100000
```

Here, `None` means the number of sentences in a batch is variable. It does not mean the model has zero sentences.

The embedding table contains 100,000 trainable scalar values because there are 10,000 rows and 10 values in each row:

```text
10000 × 10 = 100000 parameters
```

The `input_length` argument is deprecated in some recent Keras versions, as the warning in the notebook indicates. In current code, it is usually fine to omit `input_length` and let the model infer the sequence length from its input.

## 12. Step 10 — Run `predict()`

```python
model.predict(padded_sentences)
```

This sends the 7 padded sequences through the Embedding layer and returns their embedding vectors.

Input shape:

```text
(7, 8)
```

Output shape:

```text
(7, 8, 10)
```

Read the output dimensions from left to right:

1. `7` — number of sentences.
2. `8` — integer positions in each padded sentence.
3. `10` — numbers in the embedding vector for each position.

So the output is like a stack of 7 matrices, each of shape `(8, 10)`.

### Why are the output values small decimals?

The Embedding layer starts with initialized numerical weights. Since this notebook has not trained the model, the vectors are still initial values rather than learned word representations. The exact decimals can change between runs because the initialization may be random.

### What happens to repeated IDs?

The same integer ID looks up the same row in the embedding table during a given model state. For example, every occurrence of ID `9817` uses the same 10-value vector. If the embedding is trained later, that row can be updated by training.

Likewise, each padding `0` looks up row `0` in this notebook because masking has not been enabled. Consequently, the vectors at padding positions are not guaranteed to be all zeros.

## 13. A compact shape dry run

Using the first sentence:

```text
Text:
"the glass of milk"

After one_hot():
[9817, 3170, 565, 6200]                shape conceptually: (4,)

After padding to length 8:
[0, 0, 0, 0, 9817, 3170, 565, 6200]   shape: (8,)

After adding it as a batch of one:
shape: (1, 8)

After Embedding(dimension=10):
shape: (1, 8, 10)
```

For the complete dataset:

```text
7 sentences
    ↓
7 sequences of 8 integer IDs
    ↓
Input shape:  (7, 8)
    ↓
Embedding layer: 10 values for every position
    ↓
Output shape: (7, 8, 10)
```

## 14. Important concepts to remember

| Concept | Meaning in this notebook |
|---|---|
| `one_hot()` | Converts words to integer IDs; it does not return full binary one-hot vectors here. |
| `vocabulary_size=10000` | Configured size of the index range / embedding table. |
| `pad_sequences()` | Makes sequences the same length by adding zeros. |
| `padding='pre'` | Adds zeros at the beginning of shorter sequences. |
| `sentence_length=8` | Every input sequence is represented by 8 positions. |
| `dimension=10` | Each token ID maps to 10 numerical values. |
| `Embedding(10000, 10)` | A trainable lookup table with shape `(10000, 10)`. |
| `model.compile()` | Configures the optimizer and loss; it does not train by itself. |
| `model.predict()` | Runs the input through the current model and returns its output. |
| `model.fit()` | Would perform training if suitable training data and targets were supplied. |
| Output `(7, 8, 10)` | 7 sentences, 8 positions per sentence, 10 numbers per position. |

## 15. What is missing if the goal is to learn meaningful embeddings?

The notebook demonstrates the **mechanics** of converting text into embedding vectors. It does not yet teach the vectors meaningful relationships, because no training step is shown.

To learn useful embeddings, a complete task generally needs:
1. A defined prediction objective.
2. Training inputs and corresponding targets.
3. A suitable loss function.
4. A training call such as `model.fit(...)`.
5. Enough relevant data and evaluation to determine whether the learned representations are useful.

The exact training code cannot be determined from this notebook alone because it does not define a prediction task or provide target labels.

## 16. Quick self-test

**Question 1:** The padded input has shape `(7, 8)` and the embedding dimension is `10`. What is the output shape?

**Answer:** `(7, 8, 10)`.

**Question 2:** Does `one_hot()` in this notebook directly return a binary vector with 10,000 elements for each word?

**Answer:** No. It returns an integer ID for each token.

**Question 3:** Does `model.compile('adam', 'mse')` train the model?

**Answer:** No. It configures the optimizer and loss. Training would require a call such as `model.fit(...)` with appropriate data and targets.

**Question 4:** Why are zeros added to the beginning of `the glass of milk`?

**Answer:** The sequence has 4 token IDs, but the configured length is 8. `padding='pre'` adds four zeros at the beginning.

**Question 5:** Does an embedding vector automatically represent a word's meaning correctly as soon as the layer is created?

**Answer:** No. The initial values are not yet trained for a task. Meaningful patterns must be learned through an appropriate training objective and data.
