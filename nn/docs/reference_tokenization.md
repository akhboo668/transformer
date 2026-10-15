# Reference: Tokenization and Vocabularies

> This is a supplementary reference document. Not part of the main 10-step build,
> but covers important context about how real-world models build their vocabularies.

[< Back to Overview](00_overview.md) | [Back to Embedding (Step 1f)](01_math_primitives.md)

---

## The Problem

Models work with numbers, but input is text. We need to convert text into a sequence
of integers (token IDs). The mapping from text to IDs is called the **vocabulary**.

## Why Not Characters or Words?

```
Character-level:  "unhappiness" --> ['u','n','h','a','p','p','i','n','e','s','s']
  Problem: 11 tokens for one word. Sequences very long. Attention is O(n^2) -- slow.

Word-level:       "unhappiness" --> ['unhappiness']
  Problem: what about "unhappier"? Need separate entries for every word form.
  Vocabulary explodes to millions. Rare words get no representation.

Subword-level:    "unhappiness" --> ['un', 'happi', 'ness']
  Sweet spot: small vocabulary (~50K) covers everything. Common words stay whole,
  rare words split into meaningful pieces.
```

## Tokenization Algorithms

### BPE (Byte Pair Encoding) -- used by GPT-2, GPT-3, GPT-4

```
Start with: individual characters as vocabulary
            {'a', 'b', 'c', ..., 'z', ' ', '.', ...}

Repeat:
  1. Count all adjacent pairs in training data
  2. Find the most frequent pair
  3. Merge into a new token, add to vocabulary
  4. Stop when vocab reaches desired size (e.g., 50,257)

Example merges:
  Step 1: ('t', 'h')  --> 'th'      (most frequent pair)
  Step 2: ('th', 'e') --> 'the'
  Step 3: ('i', 'n')  --> 'in'
  Step 4: ('in', 'g') --> 'ing'
  ... 50,000 merges later: vocabulary of 50,257 tokens
```

Result:

```
"the"            --> ["the"]                    (1 token -- very common)
"transformer"    --> ["transform", "er"]        (2 tokens)
"unhappiness"    --> ["un", "happi", "ness"]    (3 tokens)
"defenestration" --> ["def", "en", "est", "ration"]  (4 tokens -- rare)
```

### WordPiece -- used by BERT, original Transformer

Similar to BPE but uses likelihood-based merge criterion (maximizes probability of
training data). Uses ## prefix for continuation tokens.

```
"playing"  --> ["play", "##ing"]
"unplayed" --> ["un", "##play", "##ed"]
```

### SentencePiece -- used by LLaMA, T5

Works on raw text without pre-tokenization. Language-agnostic.

```
"I like cats"  --> ["▁I", "▁like", "▁cats"]     (▁ = space is part of token)
"猫が好き"      --> ["猫", "が", "好き"]            (works for any language)
```

### Comparison

```
Algorithm      Used by                      Special feature
─────────      ───────                      ───────────────
BPE            GPT-2, GPT-3, GPT-4         Frequency-based merging
WordPiece      BERT, Original Transformer   Likelihood-based merging
SentencePiece  LLaMA, T5, mBART            Language-agnostic, no pre-tokenization
```

## Training Data Sources (GPT-4 scale)

```
Source                     Example content                    ~Share
──────                     ───────────────                    ──────
Web crawl (CommonCrawl)    Billions of web pages              ~60%
Books                      Fiction, non-fiction, textbooks     ~15%
Academic papers            ArXiv, PubMed                      ~5%
Code                       GitHub repositories                ~5%
Wikipedia                  All languages                      ~3%
Curated datasets           StackOverflow, Reddit, forums      ~10%
Specialized data           Legal, medical, financial          ~2%
```

The vocabulary is built FROM this data -- tokens appearing frequently earn their own IDs.

## What a Typical 50K Vocabulary Contains

```
Type                 Examples                          ~Count
────                 ────────                          ──────
Single characters    a, b, c, 0, 1, !, @              ~256
Common words         the, is, and, of, to             ~5,000
Common subwords      ing, tion, pre, un, ed            ~10,000
Less common words    transformer, algorithm             ~15,000
Partial words        trans, form, algo                  ~15,000
Code tokens          def, return, {, }, ==              ~5,000
Special tokens       <PAD>, <EOS>, <UNK>               ~5-10
```

## The Byte-Level Trick (GPT-2+)

GPT-2 starts with all 256 bytes as base vocabulary. This means every possible text can
be tokenized -- no <UNK> (unknown) token needed.

```
"🎉" --> [tokens for its UTF-8 bytes: 240, 159, 142, 137]
```

## Example: Tokenizing Python Code

How a Python-trained model tokenizes code:

```python
def calculate_total(items):
    total = 0
    for item in items:
        if item.price > 0:
            total += item.price * item.quantity
    return total
```

Approximate tokenization:

```
["def", " calculate", "_total", "(", "items", "):", "\n"]
["    ", "total", " =", " 0", "\n"]
["    ", "for", " item", " in", " items", ":\n"]
["        ", "if", " item", ".", "price", " >", " 0", ":\n"]
["            ", "total", " +=", " item", ".", "price", " *", " item", ".", "quantity", "\n"]
["    ", "return", " total"]
```

Key observations:
- **Keywords** ("def", "for", "if", "return"): single tokens
- **Indentation** ("    ", "        "): single tokens (model sees indent level as one concept)
- **Common patterns** ("):", " = ", " += "): often merge into single tokens
- **CamelCase splits**: "ShoppingCart" --> "Shopping", "Cart"
- **snake_case splits**: "user_id" --> "user", "_id"

A well-built vocabulary lets the model reason at the level of programming concepts,
not individual characters. That's why it can generate syntactically correct code.

### Python-Specific Vocabulary Entries

```
Token            Why it becomes a single token
─────            ─────────────────────────────
"def "           Every function definition
"class "         Every class definition
"self."          Appears constantly in methods
"__init__"       Every constructor
"    "           4-space indent (standard Python)
"        "       8-space indent (nested block)
"import "        Top of every file
".append("       Common list operation
"range("         Common in loops
"len("           Very frequent function
"isinstance("    Type checking pattern
"# "             Start of comment
'"""'            Docstring delimiter
"-> "            Return type annotation
```

### Specialized Code Models

```
Model          Vocab size    Special features
─────          ──────────    ────────────────
GPT-4          ~100K         Mixed English + code
CodeLlama       32K          Optimized for code patterns
StarCoder       49K          80+ language tags, fill-in-middle tokens
```

## Vocabulary Size Across Models

```
Model                 Vocab size    Token type
─────                 ──────────    ──────────
Our toy model              13       Hand-defined digits + special
Original Transformer     37,000     WordPiece (BPE variant)
GPT-2                    50,257     BPE
BERT                     30,522     WordPiece
LLaMA                    32,000     SentencePiece
GPT-4                   ~100,000    BPE (byte-level)
```

## The Key Insight

Vocabulary is a **compression scheme**. BPE finds the most efficient encoding:

```
Frequent patterns  --> short (1 token):     "the", "def", "self."
Rare patterns      --> long (many tokens):  "defenestration" = 4 tokens
```

Like Morse code: common letters get short codes (E = "."), rare letters get long codes
(Q = "--.-"). BPE does the same for text.

## Our Project

We skip all this complexity. Our vocabulary is hand-defined:

```
ID    Token    Meaning
──    ─────    ───────
0-9   digits   the numbers to sort
10    PAD      padding
11    SOS      Start Of Sequence
12    EOS      End Of Sequence
```

This lets us focus on the Transformer architecture without tokenization complexity.