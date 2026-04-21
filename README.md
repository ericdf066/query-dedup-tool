# Query Dedup Tool

A practical tool for **query deduplication and similarity grouping**, designed for dataset cleaning, benchmark construction, search-log consolidation, and question pool management.

It supports:

- **Exact deduplication**
- **Near-text deduplication** with **SimHash**
- **Local semantic-like grouping** with **TF-IDF + cosine similarity + clustering**
- **GUI-based usage** for non-technical users
- **Traceable outputs** for audit and manual review

---

## Screenshot

![Query Dedup Tool GUI](query去重工具.png)

---

## Why this project

In real-world query datasets, duplicates usually appear in multiple forms:

- exactly repeated queries
- formatting variants
- slight rewrites
- locally similar short texts
- semantically adjacent expressions

This tool is built to turn a noisy query set into a cleaner, more compact, and more reusable dataset.

Instead of relying on heavy remote embedding services, this project adopts a **lightweight and deployable local pipeline**, making it easier to run, share, and reproduce.

---

## Features

### 1. Text normalization
Standardizes queries before deduplication:

- lowercase conversion
- trimming spaces
- collapsing repeated spaces
- removing common punctuation

### 2. Exact deduplication
Removes:

- identical queries
- normalization-equivalent queries

### 3. Near-text deduplication
Uses **SimHash** to detect highly similar short-text variations.

### 4. Local semantic-like grouping
Uses:

- **TF-IDF**
- **cosine similarity**
- **clustering / connected grouping**

to merge queries that are not text-identical but still close in expression.

### 5. GUI support
The tool includes a local GUI so that non-engineering users can:

- select CSV files
- specify the query column
- configure parameters
- choose output directories
- run the full pipeline with one click

---

## Use cases

This project is useful for:

- query dataset cleaning
- benchmark construction
- WebDev / product requirement dataset preparation
- search log consolidation
- FAQ / issue pool cleanup
- preprocessing before intent analysis or clustering

---

## Method overview

The full pipeline is:

```text
Input
→ Normalization
→ Exact deduplication
→ Near-text deduplication (SimHash)
→ Local semantic grouping (TF-IDF)
→ Result export
```

### Layer-by-layer intuition

#### Input
Raw query sets may contain duplicates, formatting variants, and similar expressions.

#### Normalization
Examples:

- `How to learn Python?` → `how to learn python`
- ` 北京天气怎么样！ ` → `北京天气怎么样`

#### Exact deduplication
Examples:

- `北京天气怎么样`
- `北京天气怎么样！`
- ` 北京天气怎么样 `

After normalization, they collapse into one representative query:

- `北京天气怎么样`

#### Near-text deduplication
Examples:

- `北京天气怎么样`
- `北京今天天气怎么样`

These are not identical, but they are textually very close and may be grouped together.

#### Local semantic-like grouping
Examples:

- `怎么学习 Python`
- `Python 入门怎么学`
- `如何快速掌握 Python`

These can be grouped under one semantic representative query.

---

## Project structure

```text
query-dedup-tool/
├── Query Dedup Tfidf Gui Tool.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
└── query去重工具.png
```

---

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy scikit-learn
```

---

## Quick start

Run the GUI tool:

```bash
python3.12 "Query Dedup Tfidf Gui Tool.py"
```

Then:

1. Select your CSV file
2. Enter the query column name
3. Choose an output folder
4. Set text-dedup parameters
5. Optionally enable local semantic grouping
6. Click **Start**

---

## Input format

The input file should be a CSV containing one query column.

Example:

```csv
query
北京天气怎么样
北京天气怎么样！
北京今天天气怎么样
怎么学习 Python
Python 入门怎么学
如何快速掌握 Python
```

---

## Output files

### `deduped_queries.csv`
Representative queries after text-level deduplication.

### `query_groups.csv`
Mapping from original queries to text-level representative queries.

### `semantic_deduped_queries.csv`
Representative queries after local semantic grouping.

### `semantic_query_groups.csv`
Mapping from representative queries to semantic groups.

---

## Design principles

This project is built around the following priorities:

- **Practicality over complexity**
- **Local deployability**
- **Reproducibility**
- **Interpretability**
- **Low usage threshold**

The goal is not to build the strongest semantic system possible, but to build a **usable and shareable query cleaning tool**.

---

## Limitations

This tool is designed for **short-text query processing**, not for deep semantic reasoning over long documents.

Current semantic grouping is based on **local TF-IDF similarity**, which is lightweight and stable, but not equivalent to large-scale embedding-based semantic understanding.

So this project is best viewed as:

- a **data cleaning tool**
- a **query consolidation tool**
- a **benchmark preprocessing tool**

rather than a full semantic understanding engine.

---

## Recommended scenarios

Recommended:

- benchmark dataset cleaning
- WebDev task set construction
- query pool compression
- short-text similarity consolidation
- pre-labeling cleanup

Not recommended as a direct replacement for:

- deep semantic retrieval systems
- intent classification systems
- large-model semantic equivalence judgment
- long-document clustering

---

## Future work

Potential next steps:

- optional embedding-based semantic deduplication
- CLI packaging
- batch folder support
- report generation
- dataset quality analytics dashboard
- service/API deployment mode

---

## Contributing

Issues and pull requests are welcome.

---

## License

MIT License
