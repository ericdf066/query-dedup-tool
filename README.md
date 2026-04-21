# Query Dedup Tool

A practical tool for **query deduplication and similarity grouping**, designed for dataset cleaning, benchmark construction, search-log consolidation, and question pool management.

It supports:

- **Exact deduplication**
- **Near-text deduplication** with **SimHash**
- **Local semantic-like grouping** with **TF-IDF + cosine similarity + clustering**
- **GUI-based usage** for non-technical users
- **Traceable outputs** for audit and manual review

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
Uses **SimHash** to detect highly similar short-text variations, such as:

- minor rewrites
- local token insertions/deletions
- formatting-preserved near duplicates

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

## Screenshot

![Query Dedup Tool GUI](gui-home.png)
