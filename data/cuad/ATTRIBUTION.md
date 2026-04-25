# CUAD Attribution

This directory contains data derived from the **Contract Understanding
Atticus Dataset (CUAD)**, released by The Atticus Project under the
Creative Commons Attribution 4.0 International License (CC BY 4.0).

- **Source:** https://github.com/TheAtticusProject/cuad
- **License:** https://creativecommons.org/licenses/by/4.0/
- **Citation:**
  Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021).
  *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review.*
  arXiv preprint arXiv:2103.06268.

## Files in this repository

| File | Origin | License |
|------|--------|---------|
| `CUAD_v1/master_clauses.csv` | Redistributed verbatim from CUAD | CC BY 4.0 |
| `CUAD_v1/full_contract_txt/*.txt` | Redistributed verbatim from CUAD | CC BY 4.0 |
| `../cuad_schema.json` | Our schema design + normalised enums | CC BY 4.0 (inherits from the CUAD labels it references) |
| `../cuad_train.jsonl`, `../cuad_test.jsonl` | Our prompt/target format, embedding CUAD contract text and labels | CC BY 4.0 (inherits from embedded CUAD content) |
| `../../src/prepare_cuad.py` | Our adapter script (contains no CUAD data) | MIT (parent repo license) |

## What CC BY 4.0 requires

- **Attribution** — credit the original authors when redistributing (this
  file fulfils that obligation for the repository).
- **No additional restrictions** — downstream users retain the same
  CC BY 4.0 rights over the CUAD-derived content they would have from
  the original source.

## What CC BY 4.0 permits

- Redistribution, including in modified form.
- Commercial use.
- Use in academic publications, products, and services.

## Note on our derivative contributions

Where we have added original work on top of the CUAD labels — schema
design, enum normalisation heuristics, train/test split strategy,
prompt templates — those contributions are our own. When distributed
alongside CUAD-derived text they inherit CC BY 4.0's attribution
requirement for the combined work. The adapter script itself,
containing no CUAD data, is released under the parent repository's
MIT license.
