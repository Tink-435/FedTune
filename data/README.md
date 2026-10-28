# Bundled datasets

The three public datasets named in the SRS (Section 4.3) and Design Document (Section 2.2.1),
stored unmodified in their original published form. `fedtune/data/loader.py` reads them from here;
no network is needed.

| Folder | Dataset | Files | Examples | Task | Partition attribute |
|---|---|---|---|---|---|
| `ag_news/` | AG News (Zhang, Zhao & LeCun, 2015) | `train.csv`, `test.csv`, `classes.txt`, `readme.txt` | 120,000 + 7,600 | 4 topics: World, Sports, Business, Sci/Tech | topic (= label) |
| `mnli/` | MultiNLI 1.0 (Williams, Nangia & Bowman, 2018) | `train.parquet`, `validation_matched.parquet`, `validation_mismatched.parquet` | 392,702 + 9,815 + 9,832 | entailment / neutral / contradiction | genre (10 genres) |
| `xnli/` | XNLI 1.0 (Conneau et al., 2018) | `validation.parquet`, `test.parquet` | 2,490 + 5,010 pairs × 15 languages = 112,500 | entailment / neutral / contradiction | language (15) |

Sources:
* AG News CSV — https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv (the original `ag_news_csv` release; label column is 1–4)
* MNLI — https://huggingface.co/datasets/nyu-mll/multi_nli (Parquet conversion of the official release; `label` 0/1/2, `-1` rows dropped by the loader)
* XNLI — https://huggingface.co/datasets/facebook/xnli, config `all_languages` (each row holds all 15 translations; the loader expands one example per language)

Licences: AG News is provided for non-commercial research use; MultiNLI is mostly OANC (free) with some
CC BY-SA 3.0 genres; XNLI is CC BY-NC 4.0. Cite the original papers when reporting results.

Your own data: put a `.csv` or `.jsonl` file here and set `data.dataset: data/<file>`,
`data.text_fields`, `data.label_field` and (for domain/language skew) `data.attribute_field`.

## Repository note

Only `ag_news/` is tracked in git. `mnli/` and `xnli/` are listed in `.gitignore`: `mnli/train.parquet`
(214 MB) is above GitHub's 100 MB file limit. To run `configs/mnli_domain.yaml` or
`configs/xnli_language.yaml`, download the files from the sources above (or copy the folders from the
original project bundle) into `data/`. The bundled-dataset test for those two datasets fails until
they are present.
