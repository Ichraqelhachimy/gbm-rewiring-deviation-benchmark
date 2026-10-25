# Single-sample co-expression networks in IDH-wildtype glioblastoma

Code for the manuscript:

> El Hachimy I, Benamar N. *Reference-Based Single-Sample Co-expression Networks Largely Encode Expression Deviation: A Leakage-Controlled Benchmark in IDH-Wildtype Glioblastoma.* Submitted.

## Contents

| Notebook | What it does | Paper section |
|---|---|---|
| `01_cohort_assembly.ipynb` | TCGA-GBM IDH-wildtype cohort (144 patients), one RNA-seq aliquot per patient, sample QC, recount3 and GTEx metadata, GTEx cortical controls (334) | 3.1–3.2 |
| `02_stageA_B_anchors_network.ipynb` | Stage A: differential expression (PyDESeq2, 60 tumours vs 334 GTEx) and disease anchors; Stage B: STRING-constrained healthy reference network | 3.3–3.4 |
| `03_stageC_calibration_rewiring.ipynb` | GDC → recount3 bridge calibration, GTEx reference network, add-one-in single-sample networks, eNESH, density sensitivity, frozen rewiring matrix | 3.4–3.6 |
| `04_stageD_survival.ipynb` | Locked split, nested 3 × 5 cross-validation, M1–M6, out-of-fold stacking, deviation baselines R1/R2/R2b, cohort table, extra metrics, clinical positive control, supplementary tables | 3.7–3.9, 4.1, 4.5–4.6 |
| `05_split_sensitivity.ipynb` | 100 event-stratified 80:20 partitions | 3.9, 4.5 |
| `06_cgga_external.ipynb` | CGGA preparation, gene mapping, location-only harmonization with CGGA normal controls, CGGA networks and eNESH, frozen-model external evaluation | 3.4.3, 4.8–4.9 |
| `07_biomarkers.ipynb` | Single-gene discovery in TCGA development and replication in CGGA | 4.10 |
| `08_figures.ipynb` | Figures 1, 2, 3 and 5 | — |
| `revision/09_revision_analyses.ipynb` + `revision/scripts/` | Closed-form check, matched variance decomposition, reference/density/method sensitivity, selection stability, CGGA statistics, MGMT check | 3.10, 4.3–4.4, 4.7, 4.9 |

## Data

No patient-level data are included. All inputs are public:

- **TCGA-GBM**: Genomic Data Commons, https://portal.gdc.cancer.gov (STAR-Counts, clinical data)
- **TCGA and GTEx in recount3**: https://rna.recount.bio
- **CGGA** mRNAseq_693 and CGGA.normal_20: http://www.cgga.org.cn
- **STRING v12**: https://string-db.org

## How to run

1. Install the packages: `pip install -r requirements.txt` (Python 3.9 or later).
2. Download the public files above into one folder.
3. Set the environment variable `GBM_DATA_DIR` to that folder. If it is not set, the notebooks use `~/Downloads`.
4. Run the notebooks in numerical order. Each notebook writes its outputs into subfolders of the data folder, which the next notebooks read.

Random seeds are fixed: 42 for the locked split and cross-validation (outer folds 42 + 100 × repeat), 0–99 for the alternative partitions.

## License

MIT License (see `LICENSE`).
