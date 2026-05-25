# MS-SCOPE-Gen

Code for my Master Thesis: "Conformal Prediction for _de novo_ Metabolomics with Structural Similarity Guarantees"

## Summary

This repository implements a **conformal prediction** framework for **_de novo_ structure elucidation from MS/MS spectra**. MS-SCOPE-Gen predicts variable-size molecular structure sets from MS/MS spectra, ensuring (with user-specified probability) that at least one returned candidate is *structurally similar* to the true compound. Structural similarity is defined using Tanimoto similarity or MCES distance.

The codebase adapts:

- **[MIST](https://github.com/samgoldman97/mist)** (Goldman et al.): molecular fingerprint prediction from MS/MS spectra.
- **[MS-BART](https://github.com/OpenDFM/MS-BART)** (Han et al.): molecular string generation conditioned on ECFP4 fingerprints.
- **[SCOPE-gen](https://github.com/rudolfwilliam/scope-gen)** (Kladny et al.): conformal prediction for generative models, adapted here for _de novo_ metabolomics.

Experiments use data from the [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) benchmark (Bushuiev et al.), with evaluation on three different data split scenarios of increasing difficulty.

---

## Installation

```bash
conda env create -f environment.yml
conda activate ms-scope-gen
```

## Running the Pipeline

**Step 1: MS-BART workflow**

Run the following scripts in order from `MS-BART/scripts`:

1. **Data collection**  
   ```
   bash scripts/0_collect_data.sh
   ```
   Download the data from the MassSpecGym benchmark paper, including the MIST checkpoint, preprocessed spectra, spectra labels, and pretraining molecule data.

3. **Train MIST**
   ```
   bash scripts/1_train_mist.sh
   ```  
   Train the MIST model three separate times, once for each data split.

5. **Preprocess BART Data**  
   ```
   bash scripts/2_preprocess.sh
   ```  
   Prepare for BART training, generate and binarize fingerprints three separate times, once for each trained MIST.

7. **Pre-train BART**  
   ```
   bash scripts/3_pretrain.sh
   ```
   Pre-train the BART model for SMILES generation using 4M fingerprint–molecule pairs.

9. **Fine-tune BART**  
   ```
   bash scripts/4_finetune.sh
   ``` 
   Fine-tune BART three separate times, once for each data split.

11. **Evaluate the models**  
    ```
    bash scripts/5_eval.sh
    ```
    Generate candidate SMILES from binarized fingerprints using beam search (width 100).

**Step 2: SCOPE-gen conformal prediction**

Run the following scripts in order from `scope_gen/de_novo_molecules/scripts`:

1. **Format data for conformal prediction**
   ```
   python format_data.py
   ```
   Prepare generated candidate quality and calculate sample admissibility based on structural similarity.

2. **Run conformal prediction**
   ```
   python eval_all.py
   ```
   Apply SCOPE-gen across a wide range of settings using `configs/eval.jsonl`.

## Evaluation and Visualization

Evaluation results and visualizations can be reproduced in the notebook `visualization.ipynb`.
