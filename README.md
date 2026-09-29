# AMP-generator

1.Introduction
The software is a GRU-based AMP generation software, as shown in Figure 1 is the structure of the software schematic, the software is mainly composed of the following four modules, respectively, the data processing module (data_processing), sequence encoding and decoding module (simple_tokenizer), the generator module (generate_sequences) and the model training module (train).
1.1 Operation and Design
Device Selection: Supports CPU and GPU (CUDA) acceleration.
Batch Processing: Batch loading of data using PyTorch DataLoader.
Cyclic Training: Iterate training over multiple rounds and save the model after each round.
Quality control: Ensure that the generated antimicrobial peptides meet specific biochemical properties.

![image](https://github.com/user-attachments/assets/9070a700-0135-49ec-b2c2-430ddcec256c)


2. Technical characteristics

2.1 Deep learning model
Model architecture: GRU (Gated Recurrent Unit) is used as the core model, and the features of amino acid sequences are automatically extracted through multi-layer neural networks.
Training data: a large-scale dataset of known amino acid sequences is used for training to ensure that the model has good generalisation ability.

2.2 Biological constraints
Net charge: the net charge of the generated sequence needs to be greater than or equal to 3.
Hydrophobicity ratio: the hydrophobic amino acid ratio of the generated sequence should be between 0.4 and 0.6.
Aromatic amino acids: the resulting sequence must contain at least one aromatic amino acid.
Cysteine residues: the number of cysteine residues in the generated sequence must be 0 or 2.

2.3 Logging and Debugging
Logging: Key information during the generation process will be recorded in the log file, including generated sequences, validation results, etc.
Debugging support: Detailed error messages and exception handling mechanisms are provided to facilitate debugging and optimisation by developers.

4. Development Environment
   
3.1 Hardware Requirements
CPU: It is recommended to use a multi-core processor, this software development uses NVIDIA GeForce RTX 3090, driver version 550.90.07, CUDA version 12.4 as the environment for training models.
Memory: At least 4GB
Storage: at least 10GB of available space

3.2 Software Requirements
Operating system: Linux
Python: version 3.7 and above
PyTorch: 1.7 and above
Other dependent libraries: pickle, tqdm, logging

4. Installation and use

4.1 Installation steps
Install Python: Make sure Python 3.7 or above is installed on your system.
Install dependencies: Use pip to install the required dependencies.
pip install -r requirements.txt
And make sure the following files are in the current folder
data_processing.py
generate_sequences_CPU.py
generate_sequences_GPU.py
model.py
processed_data.pkl
tokeniser.pkl
train.py
trained_model.pth
and the folder: tokenizer
The folder tokenizer contains the file simple_tokenizer.py

![image](https://github.com/user-attachments/assets/b6426e9b-2508-42de-b032-e79822d8ebcd)

4.2 Usage
Load the Tokenizer:
tokenizer_path = ‘tokenizer.pkl’
tokeniser = load_tokenizer(tokenizer_path)
Load the model:
model_path = ‘trained_model.pth’
vocab_size = tokenizer.vocab_size
embedding_dim = 128
hidden_dim = 256
model = SimpleModel(vocab_size, embedding_dim, hidden_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
Generate sequences:
num_sequences = 20
min_length = 5
max_length = 10
temperature = 1.0
generated_sequences = generate_sequences(tokenizer, model.state_dict(), device, num_sequences, min_length, max_length, temperature)
Decode the sequences:
decoded_sequences = decode_sequences(generated_sequences, tokenizer)
write_sequences_to_file(decoded_sequences, filename=‘generated_sequences.txt’)

5. Run the example
You can directly use our trained model to run, here we provide both AMP generator using CPU and AMP generator using GPU accelerated computation:
# Run the generator using CPU
#This parameter determines the number of generated sequences num_sequences = 20
#This parameter determines the minimum length of the generated sequences min_length = 5
#This parameter determines the maximum length of the generated sequences max_length = 10
#This parameter determines the randomness of the generated sequences temperature = 1.0
Python generate_sequences_CPU.py
After running you will see the following progress bar showing the current progress and estimated time remaining until 100%.

![image](https://github.com/user-attachments/assets/efe9fa49-3cdb-41c7-b566-3c541bee9382)

![image](https://github.com/user-attachments/assets/0581ff2a-1ac1-4a43-8e1d-16dd957fd26a)

![image](https://github.com/user-attachments/assets/7a59514a-47dc-4154-9f53-1e098f9db321)

#Running the generator on the GPU is much faster, so we recommend that you run it this way!
Python generate_sequences_GPU.py

![image](https://github.com/user-attachments/assets/0f4d11cd-b857-4c9b-8bec-dec62e1581bb)

![image](https://github.com/user-attachments/assets/9f5413b5-98af-4e47-a93e-fb6ab3c81b5d)

6. Sample output
At the end of the process you will see two files: generation_log.txt and generated_sequences.txt.
The log file generation_log.txt will show the detection and generation during the run. If you set the number of generation to be too high, the file may not contain any content to avoid memory overflow and other problems, the content of the file is as follows:

![image](https://github.com/user-attachments/assets/3ed3d216-cfdd-4ff4-ab51-e2e7e3bb495d)

The generated_sequences.txt file has the following contents:

![image](https://github.com/user-attachments/assets/7f9fbf55-aa9e-4863-a945-f078e6834a36)

7. Training the model
If you need to train your own model to use the algorithm we have provided an example of how to do this, first make sure you have installed the environment required in requirements.txt and then prepare the fasta file which contains the AMP sequences in fasta format.
First run data_processing.py to regularise the sequence file and tokenise the sequences (tokenizer)
python data_processing.py

You will get processed_data.pkl and tokenizer.pkl for subsequent processing

![image](https://github.com/user-attachments/assets/67bf97aa-fb59-4c01-b82e-e51d74b96fa3)

![image](https://github.com/user-attachments/assets/53fa1479-cb91-4063-912d-bd77971f2bd3)

Then run train.py to train the model, after that you will see the current training progress and the number of rounds and the change in the Loss value for each round
#This function determines the number of training rounds num_epochs = 15
Python train.py

![image](https://github.com/user-attachments/assets/2db3c62c-3059-4ebd-90bf-a29130f744fc)

After that you will get the trained_model.pth model file for generation.

# AMP generator — ablation and baseline suite

Built to satisfy two submission checklist items with a defensible artefact each:

| Checklist item | Evidence produced |
|---|---|
| **Ablation experiments are included.** | `Table_S1` GMM components (AIC/BIC) · `Table_S2` GRU vs. LSTM · `Table_S3` special tokens · `Table_S7` decoding temperature · plus `rf_feature_ablation.csv` |
| **Baseline comparisons to simple/trivial models (for example, 1-nearest neighbour, random forest, most frequent class) are provided.** | `Table_S4` four generation baselines · `Table_S5` most-frequent-class / 1-NN / logistic regression vs. random forest · `Table_S6` scores, k-mer divergence, novelty |

## Install and run

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch biopython tabulate
bash run_all.sh AMPuniqmoree5_clean_rmdup.fasta namp_clean.fasta
```

Verified end-to-end on synthetic data with numpy/pandas/scikit-learn/
matplotlib/scipy, torch 2.14 and biopython 1.88.

Biopython is optional but strongly recommended: without it the novelty screen
falls back to `difflib` similarity, which is not sequence identity and is
labelled as such in the output.

Environment overrides: `SEEDS=5 EPOCHS=20 NGEN=5000 bash run_all.sh ...`
Also `TEMPS="0.8 1.0 1.2 1.5"`, `COMPONENTS="1 2 3 4 5 6 7 8"`, `KS="2 3 4 5 6"`.

`kmer_eval.py`, `novelty_eval.py` and `ablation_rf.py` pick up any
`generated_*.fasta` in the output directory automatically, so a new arm is
evaluated without editing them.

## Two things the first full run settled

**The decoding temperature was the problem, not the architecture.** At
T = 1.5 the generator lost to a bigram Markov chain on random-forest
AMP-likeness and on 2-mer agreement. T > 1 flattens the softmax, which
degrades composition and k-mer match; the value was never justified, and
diversity was already ~0.9998 so nothing needed propping up. The sweep
(`Table_S7`) is an inference-only ablation — same weights, different
decoding — so it changes no architectural claim.

**Lower temperature trades distribution match against memorisation.** Choose
the operating point on both axes: `js_*`/`aa_L1` improve as T falls, while
`frac_exact_copy` rises. Report the chosen T and the novelty figure together.

## Files

| File | Role |
|---|---|
| `common.py` | Single source of truth: cleaning rules, tokenizer, metrics, featurisation |
| `ablation.py` | The three ablations; writes the model FASTA files |
| `baseline_generation.py` | Four generation baselines; writes the baseline FASTA files |
| `ablation_rf.py` | Classifier baselines + discriminative scoring |
| `kmer_eval.py` | k-mer JS divergence, sample-size corrected |
| `novelty_eval.py` | Memorisation screen (Smith–Waterman identity) |
| `make_tables.py` | Assembles Tables S1–S7 + `supplementary_tables.md` |

Stage order matters — stages 3–5 consume the FASTA files written by 1–2.

## What was fixed, and why it matters for review

1. **`pad_id` collided with alanine.** With `use_special_tokens=False` the
   vocabulary was the 20 residues and `pad_id` fell back to `0` = `A`, so
   `padding_idx=0` froze alanine's embedding and `ignore_index=0` dropped
   every alanine target from the loss. The no-special model was crippled for
   reasons unrelated to `[EOS]`, and its lower loss was partly an artefact of
   scoring ~9% fewer tokens. Residues now always occupy ids 0–19 and `[PAD]`
   sits outside the alphabet in both arms.
2. **Loss was not comparable across arms.** Different vocabularies, different
   prediction tasks. Replaced with held-out residue perplexity **renormalised
   over the 20 residue logits**. Scoring residue targets alone is not enough:
   a model carrying `[EOS]` must reserve probability mass for termination at
   every position, so its residue probabilities are structurally smaller and
   it looks worse for a reason that has nothing to do with fit. Renormalising
   conditions on "the next token is a residue" — the identical problem in
   both arms. On the test run the two arms come out indistinguishable
   (17.53 vs 17.57), which is the honest finding: special tokens buy
   controllability, not better residue modelling.
3. **Single seed.** Everything now runs with N seeds, reported as mean ± s.d.
4. **Length JS was defined but never called.** GRU vs. LSTM had no
   distributional metric behind it; the choice rested on a mean-length
   argument that does not follow. Length and k-mer JS are now computed for
   every arm.
5. **Negatives were not length-matched.** AMPs ≤129 aa vs. a non-AMP pool
   cleaned to ≤200 aa lets `length` separate the classes almost trivially.
   Now matched bin by bin, with a feature ablation to prove the point.
6. **1-NN and logistic regression were unscaled.** Features spanned
   `length ~10²` to composition `~10⁻²`. A baseline comparison is only
   meaningful if the baselines get a fair chance; both are now in a
   `StandardScaler` pipeline.
7. **Reference bars were scored in-sample.** Real AMP/non-AMP probabilities
   now come from out-of-fold predictions.
8. **Novelty screen never touched the generator.** It ran on baselines only,
   on 200 × 5,000 subsamples, with an early `break`, using
   `difflib.SequenceMatcher` called "sequence identity". Now: every sequence,
   full reference set, Smith–Waterman with BLOSUM62, distribution reported.
9. **k-mer JS had a sample-size bias.** Real set ≫ generated set in a 160k-
   dimensional space. Now subsampled to equal token counts and bootstrapped.
10. **Cleaning rules differed per script** (`max_len` = `None`/129/200) while
    the manuscript claimed one dataset. Centralised in `common.py`.
11. **Three scripts claimed "weighted AIC/BIC"** but no weights were ever
    passed. Now plain AIC/BIC, with optional true weighting available.
12. **Generation masked nothing** and used `continue` on `[PAD]`/`[BOS]`,
    re-feeding the same token and silently burning the decoding budget.
    Forbidden ids are now masked out of the softmax, and generation is
    batched.
13. **A group with zero valid sequences disappeared from the tables.** The
    no-`[EOS]` arm emits only 150-aa sequences, so cleaning removed all of
    them and the downstream scripts skipped the group — leaving a hole a
    reader would interpret as "not run" rather than as the result. Such
    groups are now reported explicitly as `n = 0` alongside `n_generated`.

## Three claims to revise in the manuscript

- **Delete** "Diversity was computed under greedy decoding." Greedy decoding
  is not implemented anywhere in this suite and never was — `generate` always
  sampled. State the temperature instead.
- **Soften** "to rule out memorization" → "to quantify the degree of
  memorisation". A screen bounds memorisation; it cannot rule it out.
- **Reframe** the special-token result. "The model never learned to
  terminate" is not right — without `[EOS]` there is no termination mechanism
  to learn. And because the decoding budget (150) exceeds the validity
  ceiling (129), a validity rate of 0 is arithmetically forced. The honest
  claim: *without an end-of-sequence symbol, output length is set by the
  decoding budget rather than learned from the data.* Show the length
  histogram (a spike at 150) as the evidence.

Also check `model_type.csv` against `special_tokens.csv`: the old draft
reported LSTM = 1.35 and no-special = 1.35, which looks like a copy-paste
error, since those are different configurations.

## Two framing points reviewers will press

**The random forest is not an independent validator.** It is trained on the
same positive pool as the generator, on compositional features the generator
explicitly learns to reproduce. A high AMP probability measures
distributional consistency with known AMPs, not antimicrobial activity. Call
it an orthogonal discriminative model and put the caveat in the limitations.

**`training_sample` is a copy oracle, not a competitor.** It wins every
distribution metric by construction and scores 1.0 on identity. Present it as
the ceiling on distribution matching and the floor on novelty — the range
within which the other methods should be read.
