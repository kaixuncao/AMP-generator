# Supplementary Tables

Every generated set was produced by multinomial sampling at the temperature stated in the Methods; greedy decoding is not used anywhere in this study. Where a configuration was run with several random seeds, values are mean ± s.d. across seeds. Set sizes are given in the `n` columns.

## Table S1. GMM component selection on the sequence-length distribution

|   n_components |   k_params |    logL |    AIC |    BIC |   dAIC_vs_prev |   dBIC_vs_prev | means                                                            | variances                                                    | weights                                                 |
|---------------:|-----------:|--------:|-------:|-------:|---------------:|---------------:|:-----------------------------------------------------------------|:-------------------------------------------------------------|:--------------------------------------------------------|
|              1 |          2 | -120492 | 240987 | 241003 |         nan    |         nan    | 28.0626                                                          | 456.6569                                                     | 1.0000                                                  |
|              2 |          5 | -111124 | 222257 | 222298 |      -18729.8  |      -18705.2  | 16.7876;50.8979                                                  | 43.0984;515.3180                                             | 0.6695;0.3305                                           |
|              3 |          8 | -110018 | 220051 | 220117 |       -2205.74 |       -2181.14 | 15.5577;35.4375;72.9653                                          | 30.3524;84.9515;251.6654                                     | 0.6261;0.2388;0.1351                                    |
|              4 |         11 | -109720 | 219462 | 219552 |        -589.59 |        -564.99 | 12.5133;21.6240;38.7211;72.9249                                  | 13.8341;26.1814;61.8995;251.7623                             | 0.3847;0.3051;0.1747;0.1356                             |
|              5 |         14 | -109665 | 219357 | 219472 |        -104.59 |         -80    | 11.7399;19.9062;32.5861;47.7404;76.2395                          | 10.5807;14.6099;34.8684;82.2775;218.4642                     | 0.3406;0.2944;0.1607;0.0913;0.1129                      |
|              6 |         17 | -109541 | 219116 | 219255 |        -241.21 |        -216.61 | 11.7341;19.9772;32.3755;46.4105;69.2664;88.2620                  | 10.4851;14.2315;25.2480;27.3456;65.5146;175.3300             | 0.3420;0.2986;0.1554;0.0828;0.0782;0.0430               |
|              7 |         20 | -109407 | 218853 | 219017 |        -262.88 |        -238.28 | 11.7166;19.8008;31.4541;46.0777;69.0650;88.4903;121.1180         | 10.2215;11.9419;26.3210;32.7374;49.1294;50.4019;28.7438      | 0.3456;0.2793;0.1643;0.0923;0.0781;0.0374;0.0029        |
|              8 |         23 | -109350 | 218746 | 218934 |        -107.59 |         -82.99 | 11.3376;18.4567;25.8255;34.9274;46.7136;68.8849;88.5606;121.3251 | 8.9100;7.7038;8.9674;12.7402;24.8586;52.0824;50.9879;26.4146 | 0.3196;0.2504;0.1252;0.1018;0.0829;0.0802;0.0371;0.0029 |

*Free parameters k = 3n − 1 for an n-component one-dimensional full-covariance mixture. AIC/BIC are unweighted.*

## Table S2. Ablation — recurrent cell type (GRU vs. LSTM)

| model   |   n_seeds | val_residue_ppl_renorm_fmt   | validity_rate_fmt   | diversity_fmt   | mean_len_fmt     | std_len_fmt      | js_length_fmt   | js_3mer_fmt     | aa_L1_fmt       |
|:--------|----------:|:-----------------------------|:--------------------|:----------------|:-----------------|:-----------------|:----------------|:----------------|:----------------|
| GRU     |         3 | 5.8956 ± 0.0217              | 0.9875 ± 0.0052     | 0.9998 ± 0.0003 | 26.7500 ± 1.0658 | 21.2367 ± 0.5143 | 0.1656 ± 0.0187 | 0.3217 ± 0.0022 | 0.1238 ± 0.0026 |
| LSTM    |         3 | 5.3873 ± 0.0076              | 0.9893 ± 0.0051     | 0.9998 ± 0.0003 | 26.0067 ± 0.4701 | 21.2500 ± 1.1769 | 0.1706 ± 0.0177 | 0.3315 ± 0.0027 | 0.1463 ± 0.0057 |

*Identical data, tokenizer, schedule and seeds. Perplexity is computed on held-out data over amino-acid targets, renormalised over the 20 residue logits. Training loss is not comparable across arms and is recorded in the per-seed CSV only.*

## Table S3. Ablation — special tokens ([BOS]/[EOS])

| variant      |   n_seeds | val_residue_ppl_renorm_fmt   | validity_rate_fmt   | diversity_fmt   | mean_len_fmt      | std_len_fmt      | js_length_fmt   | js_3mer_fmt     | aa_L1_fmt       |
|:-------------|----------:|:-----------------------------|:--------------------|:----------------|:------------------|:-----------------|:----------------|:----------------|:----------------|
| no_special   |         3 | 5.6355 ± 0.0573              | 0.0000 ± 0.0000     | 1.0000 ± 0.0000 | 150.0000 ± 0.0000 | 0.0000 ± 0.0000  | 1.0000 ± 0.0000 | 0.2957 ± 0.0063 | 0.2079 ± 0.0135 |
| with_special |         3 | 5.8956 ± 0.0217              | 0.9875 ± 0.0052     | 0.9998 ± 0.0003 | 26.7500 ± 1.0658  | 21.2367 ± 0.5143 | 0.1656 ± 0.0187 | 0.3217 ± 0.0022 | 0.1238 ± 0.0026 |

*[PAD] is retained in both arms as a batching mechanism and is assigned an id outside the amino-acid alphabet, so no residue is excluded from the loss in either arm. Residue perplexity is renormalised over the 20 residue logits, so the arm carrying [EOS] is not penalised for the probability mass it must reserve for termination; on that like-for-like measure the two arms model residues equally well, and the difference between them is one of controllability, not of fit. Without [EOS] the output length is set entirely by the decoding budget (150 tokens) rather than learned from the data, so no sequence can fall inside the 5–129 aa validity window.*

## Table S4. Generation baselines (simple/trivial models)

| method          | description                                |   n_seeds | validity_rate_fmt   | diversity_fmt   | mean_len_fmt     | js_length_fmt   | aa_L1_fmt       | js_2mer_fmt     | js_3mer_fmt     |
|:----------------|:-------------------------------------------|----------:|:--------------------|:----------------|:-----------------|:----------------|:----------------|:----------------|:----------------|
| bigram_markov   | First-order Markov chain over residues     |         3 | 1.0000 ± 0.0000     | 1.0000 ± 0.0000 | 27.7733 ± 0.4277 | 0.0743 ± 0.0016 | 0.0127 ± 0.0005 | 0.0528 ± 0.0011 | 0.3227 ± 0.0008 |
| freq_matched    | Residues drawn at training-set frequencies |         3 | 1.0000 ± 0.0000     | 1.0000 ± 0.0000 | 28.0133 ± 0.3853 | 0.0691 ± 0.0029 | 0.0157 ± 0.0004 | 0.1298 ± 0.0022 | 0.3571 ± 0.0035 |
| random          | Uniform random over 20 canonical residues  |         3 | 1.0000 ± 0.0000     | 1.0000 ± 0.0000 | 28.4333 ± 0.1747 | 0.0752 ± 0.0083 | 0.4290 ± 0.0059 | 0.3240 ± 0.0016 | 0.4990 ± 0.0004 |
| training_sample | Copied from the training set (COPY ORACLE) |         3 | 1.0000 ± 0.0000     | 0.9655 ± 0.0035 | 27.9033 ± 0.5784 | 0.0748 ± 0.0080 | 0.0194 ± 0.0053 | 0.0559 ± 0.0011 | 0.2437 ± 0.0058 |

*All baselines draw lengths from the empirical AMP length distribution, so comparisons are not confounded by length; validity rate is therefore 1.0 by construction and is reported as a sanity check only. `training_sample` is a copy oracle, not a competing method: it marks the attainable floor on the divergence metrics and the floor on novelty.*

## Table S5. Classifier baselines vs. the random-forest validator

| classifier         | AUC             | Accuracy        | Precision       | Recall          | F1              |
|:-------------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| MostFrequentClass  | 0.5000 ± 0.0000 | 0.4997 ± 0.0001 | 0.1999 ± 0.2737 | 0.4000 ± 0.5477 | 0.2666 ± 0.3650 |
| 1-NearestNeighbour | 0.8684 ± 0.0061 | 0.8684 ± 0.0060 | 0.8336 ± 0.0074 | 0.9206 ± 0.0059 | 0.8749 ± 0.0054 |
| LogisticRegression | 0.8364 ± 0.0070 | 0.7653 ± 0.0084 | 0.7642 ± 0.0124 | 0.7678 ± 0.0194 | 0.7658 ± 0.0093 |
| RandomForest       | 0.9540 ± 0.0042 | 0.8841 ± 0.0084 | 0.8819 ± 0.0117 | 0.8873 ± 0.0102 | 0.8845 ± 0.0081 |

*Five-fold stratified cross-validation, mean ± s.d. across folds. Negatives are length-matched to the positives, so sequence length carries no label information. Scaling-sensitive baselines (1-NN, logistic regression) are fitted inside a standardisation pipeline. A feature ablation (rf_feature_ablation.csv) confirms the forest is not acting as a length detector.*

## Table S6. Evaluation of generated and baseline sequences

| group                         |    n |   n_generated |   mean_prob |   sd_prob |   frac_prob_ge_0.5 | js_2mer            | js_3mer            | js_4mer            | metric                  |   mean_max_identity |   global_max_identity |   frac_identity_gt_0.90 |   frac_exact_copy |
|:------------------------------|-----:|--------------:|------------:|----------:|-------------------:|:-------------------|:-------------------|:-------------------|:------------------------|--------------------:|----------------------:|------------------------:|------------------:|
| Baseline: bigram_markov       | 2000 |          2000 |      0.6037 |    0.1577 |             0.7575 | 0.0540 ± 0.0022    | 0.3225 ± 0.0034    | 0.7926 ± 0.0017    | smith_waterman_blosum62 |              0.6813 |                     1 |                  0.017  |            0.017  |
| Baseline: freq_matched        | 2000 |          2000 |      0.6059 |    0.1517 |             0.7745 | 0.1326 ± 0.0028    | 0.3601 ± 0.0038    | 0.8142 ± 0.0018    | smith_waterman_blosum62 |              0.6678 |                     1 |                  0.009  |            0.009  |
| Baseline: random              | 2000 |          2000 |      0.4251 |    0.1558 |             0.3365 | 0.3242 ± 0.0035    | 0.5004 ± 0.0042    | 0.9011 ± 0.0009    | smith_waterman_blosum62 |              0.6266 |                     1 |                  0.0085 |            0.0085 |
| Baseline: training_sample     | 2000 |          2000 |      0.7182 |    0.168  |             0.8945 | 0.0555 ± 0.0023    | 0.2399 ± 0.0025    | 0.6132 ± 0.0043    | smith_waterman_blosum62 |              0.9999 |                     1 |                  0.9995 |            0.9995 |
| Generated (GRU)               | 1982 |          1998 |      0.5835 |    0.1744 |             0.7185 | 0.1277 ± 0.0037    | 0.3201 ± 0.0042    | 0.7825 ± 0.0017    | smith_waterman_blosum62 |              0.7005 |                     1 |                  0.0605 |            0.0494 |
| Generated (GRU, T=0.8)        | 1986 |          2000 |      0.702  |    0.1536 |             0.9063 | 0.1115 ± 0.0044    | 0.2943 ± 0.0036    | 0.6665 ± 0.0042    | smith_waterman_blosum62 |              0.8505 |                     1 |                  0.43   |            0.3369 |
| Generated (GRU, T=1)          | 1993 |          2000 |      0.6699 |    0.1665 |             0.853  | 0.0753 ± 0.0025    | 0.2712 ± 0.0025    | 0.6978 ± 0.0021    | smith_waterman_blosum62 |              0.789  |                     1 |                  0.2449 |            0.1927 |
| Generated (GRU, T=1.2)        | 1988 |          2000 |      0.6272 |    0.1758 |             0.7812 | 0.0877 ± 0.0033    | 0.2827 ± 0.0031    | 0.7299 ± 0.0015    | smith_waterman_blosum62 |              0.7476 |                     1 |                  0.1413 |            0.1041 |
| Generated (GRU, T=1.5)        | 1982 |          1998 |      0.5835 |    0.1744 |             0.7185 | 0.1277 ± 0.0037    | 0.3201 ± 0.0042    | 0.7825 ± 0.0017    | smith_waterman_blosum62 |              0.7005 |                     1 |                  0.0605 |            0.0494 |
| Generated (GRU, no special)   |    0 |          2000 |    nan      |  nan      |           nan      | no valid sequences | no valid sequences | no valid sequences | smith_waterman_blosum62 |            nan      |                   nan |                nan      |          nan      |
| Generated (GRU, with special) | 1982 |          1998 |      0.5835 |    0.1744 |             0.7185 | 0.1277 ± 0.0037    | 0.3201 ± 0.0042    | 0.7825 ± 0.0017    | smith_waterman_blosum62 |              0.7005 |                     1 |                  0.0605 |            0.0494 |
| Generated (LSTM)              | 1983 |          2000 |      0.5699 |    0.1797 |             0.6717 | 0.1364 ± 0.0032    | 0.3340 ± 0.0030    | 0.7975 ± 0.0012    | smith_waterman_blosum62 |              0.6998 |                     1 |                  0.056  |            0.0489 |
| Real AMP (out-of-fold)        | 3992 |           nan |      0.7726 |    0.2007 |             0.8873 | nan                | nan                | nan                | nan                     |            nan      |                   nan |                nan      |          nan      |
| Real non-AMP (out-of-fold)    | 3992 |           nan |      0.2517 |    0.1816 |             0.119  | nan                | nan                | nan                | nan                     |            nan      |                   nan |                nan      |          nan      |

*Random-forest probability (real groups scored out-of-fold), sample-size-corrected k-mer Jensen–Shannon divergence, and the novelty screen (maximum identity of each generated sequence to any training sequence). A group with n = 0 produced no sequence inside the validity window, which is itself the result rather than a missing measurement. The random forest reflects distributional consistency with known AMPs, not antimicrobial activity.*

## Table S7. Ablation — decoding temperature

|   temperature |   n_seeds | validity_rate_fmt   | diversity_fmt   | mean_len_fmt     | js_length_fmt   | aa_L1_fmt       | js_2mer_fmt     | js_3mer_fmt     |   mean_max_identity |   frac_identity_gt_0.90 |   frac_exact_copy |
|--------------:|----------:|:--------------------|:----------------|:-----------------|:----------------|:----------------|:----------------|:----------------|--------------------:|------------------------:|------------------:|
|           0.8 |         3 | 0.9942 ± 0.0025     | 0.9485 ± 0.0065 | 28.2767 ± 1.0604 | 0.1269 ± 0.0086 | 0.1031 ± 0.0171 | 0.1078 ± 0.0127 | 0.2924 ± 0.0123 |              0.8469 |                  0.422  |            0.3303 |
|           1   |         3 | 0.9938 ± 0.0024     | 0.9937 ± 0.0018 | 27.4767 ± 1.3168 | 0.1361 ± 0.0060 | 0.0573 ± 0.0078 | 0.0746 ± 0.0052 | 0.2727 ± 0.0075 |              0.7853 |                  0.2403 |            0.1882 |
|           1.2 |         3 | 0.9928 ± 0.0029     | 0.9992 ± 0.0010 | 27.2400 ± 0.8221 | 0.1423 ± 0.0082 | 0.0673 ± 0.0084 | 0.0878 ± 0.0022 | 0.2857 ± 0.0053 |              0.7435 |                  0.1388 |            0.1016 |
|           1.5 |         3 | 0.9875 ± 0.0052     | 0.9998 ± 0.0003 | 26.7500 ± 1.0658 | 0.1656 ± 0.0187 | 0.1238 ± 0.0026 | 0.1265 ± 0.0023 | 0.3217 ± 0.0022 |              0.6977 |                  0.06   |            0.0489 |

*Temperature is an inference hyper-parameter: one model per seed was trained and then sampled at each value, so all differences are a decoding effect. Distribution match and novelty move in opposite directions with temperature, and the operating point is chosen on both. Novelty columns are from the first seed.*
