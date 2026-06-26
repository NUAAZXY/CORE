# Experimental results in the rebuttal

## 1. Scalability
| Dictionary Sizes     |                  | Repobench         |           | LongBench |             | Avg.      |
|----------------------|------------------|-------------------|-----------|-----------|-------------|-----------|
|                      | cross_file_first | cross_file_random | in_file   | LCC       | Repobench-P |           |
| {32, 256, 4096}      | 5.02             | 18.12             | 17.40     | 13.19     | 8.05        | 12.36     |
| {128, 1,024, 16,384} | 5.18             | 18.98             | 18.28     | 14.29     | 8.90        | 13.13     |
| {256, 2,048, 32,768} | 5.24             | **19.12**         | 18.20     | **15.93** | 8.47        | 13.39     |
| {512, 4,096, 65,536} | **5.66**         | 19.08             | **19.12** | **15.93** | **9.32**    | **13.82** |

To evaluate the scaling capability of COREGEN, we conduct experiments by progressively enlarging the dictionary size. The empirical results demonstrate that model performance improves consistently with the expansion of the dictionary, thereby providing strong evidence that COREGEN is able to effectively scale to larger model capacities.

## 2. New Baselines
| Methods    |                  | Repobench         |           | LongBench |             | Avg.      |
|------------|------------------|-------------------|-----------|-----------|-------------|-----------|
|            | cross_file_first | cross_file_random | in_file   | LCC       | Repobench-P |           |
| RLCoder    | 4.12             | 16.48             | 17.82     | 13.74     | 8.05        | 12.04     |
| GraphCoder | 4.04             | 16.08             | 17.20     | 13.19     | 8.05        | 11.71     |
| COREGEN    | **5.18**         | **18.98**         | **18.28** | **14.29** | **8.90**    | **13.13** |

We incorporate RLCoder and GraphCoder as supplementary baselines. The results demonstrate that COREGEN consistently outperforms these newly introduced models, further validating its effectiveness.

### RLCoder Implementation
Since the retriever in RLCoder is not open-sourced, we followed the approach described in the original paper and trained Unixcoder-base on 2,000 samples for 20 epochs. The experimental results show that while RLCoder outperforms the baseline in our paper, COREGEN achieves even stronger performance on repository-level code generation datasets. Moreover, RLCoder requires an additional model and extra reinforcement learning training, whereas COREGEN delivers better results with significantly less computational cost.

## 3.Function-Level Generation
| Methods | HumanEval |
|---------|-----------|
|         | pass@1    |
| RoPE    | 3.66      | 
| ALiBi   | 3.05      |
| CoDE    | 3.66      |
| COREGEN | **4.88**  | 

We evaluate different LCG methods on the short-context dataset (HumanEval). The experimental results demonstrate that even when reduced to full-attention in the short-context setting, COREGEN still achieves superior performance.

## 4. Additional Ablation
| Methods                         |                  | Repobench         |           | LongBench |             | Avg.      |
|---------------------------------|------------------|-------------------|-----------|-----------|-------------|-----------|
|                                 | cross_file_first | cross_file_random | in_file   | LCC       | Repobench-P |           |
| COREGEN                         | **5.18**         | **18.98**         | **18.28** | **14.29** | **8.90**    | **13.13** |
| w/o Multi-level Dictionary      | 4.82             | 17.48             | 18.02     | 13.19     | 7.63        | 12.23     |
| w/o TopK Activation (w L1 loss) | 4.78             | 17.16             | 18.00     | 13.74     | 7.20        | 12.17     |

We further present additional ablation experiments, which verify that our proposed Multi-level Dictionary and Top-K Activation achieve superior performance compared to the standard L1 loss, thereby underscoring the effectiveness of our design choices.

## 5. Statistical Significance

| Dataset                     | COREGEN | Baseline Mean | Mean Diff | t-statistic | p-value  |
|-----------------------------|---------|---------------|-----------|-------------|----------|
| RepoBench cross_file_first  | 5.18    | 3.17          | 2.01      | 4.85        | < 0.0001 | 
| RepoBench cross_file_random | 18.98   | 14.37         | 4.61      | 2.58        | < 0.01   | 
| RepoBench in_file           | 18.28   | 16.06         | 2.22      | 1.12        | > 0.1    |
| LongBench LCC               | 14.29   | 11.63         | 2.66      | 1.58        | < 0.1    |
| LongBench RepoBench-P       | 8.90    | 6.36          | 2.54      | 3.22        | < 0.01   |
| Overall Average             | 13.13   | 10.32         | 2.81      | 2.15        | < 0.05   |

We evaluate statistical significance through paired t-tests. The results show that COREGEN exhibits a clear and statistically significant advantage in scenarios involving cross-file dependencies, while also maintaining a significant improvement in overall average performance.

## 6. Hyperparameter Selection of Activated Features

| Activated Features |                  | Repobench         |           | LongBench |             | Avg.  |
|--------------------|------------------|-------------------|-----------|-----------|-------------|-------|
|                    | cross_file_first | cross_file_random | in_file   | LCC       | Repobench-P |       |
| {1, 1, 1}          | 4.86             | 16.54             | 15.48     | 12.09     | 7.62        | 11.32 |
| {1, 2, 8}          | 5.18             | **18.98**         | 18.28     | **14.29** | **8.90**    | 13.13 |
| {2, 4, 16}         | 5.16             | 18.58             | **18.32** | 13.74     | **8.90**    | 12.94 |
| {4, 8, 32}         | **5.24**         | 18.82             | 18.20     | **14.29** | 8.47        | 13.00 |
| {64, 128, 1024}    | 3.28             | 12.72             | 10.58     | 6.59      | 5.5         | 7.73  |

The experimental results revealed that the proposed sparse activation achieves superior performance. In contrast, using an excessive number of features (greater than 1000) leads to entanglement and reduced interpretability, whereas using too few features (e.g., a single feature) fails to capture critical dependencies. Moreover, performance remains stable across a reasonable range of parameter variations (within 2% difference), indicating that COREGEN’s effectiveness stems from its inherent robustness rather than sensitivity to precise hyperparameter tuning.


## 7. Error Cases
| Method      | Syntax Error | Semantic Error | Logic Errors | Context Errors | Other |
|-------------|--------------|----------------|--------------|----------------|-------|
| RoPE        | 5            | 10             | 25           | 55             | 5     |
| ALiBi       | 10           | 15             | 10           | 65             | 0     |
| CoDE        | 5            | 15             | 20           | 40             | 20    | 
| RAG         | 15           | 10             | 30           | 45             | 0     |
| RepoCoder   | 10           | 20             | 25           | 40             | 5     |
| RepoFuse    | 10           | 15             | 25           | 40             | 10    |
| **COREGEN** | 5            | 25             | 30           | 25             | 15    |

We conduct a systematic error analysis using manually annotated error cases in the revision, including 20 samples across different error categories (with more to be added in the updated version). We find that COREGEN performs significantly better than the baselines on contextual errors, with an error rate of 25% compared to the baselines’ average of 47.5%.

# Model Training and Evaluation Scripts

This repository provides a collection of files for training and evaluating the COREGEN method on various benchmarks. You can either train COREGEN from scratch using your own dataset or evaluate it using our pre-trained models available on Hugging Face.

## Quick Start

### Training a Model

```bash
# Basic training with default parameters
./train.sh
```

### Evaluating on Benchmarks

```bash
# Evaluate on all benchmarks
python run_longbench.py
python run_repobench.py
python eval.py
```

### Evaluating Perplexity

```bash
# Basic perplexity evaluation
python eval_ppl.py
```

## Detailed Usage

### Training Script (`train.sh`)

#### Model Parameters
- `--model_name`: Base model name (default: codellama-7b)
- `--vocab_size`: Vocabulary size (default: 32016)
- `--hidden_size`: Hidden layer size (default: 768)
- `--num_hidden_layers`: Number of hidden layers (default: 12)
- `--num_attention_heads`: Number of attention heads (default: 8)
- `--max_position_embeddings`: Maximum position embeddings (default: 32768)

#### Training Parameters
- `--train_batch_size`: Training batch size (default: 16)
- `--learning_rate`: Learning rate (default: 5e-4)
- `--max_train_steps`: Maximum training steps (default: 150000)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 1)
- `--use_wandb`: Enable Weights & Biases logging
- `--gradient_checkpointing`: Enable gradient checkpointing for memory efficiency

#### Data Parameters
- `--train_dataset`: Training dataset path (default: starcoder_20B)
- `--valid_dataset`: Validation dataset path (default: starcoder_20Btokens_val)
- `--seq_length`: Sequence length (default: 1024)
- `--extrapolate_length`: Extrapolation length (default: 8192)

### Benchmark Evaluation Script (`eval_repos.sh`)

#### General Parameters
- `--model_path`: Path to the trained model
- `--benchmark`: Benchmark to run (repobench, longbench, or all)
- `--output_dir`: Output directory for results (default: ./results)

#### RepoBench Parameters
- `--max_token_nums`: Maximum token number for prompt (default: 6000)
- `--levels`: Levels to filter by (default: 2k,4k,8k,12k,16k)
- `--language`: Programming language (default: python)
- `--max_new_tokens`: Maximum new tokens to generate (default: 32)

#### LongBench Parameters
- `--max_length`: Maximum sequence length (default: 6000)
- `--longbench_datasets`: Datasets to evaluate (default: lcc,repobench-p)

### Perplexity Evaluation Script (`eval_ppl.sh`)

#### Parameters
- `--model_name`: Model name or path
- `--local_model_path`: Local model path (if not downloading)
- `--valid_dataset`: Validation dataset path
- `--seq_length`: Sequence length (default: 8192)
- `--batch_size`: Batch size (default: 1)
- `--max_eval_steps`: Maximum evaluation steps (default: 2000)
- `--device`: Device to use (default: cuda)
- `--output_file`: Output file for results (default: evaluation_results.txt)

## Output Structure

```
results/
├── repobench/
│   ├── evaluation_results.json
│   └── predictions/
├── longbench/
│   ├── evaluation_results.json
│   └── predictions/
└── training_logs/
    ├── checkpoints/
    └── tensorboard_logs/
```





# COREGEN
# COREGEN
# COREGEN
# COREGEN
# COREGEN
