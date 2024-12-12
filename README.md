# DSS-RL

This repository contains the code for Tovi Tu and Anxu Wang's final project for CSE510 at WashU. The project aims at developing a RL-based method for diversity-promoting subset selection. It is largely motivated by instruction-tuning in LLM.

## Usage

1. Install the required packages.
2. Run the jupyter notebooks for training and evaluation. They also replicate the results we have in the report. However, the result is subject to random training dynamics.
3. For LLM training, run the finetune.sh which assumes the data files are in the playground folder. You may generate the training data by running the debug.ipynb, which is for selecting subsets from the Alpaca dataset.
4. Use our provided script to evaluate the LLM. The definition files for each individual evaluation are in the eval/ folder. We use llm-eval-harness for automatic evaluation.
