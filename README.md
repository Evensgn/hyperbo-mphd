This repo is built based on the codebase of [HyperBO](https://github.com/google-research/hyperbo). We made modifications to some original files to expand its utility and added new files under the `hyperbo/experiments_xmanager` and `hyperbo/experiments` folder.

## MPHD code

The latest experiments of MPHD were configured to be run on Google Cloud Platform (GCP) in a distributed fashion using [XManager](https://github.com/google-deepmind/xmanager). The code used for the latest version of experiments is in the directory `experiment_xmanager`.

- Synthetic Super-dataset (L) generation: see `hyperbo/experiments_xmanager/gen_synthetic_data_for_bohss.py`. This script generates the Synthetic Super-dataset (L) for the experiments of MPHD. Parameters of data-generation are configured in the script.
- Definition of experiment constants: see `hyperbo/experiments_xmanager/experiment_defs.py`. This script contains definition of experiment constants including the group ID (the dir name in the GCP storage bucket used by training/test jobs of the same experiment), paths of the datasets used (Synthetic Super-dataset (L), HPO-B Super-dataset, PD1 dataset), and other workflow-dependent constants (e.g. which super-dataset to use).
- Configuration of experiment hyperparameters and additional constant values: see `hyperbo/experiments_xmanager/hpl_bo_split_config.py`. This files defines hyperparameters such as number of iterations and learning rate for pre-training stages, setup hyperparameters of BO, ground-truth HGP, non-informative HGP, and hand-specified HGP.
- Run MPHD experiments on GCP using XManager: see `hyperbo/experiments_xmanager/hpl_bo_xmanager_launcher.py`. This script runs the experiments of MPHD on GCP using XManager. The workflow of experiments (pre-training, BO testing, etc.) are defined in the script and can be modified to run different workflows. This file calls functions in `hyperbo/experiments_xmanager/hpl_bo_split_worker.py` to run the experiments in a distributed way.
- Worker functions of MPHD experiments: see `hyperbo/experiments_xmanager/hpl_bo_split_worker.py`. This file contains the worker functions of the experiments of MPHD. The worker functions are called by `hyperbo/experiments_xmanager/hpl_bo_xmanager_launcher.py` for distributed training and testing on GCP.
- Plotting experiment results such as BO performance curves: see `hyperbo/experiments_xmanager/plot.py`.
- Test the asymptotic behavior of the =pre-training of MPHD on the Synthetic Super-dataset (L): see `hyperbo/experiments_xmanager/hpl_bo_xmanager_launcher_vary_num_dataset.py` and `hyperbo/experiments_xmanager/plot_vary_num_datasets.py`.
- Dataset loaders for the Synthetic Super-dataset (L), HPO-B Super-dataset and PD1 Dataset: see `hyperbo/bo_utils/data.py`.

The directory `experiments` contains the code used for an older version of experiments, including generation of Synthetic Super-dataset (S) and the empirical asymptotic analysis on it.

- Synthetic Super-dataset (S) generation: see `hyperbo/experiments/synthetic_data_generation.py`. This script generates the Synthetic Super-dataset (S) for the empirical asymptotic analysis. Parameters of data-generation are configured in the script.
- Worker functions of experiments on the Synthetic Super-dataset (S): see `hyperbo/experiments/test_hyperbo_plus_split_worker.py`. This file contains the worker functions. The worker functions can be called by `hyperbo/experiments/test_hyperbo_plus_split_scheduler.py` using multiprocessing.
- Test the asymptotic behavior of fitting a single GP: see `hyperbo/experiments/test_asymptotics.py`.
- Test the asymptotic behavior of the two-step pre-training of HyperBO+ on the Synthetic Super-dataset: see `hyperbo/experiments/test_hyperbo_plus_split_asymptotic_scheduler.py` and `hyperbo/experiments/test_hyperbo_plus_split_asymptotic_aggregator.py`.
- Dataset loaders for the Synthetic Super-dataset (S): see `hyperbo/bo_utils/data.py`.

The remaining part of this README document is copied from the original HyperBO repo except from slight modification to the installation instructions.

---

# HyperBO - Prior Discovery
A Jax/Flax codebase for prior discovery in meta Bayesian optimization.
The algorithm and analyses can be found in *[Pre-trained Gaussian processes for Bayesian optimization](https://arxiv.org/pdf/2109.08215.pdf)*. Slides are available [at this link](https://ziw.mit.edu/pub/hyperbo_slides.pdf) with [video at the AutoML Seminars](https://www.youtube.com/watch?v=cH4-hHXvO5c). 

Also see [GPax](https://github.com/google-research/gpax) for a more modular implementation of Gaussian processes used by HyperBO based on [Tensorflow Probability](https://www.tensorflow.org/probability) with Jax backend.

Disclaimer: This is not an officially supported Google product.

## Installation
We recommend using Python 3.7 for stability.

To install this codebase as a library inside a virtual environment, run
```
python3 -m venv env-pd
source env-pd/bin/activate
pip install --upgrade pip
pip install .
```

## Dataset
To download the dataset, please copy and paste the following link to your browser's address bar.
```
http://storage.googleapis.com/gresearch/pint/pd1.tar.gz
```
See pd1/README.txt for more information. The data is licensed under the CC-BY 4.0 license.

If you'd like to use the evaluations at each training step, the relevant columns of the data frame are
```
'valid/ce_loss'
'train/ce_loss',
'train/error_rate',
```
etc. They will hold arrays aligned with the global_step column that indicates what training step the measurement was taken at.

See the "best_\*" columns for the best measurement achieved over training.


## Usage
See tests.

## Citing
```
@article{wang2021hyperbo,
  title={Pre-training helps Bayesian optimization too},
  author={Wang, Zi and Dahl, George E and Swersky, Kevin and Lee, Chansoo and Mariet, Zelda and Nado, Zachary and Gilmer, Justin and Snoek, Jasper and Ghahramani, Zoubin},
  journal={arXiv preprint arXiv:2109.08215},
  year={2022}
}
```
