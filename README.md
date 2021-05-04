# TopicBERT-PyTorch
PyTorch implementation of [Chaudhary et al. 2020's TopicBERT](https://arxiv.org/pdf/2010.16407.pdf)

### Getting Started:

Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) if you have not already done so. Then run

```
conda env create -f environment.yml
```

This will create a Python environment that strictly adheres to the versioning indicated in the [project proposal](https://drive.google.com/file/d/1oEE8oxiM95Tf99SxUhPXgZj3GkotFtlM/view). It is intended to closely mirror Google Colab.

Then train the model via `main.py`. There are many options that can be set, run `python main.py -h` to see more.

One particularly helpful option is `-s PATH` or `--save PATH`, which saves the given options as a JSON file that
can easily be used again with `--load PATH`.

Sample `config.json`:

```js
{
    "dataset": "reuters8",
    "label_path": ".../labels.txt",
    "train_dataset_path": ".../training.tsv",
    "val_dataset_path": ".../validation.tsv",
    "test_dataset_path": ".../test.tsv",
    "num_workers": 8,
    "batch_size": 16,
    "warmup_steps": 10,
    "lr": 2e-05,
    "alpha": 0.9,
    "num_epochs": 2,
    "clip": 1.0,
    "seed": 42,
    "device": "cuda",
    "val_freq": 0.0,
    "test_freq": 0.0,
    "disable_tensorboard": false,
    "tensorboard_dir": "runs/topicbert-512",
    // directory where checkpoints should be
    "resume": ".../checkpoints/", 
    // whether to look for a checkpoint in above or just save a new one there
    "save_checkpoint_only": true, 
    "verbose": true,
    "silent": false,
    "load": null,
    "save": "config.json"
}
```

**Alternatively**, open `experiment.ipynb` in Google Colab: [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/github/liamrahav/TopicBERT-PyTorch/blob/main/experiment.ipynb)

------

## Roadmap (DONE)

- [X] Have working BERT on some dataset (SST-2)
    - Completed on 4/8/21, Liam
- [X] Reuters8 Dataset & DataLoader set up
    - Dataset & DataLoader done on 4/9/21, Liam
- [X] BERT doing standalone prediction on Reuters8
    - Done — achieves 99.5% train, 98.0% val accuracy run on Google Colab, 4/10/21, Liam 
- [X] Set up NVDM topic model on some dataset
- [X] NVDM working on Reuters8
    - Done — error behaves as expected when training, needs further analysis, 4/18/21, Liam
- [X] Create joint model (TopicBERT)
    - Coding complete, 4/19/21, Liam
- [X] Achieve near baselines with TopicBERT
    - We achieve 0.96 F1 score on Reuters8 with TopicBERT-512, outperforming the original paper marginally. See differences section for potental factors.
    - Done, 4/19/21, Liam
- [X] Move from Jupyter to Python modules
    - All "modules" converted, 4/25/21, Liam. 
    - `training` package and `main.py` complete, 4/26/21, Liam.
- [X] Measure performance baselines
    - All baselines finalized, 5/3/21, Liam.

Happy to report that the model has performance (runtime & accuracy) characteristics as expected! 

Extension Ideas:
- Pre-train VAE
    - Will likely require more complex VAE architecture to account for posterior collapse
- Test new datasets in topic classification
- Test datasets in a different domain (e.g. NLI, GLUE)

-----

## Differences

This section maintains a (non-definitive) list of differences between the original implementation and this repository's code.

- `F_MIN` set to `10` on Reuters8 dataset yields a vocab size of `K = 4832` rather than `K = 4813` reported in the original paper, despite following the same text-cleaning guidelines. We assume this will not significantly affect results.
- The original authors use `bert-base-cased`. As all data is lowercased across datasets in the original experiments, we change this to `bert-base-uncased`.
- Labels are encoded one-hot. We use `torch.max(...)[1]` to extract prediction & label indices. These indices can be converted back and forth with label strings via the `Reuters8Dataset` class (`dataset.label_mapping[index]` and `dataset.label_mapping[label_str]`).
- NVDM in the original paper uses `tanh` activation for multiliayer perceptron in NVDM. However, the author's TensorFlow implementation uses `sigmoid`. We use `GELU`, as the NVDM paper ([Miao et al. 2016](https://arxiv.org/pdf/1511.06038.pdf)) uses this as well.
- TopicBERT as described in the paper has a projection layer consisting of a single matrix $\mathbf{P} \in \mathbf{R}^{\hat{H} \times H_B}$. We add `GELU` activation after $\mathbf{P}$. The original author's TensorFlow implementation utilizes a `tf.keras.layers.Dense` layer, which adds a bias vector and `GELU` activation after $\mathbf{P}$.
