# TopicBERT-PyTorch
PyTorch implementation of [Chaudhary et. al. 2020's TopicBERT](https://arxiv.org/pdf/2010.16407.pdf)

### Getting Started:

Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) if you have not already done so. Then run

```
conda env create -f environment.yml
```

This will create a Python environment that strictly adheres to the versioning indicated in the [project proposal](https://drive.google.com/file/d/1oEE8oxiM95Tf99SxUhPXgZj3GkotFtlM/view). It is intended to closely mirror Google Colab.


------

## Roadmap

- [X] Have working BERT on some dataset (SST-2)
    - Completed on 4/8/21, Liam
- [X] Reuters8 Dataset & DataLoader set up
    - In progress, some work on 4/8/21, Liam
    - Dataset & DataLoader done on 4/9/21, Liam
- [ ] BERT doing standalone prediction on Reuters8
- [ ] Set up NVDM topic model on some dataset
- [ ] NVDM working on Reuters8
- [ ] Create joint model (TopicBERT)
- [ ] Achieve near baselines with TopicBERT
- [ ] Move from Jupyter to Python modules

Once we're here, it means we are ready to begin working on novel extensions.


-----

## Differences

This section maintains a (non-definitive) list of differences between the original implementation and this repository's code.

- `F_MIN` is set to `7`  rather than `10` in preprocessing Reuters8. The original authors may have already preprocessed the entire corpus instead of each dataset (train, val, test). Our experiments show `K = ~19,000` where `K` is vocab size, and `F_MIN = 7` yields the desired `K = ~4800`.
- The original authors use `bert-base-cased`. As all data is lowercased across datasets in the original experiments, we change this to `bert-base-uncased`.