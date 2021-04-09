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
- [ ] Reuters8 Dataset & DataLoader set up
    - In progress, some work on 4/8/21, Liam
    - Uploaded original dataset from paper, 4/9/21, Liam
- [ ] BERT doing standalone prediction on Reuters8
- [ ] Set up NVDM topic model on some dataset
- [ ] NVDM working on Reuters8
- [ ] Create joint model (TopicBERT)
- [ ] Achieve near baselines with TopicBERT
- [ ] Move from Jupyter to Python modules

Once we're here, it means we are ready to begin working on novel extensions.
