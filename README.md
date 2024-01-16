# Machine Learning Operations project

Alex Belai s233423 `<br/>`
Jan Ljubas s237214 `<br/>`
Noa Margeta 232470 `<br/>`
Ana Marija Pavičić s232468 `<br/>`
Filip Penzar s232452 `<br/>`

## Project description

This is a group project for the DTU course Machine Learning Operations [DTU:02476](https://skaftenicki.github.io/dtu_mlops/projects/).

### Goal

Our project revolves around creating and deploying a Machine Learning model for an Image Classification problem.
The goal for our model is to classify facial features (e.g. beard, glasses etc.) on images of people.

### Framework

We used [PyTorch](https://pytorch.org/) to develop our model. Furthermore, we used the [Transformer framework](https://github.com/huggingface/transformers) - the multimodal framework used in various cutting edge technologies. The Transformer framework performs approximately as well as some state-of-the-art Convolutional Neural Network models (CNNs), but requires much less time and computational resources to train.

### Data

Our dataset for the model is the Kaggle dataset [CelebFaces Attributes](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data). It consists of over 200k images of celebrities with 40 binary attribute annotations.

### Model

The model we chose to fine-tune with our dataset is [Vision Transformer(ViT)](https://huggingface.co/docs/transformers/model_doc/vit), which was pretrained on 14 million images and can classify 1000 different items. We could also have used other ViT-type models, such as [ViTMAE](https://huggingface.co/docs/transformers/model_doc/vit_mae) or [ViTMSN](https://huggingface.co/docs/transformers/model_doc/vit_msn). However, for this project we chose the elementary model.

## Project structure

```
├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── src  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```
