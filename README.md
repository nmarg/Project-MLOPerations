# Machine Learning Operations project

Alex Belai s233423 `<br/>`
Jan Ljubas s237214 `<br/>`
Noa Margeta s232470 `<br/>`
Ana Marija Pavičić s232468 `<br/>`
Filip Penzar s232452 `<br/>`

### Have you ever wondered if machines find you attractive?

We have found ourselves asking the same question and decided to find the answer through this project.

## Project information

This is a group project for the DTU course Machine Learning Operations [DTU:02476](https://skaftenicki.github.io/dtu_mlops/projects/).

### Goal

Our goal was to create and deploy a model that can make inference on wheteher a person is attractive or not. The main focus of the project was not the accuracy of the model itself, but the operations part around the ML model necessary to build and deploy it. (So don't take the predictions too seriosly)

### Data

We got our data from the Kaggle dataset [CelebFaces Attributes](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data). It consists of over 200k images of celebrities with 40 binary attribute annotations, but we only chose to analyze the "Attractive" attribute.

### Model

We used [PyTorch](https://pytorch.org/) for the development. More precisely, we used the [Transformer framework](https://github.com/huggingface/transformers) from huggingface and decided to fine-tune the [Vision Transformer(ViT)](https://huggingface.co/docs/transformers/model_doc/vit), which was pretrained on 14 million images and can classify 1000 different items.

## Running locally

To run the model locally you will need to fetch the model from the Google Cloud Platform. You will need to contact nmarg@gmail.com to give you access to the Google Bucket.

After you have downloaded the model (and placed it under `models/model0`) you have to build the Docker image:
```
docker build -f dockerfiles/server.dockerfile -t server_model_image .
```
And then run the container:
```
docker run --name server server_model_image:latest
```

You can access the server:

```
curl -X 'POST' \
        'http://localhost:8080/predict/' \
        -H 'accept: application/json' \
        -H 'Content-Type: multipart/form-data' \
        -F 'data=@/path/to/your/image.jpg;type=image/jpeg'
```

Which will give you back the model's inference on the provided image.

## Project structure

The project structure was generated using the [cookiecutter data science template](https://github.com/drivendata/cookiecutter-data-science).
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
