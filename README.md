# CSE6250_Project

This repo intends to reproduce the work from this paper "Readmission prediction via deep contextual embedding of clinical concepts" by Cao Xiao et al. [link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0195024#pone.0195024.ref026)

## Data

A copy of the data is in the data directory labeled 'S1_File.txt', which is downloaded from [here](https://doi.org/10.1371/journal.pone.0195024.s001).

## Getting Started

Install the dependencies from requirements.txt. 

```
pip install -r requirements.txt
```

It is recommended to first create a virtual environment before installing dependencies.

## Data preparation

The following command will generate the readmission labels, and group the sequence of visits by date and patient id. It will save the cleansed data as a pickle file and split the data to training, validation, and test. The training, validaiton, and test data will also be saved as pickle files.

```
python create_dataset.py
```

## Create Embeddings

The following command will train a Word2Vec model based on the diagnosis descriptions (feature id) from the cleansed data. It will create the embedding vector for each patient EHR prior to the hospital discharge date. 

```
python embedding.py
```

## Models

The model class is in 

```
model.py
```

## Word2Vec + Logistic Regression Readmisison Prediction

word2vec_lr notebook has the training and evaluation results for the readmission model based on Word2Vec and logistic regression

## RNN Readmission Prediction

rnn_model notebook has the training and evaluation results for the RNN hospital readmissio model

## Hybrid TopicRNN (CONTENT) Readmission Prediction

content notebook has the training and evaluation results for the CONTENT hospital readmission model 

## Other scripts

The plotting funtion to plot the loss and accuracy curves are in 

```
plots.py
```

The utility function for the training script and Custom torch dataset are in 

```
utils.py
```
