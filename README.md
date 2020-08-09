# Adversarial Examples for Models of Code - GNN

An adversary for graph neural networks (GNNs) with feature-wise linear modulation ([Brockschmidt, 2019](#brockschmidt-2019)).
This is an official implemention of the model described in:

Noam Yefet, [Uri Alon](http://urialon.cswp.cs.technion.ac.il) and [Eran Yahav](http://www.cs.technion.ac.il/~yahave/),
"Adversarial Examples for Models of Code", 2019 
https://arxiv.org/abs/1910.07517

The adversary implemented on five model types while running the VarMisuse task:
* Gated Graph Neural Networks (GGNN) ([Li et al., 2015](#li-et-al-2015)).
* Relational Graph Convolutional Networks (RGCN) ([Schlichtkrull et al., 2016](#schlichtkrull-et-al-2016)).
* Relational Graph Attention Networks (RGAT) - a generalisation of Graph Attention Networks ([Veličković et al., 2018](#veličković-et-al-2018)) to several edge types.
* Graph Neural Network with Edge MLPs (GNN-Edge-MLP) - a variant of RGCN in which messages on edges are computed using full MLPs, not just a single layer.
* Relational Graph Dynamic Convolution Networks (RGDCN) - a new variant of RGCN in which the weights of convolutional layers are dynamically computed.
* Graph Neural Networks with Feature-wise Linear Modulation (GNN-FiLM) - a new extension of RGCN with FiLM layers.

Table of Contents
=================
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
  * [Configuration](#configuration)

## Requirements
On Ubuntu:
  * [Python3](https://www.linuxbabe.com/ubuntu/install-python-3-6-ubuntu-16-04-16-10-17-04). To check if you have it:
> python3 --version
  * TensorFlow - version 1.13.1 or newer ([install](https://www.tensorflow.org/install/install_linux)). To check TensorFlow version:
> python3 -c 'import tensorflow as tf; print(tf.\_\_version\_\_)'
  * For [creating a new dataset](#creating-and-preprocessing-a-new-java-dataset) - [Java JDK](https://openjdk.java.net/install/)

## Quickstart

### Step 0: Cloning this repository and switch branch
```
git clone https://github.com/noamyft/tf-gnn-samples.git
cd tf-gnn-samples
git checkout adversary
```

### Step 1: Download dataset 
We provided a sub-data of the VarMisuse task (the data we used for evalutaion). You can download it from [here](https://drive.google.com/file/d/1SARyWiRl8CWVcHmdiCshAHiEwFNoJQ1D/view?usp=sharing)

Then run the following commands:
```
mdkir data
cd data
tar -xzf ../varmisuse_small.tar.gz
```

Alternatively, You can use the entire VarMisuse dataset. Please follow the instruction under "VarMisuse" section in https://github.com/microsoft/tf-gnn-samples

### Step 2: Downloading a trained models
we provide pretrained models for GNN & GNN-FiLM. You can download them from [here](https://drive.google.com/file/d/1GqANMlsnRpzYcXCNzkS2VnKfD3Im3yjN/view?usp=sharing)
Then run the following commands:
```
mdkir trained_models
cd trained_models
```
and unzip the file in trained_models.

### Step 3: Run adversary on the trained model

Once you download the preprocessed datasets and pretrained model - you can run the adversary on the model, by run:

```
python3 code2vec.py trained_models/VarMisuse_GGNN_2019-09-23-17-42-12_23483_best_model.pickle data/varmisuse_small/graphs-testonly
```
**note:** the adversary may take some time to run (even on GPU).

### Configuration

You can change hyper-parameters by set the following Variables in models/sparse_graph_model.py:
* _TARGETED_ATTACK_ - set the type of attack (True for targeted, false otherwise).
* _ADVERSARIAL_DEPTH_ - the BFS search's depth (3 by default).
