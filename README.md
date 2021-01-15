# Stochastic variance reduced gradient methods for training DeepNeural Networks

*by Alexander Apostolov*
*Advised by: Dr. Tatjana Chavdarova and Prof. Dr. Martin Jaggi*

This project focuses on getting empirical insights on why stochstic variance reduced methods who work better than other stochastic methods such as SGD or ADAM theoretically and on small neural networks do have worse results on deep neural networks.

## Implemented algorithms:
- **SVRG**<br/>
Main stochastic variance reduced method studied in this project. [Original paper](https://papers.nips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html)
- **STORM**<br/>
Another stochastic variance reduced method. [Original paper](https://arxiv.org/abs/1905.10018)
- **SGD**<br/>
- **Adam**<br/>
[Original paper](https://arxiv.org/abs/1412.6980#:~:text=We%20introduce%20Adam%2C%20an%20algorithm,estimates%20of%20lower%2Dorder%20moments.)
- **AdaGrad**<br/>
[Original paper](https://jmlr.org/papers/v12/duchi11a.html)

## Repository structure

- `report/` files to create and compile the report as well as the midterm presentation as well as PDF versions of these documents.
- `src/` Code that was used for this project. Further details in [this section](#details-of-the-code) and explanation on how to use it in [this section](#how-to-run-experiments).

## Details of the code

- `crossvalidation.py`<br/>
Contains methods to perform cross-validation on the different optimization algorithms to select their hyperparameters.
- `data_helpers.py`<br/>
Contains methods to save and load results from experiments.
- `metaInit.py`<br/>
Contains the code from the original paper of the weight initialization algorithmn [MetaInit](https://papers.nips.cc/paper/2019/hash/876e8108f87eb61877c6263228b67256-Abstract.html).
- `networks.py`<br/>
Contains the code for the different neural networks used in this project and their variants (LeNet, ResNet18 and ResNet101)
- `storm_optim.py`<br/>
Contains an implementation of the STORM algorithm.
- `svrg.py`<br/>
Contains an implementation of the SVRG algorithm.
- `train_optimizer.py`<br/>
Contains methods to train already implemented optimizers.
- `training.py`<br/>
Contains the main methods used to run the experiments with all algorithms.
- `training_helpers.py`<br/>
Contains helper functions to calculate accuracy and metrics on the test set during training.
- `visualisation_helpers.py`<br/>
Contains methods to vizualize the results from the experiments.

## How to run experiments
A jupyter notebook run on Google colab is given as an example [here](https://colab.research.google.com/drive/1RuFOIbbalhViiaWPK3xN8Qs-3dGmqMmW?usp=sharing
). To make it work you need to clone the repository (at least the `src` forlder) in a google drive that also contains two folders `saved` and `data`. More explanation is given in the jupyter notebook.

