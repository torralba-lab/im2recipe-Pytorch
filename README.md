# im2recipe: Learning Cross-modal Embeddings for Cooking Recipes and Food Images

This repository contains the code to train and evaluate models from the paper:  
_Learning Cross-modal Embeddings for Cooking Recipes and Food Images_

Important note: In this repository the Skip-instructions has not been reimplemented in Pytorch, instead needed features are provided to train, validate and test the tri_joint model.

Clone it using:

```shell
git clone --recursive https://github.com/torralba-lab/im2recipe-Pytorch.git
```

If you find this code useful, please consider citing:

```
@inproceedings{salvador2017learning,
  title={Learning Cross-modal Embeddings for Cooking Recipes and Food Images},
  author={Salvador, Amaia and Hynes, Nicholas and Aytar, Yusuf and Marin, Javier and 
          Ofli, Ferda and Weber, Ingmar and Torralba, Antonio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
```

## Contents
1. [Installation](#installation)
2. [Recipe1M Dataset](#recipe1m-dataset)
3. [Vision models](#vision-models)
4. [Out-of-the-box training](#out-of-the-box-training)
5. [Prepare training data](#prepare-training-data)
6. [Training](#training)
7. [Testing](#testing)
8. [Pretrained model](#pretrained-model)
9. [Contact](#contact)

## Installation

Install [PyTorch](http://pytorch.org/) following the official website start guide

We use Python2.7. Install dependencies with ```pip install -r requirements.txt```

## Recipe1M Dataset

Our Recipe1M dataset is available for download [here](http://im2recipe.csail.mit.edu/dataset/download).

## Vision models

This current version of the code uses a pre-trained ResNet-50.

## Out-of-the-box training

To train the model, you will need the following files:
* `data/train_lmdb`: LMDB (training) containing skip-instructions vectors, ingredient ids and categories.
* `data/train_keys`: pickle (training) file containing skip-instructions vectors, ingredient ids and categories.
* `data/val_lmdb`: LMDB (validation) containing skip-instructions vectors, ingredient ids and categories.
* `data/val_keys`: pickle (validation) file containing skip-instructions vectors, ingredient ids and categories.
* `data/test_lmdb`: LMDB (testing) containing skip-instructions vectors, ingredient ids and categories.
* `data/test_keys`: pickle (testing) file containing skip-instructions vectors, ingredient ids and categories.
* `data/text/vocab.bin`: ingredient Word2Vec vocabulary. Used during training to select word2vec vectors given ingredient ids.

The links to download them are available [here](http://im2recipe.csail.mit.edu/dataset/download). LMDBs and pickle files can be found in train.tar, val.tar and test.tar. 

## Prepare training data

We also provide the steps to format and prepare Recipe1M data for training the trijoint model. We hope these instructions will allow others to train similar models with other data sources as well.

### Choosing semantic categories

We provide the script we used to extract semantic categories from bigrams in recipe titles:

- Run ```python bigrams --crtbgrs```. This will save to disk all bigrams in the corpus of all recipe titles in the training set, sorted by frequency.
- Running the same script with ```--nocrtbgrs``` will create class labels from those bigrams adding food101 categories.

These steps will create a file called ```classes1M.pkl``` in ```./data/``` that will be used later to create the LMDB file including categories.

### Word2Vec

Training word2vec with recipe data:

- Run ```python tokenize_instructions.py train``` to create a single file with all training recipe text.
- Run the same ```python tokenize_instructions.py``` to generate the same file with data for all partitions (needed for skip-thoughts later).
- Download and compile [word2vec](https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip)
- Train with:

```
./word2vec -hs 1 -negative 0 -window 10 -cbow 0 -iter 10 -size 300 -binary 1 -min-count 10 -threads 20 -train tokenized_instructions_train.txt -output vocab.bin
```

- Run ```python get_vocab.py vocab.bin``` to extract dictionary entries from the w2v binary file. This script will save ```vocab.txt```, which will be used to create the dataset later.
- Move ```vocab.bin``` and ```vocab.txt``` to ```./data/text/```.

### Skip-instructions (Torch)

In this repository the Skip-instructions is not implemented in Pytorch, instead we provide the necessary files to train, validate and test tri_joint model. 

### Creating LMDB file

Navigate back to ```./```. Run the following from ```./scripts```:

```
python mk_dataset.py 
--vocab /path/to/w2v/vocab.txt 
--sthdir /path/to/skip-instr_files/
```

## Training

- Train the model with: 
```
python train.py 
--img_path /path/to/images/ 
--data_path /path/to/lmdbs/ 
--ingrW2V /path/to/w2v/vocab.bin
--snapshots snapshots/
--valfreq 10
```

*Note: Again, this can be run without arguments with default parameters if files are in the default location.*

- You can set ```-batchSize``` to ~160. This is the default config, which will make the model converge in less than 3 days. Pytorch version requires less memory. You should be able to train the model using two TITAN X 12gb with same batch size. In this version we are using LMDBs to load the instructions and ingredients instead of a single HDF5 file.

## Testing

- Extract features from test set ```python test.py --model_path=snapshots/model*.tar```. They will be saved in ```results```.
- After feature extraction, compute MedR and recall scores with ```python scripts/rank.py --path_results=results```.

## Pretrained model

Our best model can be downloaded [here](http://data.csail.mit.edu/im2recipe/model_e220_v-4.700.pth.tar).
You can test it with:
```
python test.py --model_path=snapshots/model_e220_v-4.700.pth.tar
```

## Contact
#lalla
For any questions or suggestions you can use the issues section or reach us at jmarin@csail.mit.edu or amaia.salvador@upc.edu.
