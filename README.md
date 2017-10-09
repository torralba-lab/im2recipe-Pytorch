# im2recipe: Learning Cross-modal Embeddings for Cooking Recipes and Food Images

This repository contains the code to train and evaluate models from the paper:  
_Learning Cross-modal Embeddings for Cooking Recipes and Food Images_

Important note: 

Clone it using:

```shell
git clone --recursive https://github.com/torralba-lab/im2recipe-pytorch.git
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
8. [Visualization](#visualization)
9. [Pretrained model](#pretrained-model)
10. [Contact](#contact)

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

The links to download them are available [here](http://im2recipe.csail.mit.edu/dataset/download).

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

In this repository the Skip-instructions is not implemented in Pytorch, instead we provide the needed features to train, validate and test tri_joint model. 

### Creating LMDB file

Navigate back to ```./```. Run the following from ```./scripts```:

```
python mk_dataset.py 
-vocab /path/to/w2v/vocab.txt 
-dataset /path/to/recipe1M/ 
-stvecs /path/to/skip-instr_files/
```

## Training

- Train the model with: 
```
python train.py 
--data_path /path/to/lmdb/files/ 
--ingrW2V /path/to/w2v/vocab.bin
--snapfile snapshots/
--valfreq 10
```

*Note: Again, this can be run without arguments with default parameters if files are in the default location.*

- You can use multiple GPUs to train the model with the ```-ngpus``` flag. With 4 GTX Titan X you can set ```-batchSize``` to ~150. This is the default config, which will make the model converge in about 3 days.
- Plot loss curves anytime with ```python plotcurve.py -logfile /path/to/logfile.txt```. If ```dispfreq``` and ```valfreq``` are different than default, they need to be passed as arguments to this script for the curves to be correctly displayed. Running this script will also give you the elapsed training time. ```logifle.txt``` should contain the stdout of ```main.lua```. Redirect it with ```th main.lua > /path/to/logfile.txt ```.

## Testing

- Extract features from test set ```python test.py --model_path=snapshots/model*.tar```. They will be saved in ```results```.
- After feature extraction, compute MedR and recall scores with ```python scripts/rank.py --path_results=results```.

## Visualization

We provide a script to visualize top-1 im2recipe examples in ```./pyscripts/vis.py ```. It will save figures under ```./data/figs/```.

## Pretrained model

Our best model can be downloaded [here](http://data.csail.mit.edu/im2recipe/im2recipe_model.t7.gz).
You can test it with:
```
th main.lua -test 1 -loadsnap im2recipe_model.t7
test.py --loadsnap im2recipe_model.t7
```

## Contact

For any questions or suggestions you can use the issues section or reach us at jmarin@csail.mit.edu or amaia.salvador@upc.edu.
