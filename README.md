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
@article{marin2019learning,
  title = {Recipe1M+: A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images},
  author = {Marin, Javier and Biswas, Aritro and Ofli, Ferda and Hynes, Nicholas and 
  Salvador, Amaia and Aytar, Yusuf and Weber, Ingmar and Torralba, Antonio},
  journal = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  year = {2019}
}

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
2. [Recipe1M Dataset](#recipe1m-and-recipe1m-datasets)
3. [Vision models](#vision-models)
4. [Out-of-the-box training](#out-of-the-box-training)
5. [Prepare training data](#prepare-training-data)
6. [Training](#training)
7. [Testing](#testing)
8. [Pretrained model](#pretrained-model)
9. [Recipes with nutritional info](#recipes-with-nutritional-info)
10. [Contact](#contact)

## Installation

```
docker build -t im2recipe .
docker run -it im2recipe
```
You may use a volume to give snapshots, data, etc to the docker container.

If you are not using Docker, we do recommend to create a new environment with Python 3.7. Right after it, run ```pip install --upgrade cython``` and then install the dependencies with ```pip install -r requirements.txt```. Notice that this will install the latest PyTorch version available. Once you finish, you will need to install [torchwordemb](https://github.com/iamalbert/pytorch-wordemb). In order to do that (or at least the way we found it worked for us), we downloaded and installed it via ```python setup.py install```. In case you get an error related to  ```return {vocab, dest};```, you just need to change the original code to ```return VocabAndTensor(vocab, dest);```, and run ```python setup.py install``` again.

## Recipe1M and Recipe1M+ Datasets

In order to get access to the dataset, please fill the following form [here](https://forms.gle/EzYSu8j3D1LJzVbR8).

## Vision models

This current version of the code uses a pre-trained ResNet-50.

## Out-of-the-box training

To train the model, you will need to create following files:
* `data/train_lmdb`: LMDB (training) containing skip-instructions vectors, ingredient ids and categories.
* `data/train_keys`: pickle (training) file containing skip-instructions vectors, ingredient ids and categories.
* `data/val_lmdb`: LMDB (validation) containing skip-instructions vectors, ingredient ids and categories.
* `data/val_keys`: pickle (validation) file containing skip-instructions vectors, ingredient ids and categories.
* `data/test_lmdb`: LMDB (testing) containing skip-instructions vectors, ingredient ids and categories.
* `data/test_keys`: pickle (testing) file containing skip-instructions vectors, ingredient ids and categories.
* `data/text/vocab.txt`: file containing all the vocabulary found within the recipes.

And download the following ones: 
* `data/text/vocab.bin`: ingredient Word2Vec vocabulary. Used during training to select word2vec vectors given ingredient ids.
* `data/food101_classes_renamed.txt`: Food101 classes used to create the bigrams.
* `data/encs_train_1024.t7`: Skip-instructions train partition.
* `data/encs_val_1024.t7`: Skip-instructions val partition.
* `data/encs_test_1024.t7`: Skip-instructions test partition.
* `data/recipe1M/layer2+.json`: Recipe1M+ layer2.
* `data/images/Recipe1M+_{a..f}.tar`: 6 Tar files containing part of the images available in Recipe1M+ (~210Gb each).
* `data/images/Recipe1M+_{0..9}.tar`: 10 Tar files containing part of the images available in Recipe1M+ (~210Gb each).

To download these files, you first need to complete the following [form](https://forms.gle/EzYSu8j3D1LJzVbR8). After submission, we will share the necessary download links via email. Access to the dataset is granted only for research purposes to universities and research institutions. Original Recipe1M LMDBs and pickle files can be found in train.tar, val.tar and test.tar.

It is worth mentioning that the code is expecting images to be located in a four-level folder structure, e.g. image named `0fa8309c13.jpg` can be found in `./data/images/0/f/a/8/0fa8309c13.jpg`. Each one of the Tar files contains the first folder level, 16 in total. If you do not have enough space after downloading the Tar files, you can try to mount them locally and access them. We did use [ratarmount](https://github.com/mxmlnkn/ratarmount) in our latest test experiments. In order to properly access the images with ratarmount, we temporarily changed our code. We basically tried up to three times to load an image within our `default_loader`.

## Prepare training data

We also provide the steps to format and prepare Recipe1M/Recipe1M+ data for training the trijoint model. We hope these instructions will allow others to train similar models with other data sources as well.

### Choosing semantic categories

We provide the script we used to extract semantic categories from bigrams in recipe titles:

- Run ```python bigrams --crtbgrs```. This will save to disk all bigrams in the corpus of all recipe titles in the training set, sorted by frequency. Note that you will need to create first ```vocab.txt``` running ```python get_vocab.py ../data/vocab.bin``` within ```./scripts/```.
- Running the same script again with ```--nocrtbgrs``` will create class labels from those bigrams adding food101 categories.

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
Notice, that layer2 within ```./data/recipe1M/layer2.json``` will need to be replaced by layer2+.json in order to create our extended Recipe1M+ dataset.

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

Our best model trained with Recipe1M+ (journal extension) can be downloaded [here](http://data.csail.mit.edu/im2recipe/model_e500_v-8.950.pth.tar).

You can test it with:
```
python test.py --model_path=snapshots/model_e500_v-8.950.pth.tar
```
Our best model trained with Recipe1M (CVPR paper) can be downloaded [here](http://data.csail.mit.edu/im2recipe/model_e220_v-4.700.pth.tar).

## Recipes with nutritional info

We also provide a subset of recipes with nutritional information. Below you can see an example:
```
{'fsa_lights_per100g': {'fat': 'green',
  'salt': 'green',
  'saturates': 'green',
  'sugars': 'orange'},
 'id': '000095fc1d',
 'ingredients': [{'text': 'yogurt, greek, plain, nonfat'},
  {'text': 'strawberries, raw'},
  {'text': 'cereals ready-to-eat, granola, homemade'}],
 'instructions': [{'text': 'Layer all ingredients in a serving dish.'}],
 'nutr_per_ingredient': [{'fat': 0.8845044000000001,
   'nrg': 133.80964,
   'pro': 23.110512399999998,
   'sat': 0.26535132,
   'sod': 81.64656,
   'sug': 7.348190400000001},
  {'fat': 0.46,
   'nrg': 49.0,
   'pro': 1.02,
   'sat': 0.023,
   'sod': 2.0,
   'sug': 7.43},
  {'fat': 7.415,
   'nrg': 149.25,
   'pro': 4.17,
   'sat': 1.207,
   'sod': 8.0,
   'sug': 6.04}],
 'nutr_values_per100g': {'energy': 81.12946131894766,
  'fat': 2.140139263515891,
  'protein': 6.914436593565536,
  'salt': 0.05597816738985967,
  'saturates': 0.36534716195613937,
  'sugars': 5.08634103436144},
 'partition': 'train',
 'quantity': [{'text': '8'}, {'text': '1'}, {'text': '1/4'}],
 'title': 'Yogurt Parfaits',
 'unit': [{'text': 'ounce'}, {'text': 'cup'}, {'text': 'cup'}],
 'url': 'http://tastykitchen.com/recipes/breakfastbrunch/yogurt-parfaits/',
 'weight_per_ingr': [226.796, 152.0, 30.5]}
```
Note that these recipes include the matched ingredients from USDA instead of the original ones. There are 35,867 recipes for training, 7,687 for validation and 7,681 for testing. In order to obtain the grams of salt, we multiplied the sodium by 2.5 and divided it by 1000. Total weight per ingredient, fat, proteins/pro, salt, saturates/sat and sugars/sug are expressed in grams. Sodium/sod is expressed in mg and energy/nrg in kcal. FSA traffic lights are also included per 100g.

## Contact

For any questions or suggestions you can use the issues section or reach us at jmarin@csail.mit.edu.
