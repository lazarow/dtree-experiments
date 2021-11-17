# Decision Tree Experiments

The repository contains 3 runners:
- C4.5 (from Weka);
- CART (from sklearn);
- EVO-Tree (implementation based on [the article _Evolutionary Algorithm for Decision Tree Induction_](https://hal.inria.fr/hal-01405549/document) and [this repository](https://github.com/jamesnzeex/Decision_tree_with_genetic_algorithm/blob/main/Decision_tree_with_genetic_algorithm.ipynb)),

and several datasets for experiments, which are declared in the `datasets.txt` file.

The trainde models as well as results in the form of YAML file are saved in the `out` folder.

## Weka's C4.5

In order to run experiments with the C4.5 algorithm you have to install `JDK` and `maven` (add it to the `PATH` to have the `mvn` command).

```
# verify required software
javac --version
mvn --version

# run
cd weka
mvn compile
```

## CART

In order to run experiments with the CART algorithm you have to install `python`.

```
# verify required software
python --version
pip --version

# optional installation of virtualenv
pip install virtualenv

# run
cd cart
virtualenv .
rm .gitignore           # required as virtualenv add it automatically
source bin/activate     # for linux
Scripts\activate.bat    # for windows
pip install -r requirements.txt
python run.py
```

## EVO-Tree

In order to run experiments with the EVO-Tree algorithm you have to install `python`.

```
# verify required software
python --version
pip --version

# optional installation of virtualenv
pip install virtualenv

# run
cd evo-tree
virtualenv .
rm .gitignore           # required as virtualenv add it automatically
source bin/activate     # for linux
Scripts\activate.bat    # for windows
pip install -r requirements.txt
python run.py
```
