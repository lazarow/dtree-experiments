# CART algorithm test

## How to run the code?

### Step 0
_(Optional)_ Installation of the `virtualenv` if you don't have it already. Install it globally.
```
pip install virtualenv
```

### Step 1
Clone this repository and step into a created folder.
```
git clone https://github.com/lazarow/dtree-tests-102021.git
cd dtree-tests-102021
```

### Step 2
Create an virtual environment with `virtualenv`.
```
virtualenv .
```
Some folders and files will be created. Now, activate the virtual environment.
```
source bin/activate     # for linux
Scripts\activate.bat    # for windows
```

### Step 3
Install required libraries. It has to be done once only, the environment will store downloaded libraries.
```
pip install -r requirements.txt
```

### Step 4
Run the tests.
```
python run_cart.py
```
The generated models will be saved into the `output/trees` folder. Beware, re-run of the scripts will use saved models.
In order to re-do all experiments, please clear the `output/trees` folder.

The results of testing of generated models are saved as YAML files places in the `output/yaml` folder. The example YAML result:
```
confusion matrix:
- - 67
  - 0
- - 33
  - 30
cpu time: 0.0
database: monks-1
metrics:
  accuracy: 0.7461538461538462
  f1: 0.7461538461538462
  precision: 0.7461538461538462
  recall: 0.7461538461538462
name: CART (criterion=gini, splitter=best, max depth=3)
time of experiment: 21-10-2021 14:19:22
tree:
  height: 3
  serialization: output/trees/cart_gini_best_3_monks-1.dump
  size: 9
```

### Clean up
Deactivate the virtual environment or just close the terminal.
```
source bin/deactivate     # for linux
Scripts\deactivate.bat    # for windows
```
