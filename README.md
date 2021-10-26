# Decision Tree Experiments

The repository contains 3 runners:
- C4.5 (from Weka);
- CART (from sklearn);
- EVO-Tree,
and 9 datasets for experiments, which are declared in the `datasets.txt` file.

## Weka's C4.5

In order to run experiments with the C4.5 algorithm you have to install `JDK` and `maven` (add it to the `PATH` to have the `mvn` command).

### Verify required software

```
javac --version
mvn --version
```

### Run

```
cd weka
mvn compile
```

The above command will download all required libaries and conduct experiment. The output will be saved in the `out` folder.

## CART

## How to run the code?

### Step 0
_(Optional)_ Installation of the `virtualenv` if you don't have it already. Install it globally.
```
pip install virtualenv
```

### Step 1
Create an virtual environment with `virtualenv`.
```
cd cart
virtualenv .
```
Some folders and files will be created. Now, activate the virtual environment.
```
source bin/activate     # for linux
Scripts\activate.bat    # for windows
```

### Step 2
Install required libraries. It has to be done once only, the environment will store downloaded libraries.
```
pip install -r requirements.txt
```

### Step 3
Run the tests.
```
python run.py
```
The output will be saved in the `out` folder.

