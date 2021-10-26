# LOADING LIBRARIES
import pandas as pd
import yaml
import pickle
import os.path
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from datetime import datetime
from time import process_time

# DATASETS (DATABASES) LIST
# If you add a new dataset/database, add its name here (without suffixes).
datasetsFile = open('../datasets.txt', 'r')
datasets = datasetsFile.read().splitlines()

for dataset in datasets:

    # INITIALIZATION OF CART ALGORITHM WITH THE DEFAULT SETTINGS
    cart = DecisionTreeClassifier()
    name = "CART"
    filename = "cart_" + dataset
    serialization_filepath = "../out/cart/" + filename + "_tree.model"
    yaml_filepath = "../out/cart/" + filename + "_results.yml"

    # CHECKING IF A DECISION TREE IS ALREADY GENERATED
    if os.path.isfile(serialization_filepath) and os.path.isfile(yaml_filepath):
        # Loading a saved decision tree...
        infile = open(serialization_filepath, "rb")
        cart = pickle.load(infile)
        infile.close()
        with open(yaml_filepath, "r") as f:
            results = yaml.full_load(f)
    else:
        # Training a new decision tree...
        train_data = pd.read_csv("../datasets/" + dataset + "_trte.data", header=None)
        test_data = pd.read_csv("../datasets/" + dataset + "_clean.data", header=None)
        timer_start = process_time() 
        cart = cart.fit(train_data.iloc[:,:-1], train_data.iloc[:,-1])
        timer_stop = process_time()
        # Saving the model
        outfile = open(serialization_filepath, "wb")
        pickle.dump(cart, outfile)
        outfile.close()
        # Testing
        predicted = cart.predict(test_data.iloc[:,:-1])
        results = {
            "method": "CART",
            "database": dataset,
            "confusion matrix": metrics.confusion_matrix(test_data.iloc[:,-1], predicted).tolist(),
            "tree": {
                "size": cart.tree_.node_count,
                "height": cart.get_depth(),
                "serialization": serialization_filepath
            },
            "time of experiment": datetime.utcnow().strftime("%d-%m-%Y %H:%M:%S"),
            "cpu time": timer_stop - timer_start
        }
        yaml_output = yaml.dump(results)
        with open(yaml_filepath, "w") as f:
            f.write(yaml_output)
