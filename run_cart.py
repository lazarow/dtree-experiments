# LOAD LIBRARIES BEGIN
import pandas as pd
import yaml
import pickle
import os.path
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from datetime import datetime
from time import process_time
# LOAD LIBRARIES END

# DATASETS (DATABASES) LIST
# If you add a new dataset/database, add its name here (without suffixes).
datasets = [
    "breast-cancer-wisconsin",
    "dermatology",
    "house-votes-84",
    "lymphography",
    "monks-1",
    "SouthGermanCredit",
    "soybean-large",
    "tic-tac-toe",
    "zoo"
]

# RANGE OF MAX DEPTH VALUES
depths = [2, 3, 4, 5, 6]

for dataset in datasets:
    for max_depth in depths:

        # INITIALIZATION OF CART ALGORITHM WITH A SPECIFIC MAX DEPTH
        cart = DecisionTreeClassifier(max_depth = max_depth)
        name = "CART (criterion=" + cart.criterion + ", splitter=" + cart.splitter + ", max depth=" + str(cart.max_depth) + ")"
        filename = "cart_" + cart.criterion + "_" + cart.splitter + "_" + str(cart.max_depth) + "_" + dataset
        serialization_filepath = "output/trees/" + filename + ".dump"
        yaml_filepath = "output/yaml/" + filename + ".yml"

        # CHECKIN IF A DECISION TREE IS ALREADY GENERATED
        if os.path.isfile(serialization_filepath) and os.path.isfile(yaml_filepath):
            # Loading a saved decision tree...
            infile = open(serialization_filepath, "rb")
            cart = pickle.load(infile)
            infile.close()
        else:
            # Training a new decision tree...
            train_data = pd.read_csv("datasets/" + dataset + "_trte.data", header=None)
            test_data = pd.read_csv("datasets/" + dataset + "_clean.data", header=None)
            timer_start = process_time() 
            cart = cart.fit(train_data.iloc[:,:-1], train_data.iloc[:,-1])
            timer_stop = process_time()
            # Saving the model
            outfile = open(serialization_filepath, "wb")
            pickle.dump(cart, outfile)
            outfile.close()
            # Testing
            predicted = cart.predict(test_data.iloc[:,:-1])
            yaml_output = yaml.dump({
                "name": "CART (criterion=" + cart.criterion + ", splitter=" + cart.splitter + ", max depth=" + str(cart.max_depth) + ")",
                "database": dataset,
                "confusion matrix": metrics.confusion_matrix(test_data.iloc[:,-1], predicted).tolist(),
                "tree": {
                    "size": cart.tree_.node_count,
                    "height": cart.get_depth(),
                    "serialization": serialization_filepath
                },
                "time of experiment": datetime.utcnow().strftime("%d-%m-%Y %H:%M:%S"),
                "cpu time": timer_stop - timer_start,
                "metrics": {
                    "accuracy": float(metrics.accuracy_score(test_data.iloc[:,-1], predicted)),
                    "precision": float(metrics.precision_score(test_data.iloc[:,-1], predicted, average='micro')),
                    "recall": float(metrics.recall_score(test_data.iloc[:,-1], predicted, average='micro')),
                    "f1": float(metrics.f1_score(test_data.iloc[:,-1], predicted, average='micro'))
                }
            })
            with open(yaml_filepath, "w") as f:
                f.write(yaml_output)
