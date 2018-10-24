# random_forest_regressor
Implementation of a random forest regressor with hard-coded feature engineering for a real-estate housing price dataset using scikit-learn.

Usage: 
  Run re_forest.py with no arguments for default values:
    - random_state = None
    - num_trees = 100
    - max_depth = None
    - print interior data = False
    
  Run realestate_forest.py with hyperparameters:
    - python3 re_forest.py {int} {int} {int} {Bool}
    - python3 re_forest.py 5 250 12 True
    
  Data:
    - test.csv
    - train.csv
