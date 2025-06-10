import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Set up system path
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import relevant packages
from src.evaluation import StatisticalEval

def main():
    # Initialize the evaluation class
    eval = StatisticalEval()

    # Run the evaluation
    eval(metric_used="AUC", n_runs=50)

if __name__ == "__main__":
    main()