# Import relevant packages
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns
import scipy.stats as stats
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Config
pd.set_option('display.max_columns', None) # Ensure all columns are displayed
warnings.filterwarnings("ignore")

# ----------------------------------------
# Function to detect variable types
# ----------------------------------------

def data_info(df):
    """
    Detects whether each column in the DataFrame is categorical or numerical.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    list: A list of dictionaries containing variable names, types, statistical summaries, and missing values.
    """
    variable_info = []
    
    for col in df.columns:
        var_dict = {"var_name": col}
        
        # Identify whether the variable is numerical or categorical
        if pd.api.types.is_numeric_dtype(df[col]):
            var_dict["var_type"] = 'numerical'
            # Compute summary statistics for numerical variables
            var_dict["mean"] = df[col].mean()
            var_dict["median"] = df[col].median()
            var_dict["std_dev"] = df[col].var()**0.5
            var_dict["skewness"] = df[col].skew()
            var_dict["kurtosis"] = df[col].kurt()
            var_dict["min"] = df[col].min()
            var_dict["max"] = df[col].max()
        else:
            if set(df[col].dropna().unique()) == set(["yes", "no"]):
                var_dict["var_type"] = 'binary'
            else:
                var_dict["var_type"] = 'categorical'
            var_dict["num_unique"] = df[col].nunique()
            var_dict["unique_list"] = df[col].unique().astype(str)
            var_dict["mode"] = df[col].mode()[0]

        # Store data type
        var_dict["data_type"] = df[col].dtype

        # Count missing values
        var_dict['missing'] = df[col].isnull().sum()

        variable_info.append(var_dict)

    var_info = pd.DataFrame(variable_info, columns=['var_name', 'var_type', 'data_type'
                                                                 , 'missing', 'mean', 'median', 'mode' 
                                                                 , 'std_dev', 'skewness', 'kurtosis'
                                                                 , 'min', 'max'
                                                                 , 'num_unique', 'unique_list'
                                                                 ])
    
    return var_info

# ----------------------------------------
# Class for Data Distribution Visualization
# ----------------------------------------

class VisualDistr:
    """
    Class for visualizing the distributions of numerical and categorical variables.
    """

    def __init__(self, data, categorical_vars=None, numerical_variables=None):
        self.data = data
        self.categorical_vars = categorical_vars
        self.numerical_variables = numerical_variables

    def plot_numerical_distributions(self, num_cols=5, bins=30, kde=True):
        """
        Plots histograms of numerical variables with optional KDE overlay.
        
        Parameters:
        num_cols (int): Number of columns in the subplot layout.
        bins (int): Number of bins for histogram.
        kde (bool): Whether to overlay a KDE plot.

        Returns:
        module: The set of histograms that present the distributions of numerical variables.
        """
        num_vars = len(self.numerical_variables)
        num_rows = num_vars // num_cols + 1  # Dynamically calculate rows
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))
        axes = axes.flatten()
        
        for i, variable in enumerate(self.numerical_variables):
            if variable not in self.data.columns:
                raise ValueError(f"Column '{variable}' not found in DataFrame.")
            
            var_data = self.data[variable].dropna()  # Remove missing values
            
            if not pd.api.types.is_numeric_dtype(var_data):
                raise ValueError(f"Column '{variable}' must be numeric.")
            
            ax = axes[i]
            sns.histplot(var_data, bins=bins, kde=False, color='#1f78b4', edgecolor='black', label="Histogram", stat='density', ax=ax)
            
            if kde:
                sns.kdeplot(var_data, color='black', linewidth=1, label="KDE", ax=ax)
            
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title(f"{variable}")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()

        return plt

    def plot_categorical_distributions(self, num_cols=5, label=None):
        """
        Plots bar plots of categorical variables.
        
        Parameters:
        num_cols (int): Number of columns in the subplot layout.
        label (str, optional): Target label column name, if any (excluded from plots).

        Returns:
        module: The set of histograms that present the distributions of categorical variables.
        """

        if label and label in self.categorical_vars:
            self.categorical_vars = [var for var in self.categorical_vars if var != label]

        num_vars = len(self.categorical_vars)
        num_rows = num_vars // num_cols + 1  # Dynamically calculate rows
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))
        axes = axes.flatten()

        for i, variable in enumerate(self.categorical_vars):
            if variable not in self.data.columns:
                raise ValueError(f"Column '{variable}' not found in DataFrame.")

            var_data = self.data[variable].dropna()  # Remove NaNs
            
            ax = axes[i]
            sns.countplot(y=var_data, order=var_data.value_counts().index, palette="Paired", ax=ax)
            
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title(f"{variable}")
            ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()

        return plt

    def compute_categorical_distributions(self):
        """
        Computes the distribution of categorical variables.

        Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - 'var_name': Name of the categorical variable.
            - 'value': Unique category.
            - 'count': Number of occurrences of the category.
            - 'pct': Proportion of occurrences relative to the total.
        """

        # Initialize an empty DataFrame with predefined columns
        categorical_distributions = pd.DataFrame(columns=["var_name", "value", "count", "pct"])

        # Iterate over each categorical variable
        for var in self.categorical_vars:
            # Compute value counts and reset index to convert it into a DataFrame
            var_distr = pd.DataFrame(self.data[var].value_counts()).reset_index()
            
            # Rename columns for clarity
            var_distr.columns = ["value", "count"]

            # Add the variable name as a new column
            var_distr["var_name"] = var
            
            # Compute the percentage of each category
            var_distr["pct"] = var_distr["count"] / var_distr["count"].sum()

            # Concatenate results to the main DataFrame
            categorical_distributions = pd.concat([categorical_distributions, var_distr], ignore_index=True)

        return categorical_distributions

# ----------------------------------------
# Class for Correlation Analysis
# ----------------------------------------

class VisualCorr:
    """
    Class for analyzing correlations between numerical and categorical variables.
    """

    def __init__(self, data, categorical_vars, numerical_variables):
        self.data = data
        self.categorical_vars = categorical_vars
        self.numerical_variables = numerical_variables

    def num_num_corr(self):
        """
        Plots a heatmap of correlations between numerical variables.

        Returns:
        module: The heatmap that presents Pearson's correlation between numerical variables.
        """
        corr_matrix = self.data[self.numerical_variables].corr()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)

        return plt

    def cat_num_corr(self):
        """
        Computes the Kruskal-Wallis test for categorical vs. numerical variable associations.
        Displays a heatmap of statistical significance.

        Returns:
        module: The heatmap that presents the p-values of the Kruskal-Wallis test for categorical vs. numerical variables.
        """
        kw_test = []
        for cat_col in self.categorical_vars:
            for num_col in self.numerical_variables:
                data = self.data[[cat_col, num_col]].dropna()
                if data[cat_col].nunique() > 1:
                    stat, p_value = stats.kruskal(*[data[data[cat_col] == category][num_col] for category in data[cat_col].unique()], nan_policy='propagate')
                    kw_test.append({'cat_col': cat_col, 'num_col': num_col, 'stat': stat, 'p_value': p_value})
        
        kw_test_df = pd.DataFrame(kw_test)
        kw_test_df = kw_test_df.pivot(index=['cat_col'], columns=['num_col'], values=['p_value'])
        kw_test_df.columns = kw_test_df.columns.droplevel()

        # Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(kw_test_df, cmap="coolwarm_r", annot=True, fmt=".2f", linewidths=0.5)
        plt.xlabel("")
        plt.ylabel("")

        return plt

    def cramers_v(self, x, y):
        """
        Computes Cramér's V statistic for categorical-categorical correlation.

        Returns:
        float: Cramér's V statistic
        """
        confusion_matrix = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        
        return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))
    
    def cat_cat_corr(self):
        """
        Computes and visualizes the correlation between categorical variables using Cramér’s V.

        Cramér’s V is a measure of association between two categorical variables, ranging from 0 to 1.
        - A value close to 0 indicates weak association.
        - A value close to 1 indicates a strong association.

        Returns:
        module: The heatmap that presents Cramér’s V correlation between categorical variables.
        """

        # Compute Cramér’s V for all categorical variables
        cramers_v_matrix = pd.DataFrame(index=self.categorical_vars, columns=self.categorical_vars)

        # Iterate over all pairs of categorical variables
        for col1 in self.categorical_vars:
            for col2 in self.categorical_vars:
                if col1 != col2:
                    cramers_v_matrix.loc[col1, col2] = self.cramers_v(self.data[col1], self.data[col2])

        # Convert to numeric for heatmap
        cramers_v_matrix = cramers_v_matrix.astype(float)

        # Heatmap visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(cramers_v_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.xlabel("")
        plt.ylabel("")

        return plt
    
# ----------------------------------------
# Function for t-SNE Visualization
# ----------------------------------------

def tsne_scatter(features, labels, dimensions=2):
    if dimensions not in (2, 3):
        raise ValueError('Make sure the dimensions parameter is 2 or 3')

    # t-SNE dimensionality reduction
    features_embedded = TSNE(n_components=dimensions, random_state=42).fit_transform(features)
    
    # Initialising the plot
    fig, ax = plt.subplots(figsize=(8,8))
    
    # Counting dimensions
    if dimensions == 3: ax = fig.add_subplot(111, projection='3d')

    # Plotting data
    ax.scatter(
        *zip(*features_embedded[np.where(labels==1)]),
        marker='o',
        color='r',
        s=2,
        alpha=0.7,
        label='Fraud'
    )
    ax.scatter(
        *zip(*features_embedded[np.where(labels==0)]),
        marker='o',
        color='g',
        s=2,
        alpha=0.3,
        label='Clean'
    )

    # Display the plot
    plt.legend(loc='best')
    
    return plt