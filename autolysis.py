import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import json
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# AI Proxy imports
import requests

def convert_numpy_types(obj):
    """
    Convert NumPy types to standard Python types for JSON serialization
    
    :param obj: Object to convert
    :return: Converted object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

class EnhancedAutolysisAnalyzer:
    def __init__(self, csv_path: str):
        """
        Initialize the analyzer with the CSV file path
        
        :param csv_path: Path to the input CSV file
        """
        self.csv_path = csv_path
        self.df = None

        # Try reading the CSV with utf-8 encoding and fallback to other encodings
        try:
            self.df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.df = pd.read_csv(csv_path, encoding='latin1')
            except UnicodeDecodeError:
                try:
                    self.df = pd.read_csv(csv_path, encoding='ISO-8859-1')
                except Exception as e:
                    raise ValueError(f"Unable to read the file due to encoding issues: {e}")

        self.api_token = os.environ.get("AIPROXY_TOKEN")
        
        if not self.api_token:
            raise ValueError("AIPROXY_TOKEN environment variable not set")
        
        # Prepare analysis results storage
        self.analysis_results = {}
        self.charts = []
    
    def get_dataset_overview(self) -> Dict[str, Any]:
        """
        Generate a comprehensive overview of the dataset
        
        :return: Dictionary with dataset insights
        """
        overview = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "column_info": {},
            "missing_values": {},
            "data_distribution": {}
        }
        
        for column in self.df.columns:
            column_data = {
                "dtype": str(self.df[column].dtype),
                "non_null_count": int(self.df[column].count()),
                "unique_values": int(self.df[column].nunique())
            }
            
            # Check for missing values
            missing_count = self.df[column].isnull().sum()
            if missing_count > 0:
                overview["missing_values"][column] = {
                    "count": int(missing_count),
                    "percentage": float((missing_count / len(self.df)) * 100)
                }
            
            # Add distribution info for numeric columns
            if pd.api.types.is_numeric_dtype(self.df[column]):
                column_data["distribution"] = {
                    "skewness": float(self.df[column].skew()),
                    "kurtosis": float(self.df[column].kurtosis())
                }
            
            overview["column_info"][column] = column_data
        
        return overview
    
    def perform_advanced_analysis(self):
        """
        Perform advanced statistical and machine learning analyses, including outlier detection with scatter plots.
        """
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns

        if len(numeric_columns) > 1:  # Ensure at least two numeric columns for scatter plot
            # Prepare numeric data
            numeric_data = self.df[numeric_columns]
            
            # Select two numeric columns for scatter plot
            col_x, col_y = numeric_columns[:2]  # You can customize this or make it dynamic
            
            # Calculate IQR for outlier detection
            Q1_x, Q3_x = numeric_data[col_x].quantile(0.25), numeric_data[col_x].quantile(0.75)
            IQR_x = Q3_x - Q1_x
            lower_x, upper_x = Q1_x - 1.5 * IQR_x, Q3_x + 1.5 * IQR_x

            Q1_y, Q3_y = numeric_data[col_y].quantile(0.25), numeric_data[col_y].quantile(0.75)
            IQR_y = Q3_y - Q1_y
            lower_y, upper_y = Q1_y - 1.5 * IQR_y, Q3_y + 1.5 * IQR_y

            # Identify outliers
            outliers = numeric_data[
                (numeric_data[col_x] < lower_x) | (numeric_data[col_x] > upper_x) |
                (numeric_data[col_y] < lower_y) | (numeric_data[col_y] > upper_y)
            ]

            # Scatter plot with outliers highlighted
            plt.figure(figsize=(10, 6))
            plt.scatter(numeric_data[col_x], numeric_data[col_y], label='Normal Data', alpha=0.7)
            plt.scatter(outliers[col_x], outliers[col_y], color='red', label='Outliers', alpha=0.7)
            plt.title(f'Scatter Plot with Outliers: {col_x} vs {col_y}')
            plt.xlabel(col_x)
            plt.ylabel(col_y)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'scatter_outliers_{col_x}_vs_{col_y}.png')
            plt.close()
            self.charts.append(f'scatter_outliers_{col_x}_vs_{col_y}.png')
            
            # Store outlier information
            self.analysis_results['scatter_outliers'] = {
                "columns": [col_x, col_y],
                "outlier_count": len(outliers),
                "outlier_indices": outliers.index.tolist()
            }

        # 1. Enhanced Correlation Analysis
        if len(numeric_columns) > 0:
            correlation_matrix = numeric_data.corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, 
                        cbar_kws={'label': 'Correlation Coefficient'})
            plt.title('Advanced Correlation Heatmap')
            plt.tight_layout()
            plt.savefig('advanced_correlation_heatmap.png')
            plt.close()
            self.charts.append('advanced_correlation_heatmap.png')
            
            # 2. Principal Component Analysis (PCA) with Imputation
            # Create a pipeline that handles missing values and scales the data
            pca_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),  # Replace missing values with median
                ('scaler', StandardScaler()),  # Standardize features
                ('pca', PCA())  # Apply PCA
            ])
            
            # Fit the pipeline and transform the data
            pca_result = pca_pipeline.fit_transform(numeric_data)
            
            # Get the PCA object from the pipeline
            pca = pca_pipeline.named_steps['pca']
            
            # PCA Variance Explained Plot
            plt.figure(figsize=(10, 6))
            plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                    pca.explained_variance_ratio_, alpha=0.5, align='center')
            plt.step(range(1, len(pca.explained_variance_ratio_) + 1), 
                    np.cumsum(pca.explained_variance_ratio_), where='mid', label='Cumulative Explained Variance')
            plt.title('PCA Variance Explained')
            plt.xlabel('Principal Components')
            plt.ylabel('Proportion of Variance Explained')
            plt.legend()
            plt.tight_layout()
            plt.savefig('pca_variance_plot.png')
            plt.close()
            self.charts.append('pca_variance_plot.png')
            
            # 3. K-Means Clustering with Imputation
            clustering_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            scaled_data = clustering_pipeline.fit_transform(numeric_data)
            
            inertias = []
            max_clusters = min(10, len(numeric_columns))
            for k in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(scaled_data)
                inertias.append(kmeans.inertia_)
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, max_clusters + 1), inertias, marker='o')
            plt.title('Elbow Method for Optimal k')
            plt.xlabel('Number of clusters')
            plt.ylabel('Inertia')
            plt.tight_layout()
            plt.savefig('kmeans_elbow_plot.png')
            plt.close()
            self.charts.append('kmeans_elbow_plot.png')
            
            # Store analysis results
            self.analysis_results['pca_variance_ratio'] = pca.explained_variance_ratio_.tolist()
            self.analysis_results['top_components'] = {
                'component_variance': [float(var) for var in pca.explained_variance_ratio_],
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist()
            }
            self.analysis_results['kmeans_inertia'] = inertias
    
    def ask_llm_with_function_calling(self, functions: List[Dict], function_call: str = None):
        """
        Send a prompt to the LLM with function calling support
        
        :param functions: List of function definitions
        :param function_call: Specific function to call
        :return: LLM response
        """
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user", 
                    "content": "Analyze the following dataset and suggest advanced analytical approaches."
                }
            ],
            "functions": functions
        }
        
        if function_call:
            payload["function_call"] = {"name": function_call}
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']
        
        except Exception as e:
            print(f"Error in function calling: {e}")
            return None
    
    def generate_advanced_narrative(self):
        """
        Generate a comprehensive narrative in a story format using LLM, with images displayed after the relevant analysis.
        """
        # Step 1: Collect dataset overview and analysis results
        dataset_overview = self.get_dataset_overview()
        converted_overview = convert_numpy_types(dataset_overview)
        converted_analysis_results = convert_numpy_types(self.analysis_results)

        # Step 2: Prepare a data summary for the LLM
        full_data_summary = {
            "dataset_overview": converted_overview,
            "analysis_results": converted_analysis_results
        }

        # Step 3: Formulate the LLM prompt for storytelling
        llm_prompt = f"""
        You are an expert data storyteller. Using the following dataset overview and analysis results, craft a compelling narrative. 
        
        Dataset Overview:
        {json.dumps(converted_overview, indent=2)}
        
        Analysis Results:
        {json.dumps(converted_analysis_results, indent=2)}
        
        Story Requirements:
        1. Begin with a brief introduction of the dataset, describing its context and importance.
        2. Narrate the analysis conducted, including methods like correlation analysis, PCA, clustering, and outlier detection.
        3. Highlight key insights discovered from the analysis, with emphasis on actionable outcomes.
        4. Conclude with implications and recommendations for potential use cases.
        5. Reference visualizations such as advanced_correlation_heatmap.png, pca_variance_plot.png, kmeans_elbow_plot.png, and scatter_outliers_colX_vs_colY.png in the corresponding sections of the narrative.
        """

        # Step 4: Send the prompt to the LLM
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": llm_prompt}
            ]
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            # Extract the LLM's response
            narrative = result['choices'][0]['message']['content']

        except Exception as e:
            print(f"Error generating narrative: {e}")
            narrative = "# Dataset Analysis Report\n\nAn error occurred while generating the narrative. Please review the analysis manually."

        # Step 5: Append image references contextually to the narrative
        narrative += "\n\n## Detailed Visualizations\n"

        if 'dataset_overview' in converted_analysis_results:
            narrative += "### Dataset Overview Visualization\n"
            narrative += "![advanced_correlation_heatmap.png](advanced_correlation_heatmap.png)\n\n"

        if 'pca_variance_ratio' in converted_analysis_results:
            narrative += "### PCA Variance Plot\n"
            narrative += "This plot shows the explained variance ratio for principal components:\n"
            narrative += "![pca_variance_plot.png](pca_variance_plot.png)\n\n"

        if 'kmeans_inertia' in converted_analysis_results:
            narrative += "### K-Means Elbow Plot\n"
            narrative += "The elbow method visualization for determining the optimal number of clusters:\n"
            narrative += "![kmeans_elbow_plot.png](kmeans_elbow_plot.png)\n\n"

        if 'scatter_outliers' in converted_analysis_results:
            scatter_info = converted_analysis_results['scatter_outliers']
            narrative += f"### Scatter Plot with Outliers\nOutliers detected between {scatter_info['columns'][0]} and {scatter_info['columns'][1]} are highlighted:\n"
            narrative += f"![scatter_outliers_{scatter_info['columns'][0]}_vs_{scatter_info['columns'][1]}.png](scatter_outliers_{scatter_info['columns'][0]}_vs_{scatter_info['columns'][1]}.png)\n\n"

        # Step 6: Write the narrative to README.md
        with open('README.md', 'w', encoding='utf-8') as f:
            f.write(narrative)

        print("Narrative generation complete. Check README.md for details.")
    
    def analyze(self):
        """
        Main analysis method to orchestrate the entire process
        """
        # Perform initial data exploration
        dataset_overview = self.get_dataset_overview()
        self.analysis_results['dataset_overview'] = dataset_overview
        
        # Perform advanced analysis and visualizations
        self.perform_advanced_analysis()
        
        # Generate advanced narrative
        self.generate_advanced_narrative()

def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    analyzer = EnhancedAutolysisAnalyzer(csv_path)
    analyzer.analyze()
    
    print("Enhanced analysis complete. Check README.md and generated charts.")

if __name__ == "__main__":
    main()