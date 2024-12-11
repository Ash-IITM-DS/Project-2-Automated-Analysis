# Autolysis: Automated Data Analysis and Storytelling

## Overview

Autolysis is a powerful Python script that transforms raw CSV datasets into rich, insightful narratives. By combining advanced machine learning techniques with AI-driven storytelling, this tool offers a comprehensive approach to data exploration and analysis.

## Features

- 📊 **Comprehensive Data Exploration**
  - Automatic dataset overview
  - Advanced statistical analysis
  - Robust handling of different data types and encodings

- 🔍 **Advanced Visualization Techniques**
  - Correlation heatmaps
  - Principal Component Analysis (PCA)
  - Outlier detection scatter plots
  - K-means clustering analysis
  - Elbow method for clustering optimization

- 🤖 **AI-Powered Narrative Generation**
  - Uses GPT-4o-Mini to create human-readable stories from data
  - Translates complex statistical findings into actionable insights

- 🔬 **Flexible Analysis**
  - Works with any CSV dataset
  - Adaptable to various domains (books, happiness metrics, media ratings, etc.)

## Prerequisites

- Python 3.8+
- Required libraries: pandas, numpy, seaborn, matplotlib, scikit-learn
- AI Proxy token (set as environment variable `AIPROXY_TOKEN`)

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   uv pip install pandas numpy seaborn matplotlib scikit-learn requests
   ```

## Usage

```bash
uv run autolysis.py <dataset.csv>
```

### Example

```bash
export AIPROXY_TOKEN=your_token_here
uv run autolysis.py goodreads.csv
```

## Outputs

After running the script, you'll get:
- `README.md`: A narrative report of the data analysis
- `*.png`: Visualization charts
  - Advanced correlation heatmap
  - PCA variance plot
  - K-means elbow plot
  - Outlier scatter plots

## How It Works

1. Load and clean the CSV dataset
2. Perform statistical analysis
3. Generate advanced visualizations
4. Use AI to create a storytelling narrative
5. Save results as markdown and image files
