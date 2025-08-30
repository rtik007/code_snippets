# Code Snippets for Data Science Projects

A comprehensive collection of practical code snippets and examples for data science, machine learning, and AI projects. This repository serves as a reference library for common data science tasks, featuring implementations in Python using popular libraries like pandas, scikit-learn, PyOD, DSPy, and more.

## 📋 Table of Contents

- [Overview](#overview)
- [Categories](#categories)
- [Installation & Setup](#installation--setup)
- [Quick Start](#quick-start)
- [Detailed Documentation](#detailed-documentation)
- [Prerequisites](#prerequisites)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This repository contains battle-tested code snippets organized by functionality, designed to accelerate data science workflows. Each snippet is self-contained and includes practical examples that can be easily adapted to your specific use cases.

## 📂 Categories

### 🤖 AI & Prompt Engineering
- **DSPy Examples** - Advanced prompt optimization and engineering
- **GenAI Case Studies** - Evaluation frameworks for generative AI models
- **DeepEval** - Faithfulness evaluation for AI systems

### 🎯 Feature Selection & Engineering
- **Correlation-based Selection** - Feature selection using correlation analysis
- **Isolation Forest Methods** - Feature importance using IForest and XGBoost
- **Exhaustive Search** - Comprehensive feature subset selection
- **Feature Bagging** - Ensemble methods for feature selection

### 🔍 Outlier Detection
- **Multiple Algorithm Comparison** - Comprehensive outlier detection toolkit
- **H2O IForest Rules** - Human-readable rule extraction from Isolation Forest
- **Spark & Seaborn Integration** - Scalable outlier visualization

### 📊 Data Visualization
- **Statistical Plots** - Box plots, violin plots, QQ plots for all numeric columns
- **Correlation Analysis** - Correlation matrices and correlograms
- **Pair Plots** - Comprehensive pairwise relationship visualization
- **Distribution Analysis** - Histograms and frequency tables

### 🧮 Data Preprocessing
- **PCA for Outliers** - Principal component analysis for outlier detection
- **IQR Sampling** - Interquartile range-based sampling techniques
- **Stratified Sampling** - Balanced dataset sampling methods

### 🔧 Utilities & Environment
- **Package Auditing** - Environment and package usage reporting
- **Command Line Tools** - Utility scripts for data science workflows

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Git (for cloning the repository)

### Dependencies
The code snippets use various Python libraries. Install the core dependencies:

```bash
# Core data science libraries
pip install pandas numpy scikit-learn matplotlib seaborn

# Outlier detection
pip install pyod

# Advanced AI/ML libraries
pip install dspy-ai h2o xgboost

# Evaluation frameworks
pip install deepeval rouge-score nltk transformers torch

# Visualization and statistical analysis
pip install plotly scipy statsmodels
```

### Setup
1. Clone the repository:
```bash
git clone https://github.com/rtik007/code_snippets.git
cd code_snippets
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies based on your needs (see individual notebook requirements)

## 🏃 Quick Start

### Example 1: Feature Selection with Isolation Forest
```python
# Load the Feature_importance_based_Feature_selection_using_IForest_and_XGB.ipynb
from pyod.models.iforest import IForest
from pyod.models.xgbod import XGBOD
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Your data preprocessing and feature selection code here
```

### Example 2: DSPy Prompt Optimization
```python
# Refer to DSPy_example.ipynb for complete implementation
import dspy
import pandas as pd

# Set up your language model and optimize prompts
```

### Example 3: Comprehensive Data Visualization
```python
# Use Box_plot_for_all_numeric_columns.ipynb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate statistical plots for your dataset
```

## 📖 Detailed Documentation

### 🤖 AI & Prompt Engineering

#### DSPy Examples (`DSPy_example.ipynb`, `DSPy_example_2.ipynb`)
Advanced prompt engineering using the DSPy framework for optimizing language model interactions.

**Features:**
- Automated prompt optimization
- Few-shot learning examples
- Metric-driven prompt improvement
- Evidence-based prompt evaluation

**Use Cases:**
- Text summarization optimization
- Classification task improvement
- Content generation enhancement

#### GenAI Case Studies (`GenAI_Case_Study_example.ipynb`)
Comprehensive evaluation framework for generative AI models with multiple metrics.

**Features:**
- BLEU score computation
- ROUGE score evaluation
- Perplexity measurement
- Comparative model analysis

**Metrics Included:**
- BLEU-1, ROUGE-1, ROUGE-L
- Model perplexity comparison
- Statistical significance testing

### 🎯 Feature Selection & Engineering

#### Isolation Forest Feature Selection
Multiple approaches to feature selection using Isolation Forest and ensemble methods.

**Available Methods:**
- `Feature_importance_based_Feature_selection_using_IForest_and_XGB.ipynb`
- `Unsupervised_ML_exhaustive_search_Feature_Selection.ipynb`
- `FeatureBagging_using_iForest_and_HBOS.ipynb`

**Key Algorithms:**
- Isolation Forest (IForest)
- XGBoost Outlier Detection (XGBOD)
- Histogram-based Outlier Score (HBOS)
- Feature bagging techniques

#### Correlation-based Selection
Automated feature selection based on correlation analysis and statistical relationships.

**Features:**
- Correlation matrix generation
- Feature redundancy detection
- Correlation-based filtering
- Statistical significance testing

### 🔍 Outlier Detection

#### Comprehensive Outlier Detection (`Multiple_outlier_detection_algo_to_select_best_subset_of_features.ipynb`)
Comparison framework for multiple outlier detection algorithms.

**Supported Algorithms:**
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- Elliptic Envelope
- DBSCAN clustering
- K-Means clustering
- AutoEncoder-based detection
- HBOS, KNN, PCA methods

#### H2O Integration (`Human Readable Rule extraction from H2O IForest Model.py`)
Extract interpretable rules from H2O Isolation Forest models.

**Features:**
- Tree traversal and rule extraction
- Human-readable decision rules
- Model interpretability enhancement
- H2O framework integration

### 📊 Data Visualization

#### Statistical Plotting Suite
Comprehensive visualization tools for exploratory data analysis:

- `Box_plot_for_all_numeric_columns.ipynb` - Distribution analysis
- `Violin_plots_for_all_numeric_columns.ipynb` - Density visualization
- `QQ_plot_for_all_numeric_columns.ipynb` - Normality testing
- `Pairplot_all_columns.ipynb` - Relationship analysis
- `Correlation_matrix_and_corrgram_all_columns.ipynb` - Correlation visualization

#### Advanced Visualization
- `Freq_table_and_histgram_all_columns.ipynb` - Frequency analysis
- `Correlation_table_in_3col_output.ipynb` - Structured correlation reporting
- `Outlier_pairplot_using_Spark_andSeaborn` - Scalable outlier visualization

### 🧮 Data Preprocessing

#### PCA and Dimensionality Reduction
`Data_preprocessing_and_PCA_for_outlier_example.ipynb` provides comprehensive preprocessing pipelines.

**Features:**
- Principal Component Analysis (PCA)
- Outlier-aware preprocessing
- Dimensionality reduction techniques
- Data scaling and normalization

#### Sampling Techniques
- `Sampling_using_IQR.ipynb` - Interquartile range-based sampling
- `stratified_sampling` - Balanced dataset creation
- Advanced sampling strategies

### 🔧 Utilities & Environment

#### Environment Auditing
- `env_audit_report.py` - Comprehensive environment analysis
- `env_pakage_usage_report` - Package usage tracking
- `Get_package_last_accessed_date.ipynb` - Dependency analysis

#### Command Line Utilities
- `utils4cmd_line` - Data science workflow automation
- `test.py` - Testing utilities

## 🔧 Prerequisites

### System Requirements
- **Operating System:** Windows, macOS, or Linux
- **Python:** 3.8+ (Python 3.9+ recommended)
- **Memory:** 4GB RAM minimum (8GB+ recommended for large datasets)
- **Storage:** 2GB free space for dependencies

### Required Skills
- **Basic Python:** Understanding of Python syntax and concepts
- **Data Science Fundamentals:** Familiarity with pandas, numpy, matplotlib
- **Jupyter Notebooks:** Experience with notebook environments
- **Machine Learning Basics:** Understanding of ML concepts (helpful but not required)

### Library-specific Requirements

#### For H2O Integration:
```bash
pip install h2o
# Java 8+ required for H2O
```

#### For DSPy Examples:
```bash
pip install dspy-ai
# API keys required for language models (OpenAI, Anthropic, etc.)
```

#### For Deep Learning Examples:
```bash
pip install torch transformers
# GPU support optional but recommended
```

## 🤝 Contributing

We welcome contributions to enhance this code snippet collection! Here's how you can contribute:

### How to Contribute
1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/new-snippet`
3. **Add your code snippet** with proper documentation
4. **Test your implementation** with sample data
5. **Submit a pull request** with a clear description

### Contribution Guidelines
- **Documentation:** Include clear comments and markdown explanations
- **Examples:** Provide sample data or usage examples
- **Dependencies:** List all required libraries in your notebook
- **Naming:** Use descriptive filenames that reflect functionality
- **Code Quality:** Follow PEP 8 style guidelines
- **Testing:** Ensure your code runs without errors

### What to Contribute
- New data science techniques
- Improved versions of existing snippets
- Bug fixes and optimizations
- Documentation improvements
- Real-world use case examples

## 📄 License

This project is licensed under the MIT License - see the repository for details.

## 🆘 Support

If you encounter issues or have questions:
1. Check existing notebook documentation
2. Review the prerequisites and setup instructions
3. Open an issue on GitHub with detailed error information
4. Include your Python version and operating system

## 🚀 Latest Updates

This repository is actively maintained with regular updates including:
- New data science techniques and algorithms
- Updated library compatibility
- Enhanced documentation and examples
- Performance optimizations
- Community-contributed snippets

---

**Happy Data Science! 🎉**

*Star this repository if you find it helpful, and don't forget to check back for new snippets and updates.*
