# Content Monetization Modeler: A Predictive Dashboard for YouTube Ad Revenue

This repository contains the source code for a comprehensive machine learning and data analysis project designed to forecast YouTube ad revenue. The project develops and evaluates a suite of five distinct regression models, which are deployed within a professionally styled, interactive dashboard built using the Streamlit framework.

The primary objective is to provide content creators and media planners with a reliable, data-driven tool to optimize content strategy, perform financial forecasting, and gain deep insights into key revenue drivers.

## Project Summary and Core Deliverables

This project focused on developing a complete, end-to-end solution, from initial data processing and feature engineering to the final deployment of a polished, user-friendly application.

### Key Achievements

* **Multi-Model Predictive Engine:** A machine learning pipeline was developed to train, compare, and allow for the dynamic selection of five regression algorithms: **Random Forest, Linear Regression, Ridge, Lasso, and Gradient Boosting**. This approach facilitates benchmarking predictive accuracy and provides flexibility.

* **Advanced Feature Engineering:** Raw video performance metrics were transformed into high-impact predictive features such as `engagement_rate` and `avg_watch_time`, which are critical for capturing the non-linear relationships that influence ad revenue (CPM).

* **Professional Streamlit Dashboard:** The dashboard was designed with a focus on aesthetics and user experience. It features a custom CSS theme that mirrors the YouTube platform's dark aesthetic, along with a responsive, flexible layout that ensures content integrity across various screen sizes.

* **Interactive Insights and Analysis:** The application includes dynamic visualizations that enable users to explore model performance, compare revenue across different countries and categories, and analyze feature importance to better understand the model's predictions.

## Dataset and Feature Engineering

The project relies on a cleaned dataset, `youtube_data_cleaned.csv`. The quality of the revenue predictions is directly linked to the effectiveness of the feature engineering process, which transformed raw data into meaningful and predictive variables.

| Raw Feature | Engineered Feature | Strategic Rationale |
|-------------|------------------|------------------|
| `views`, `likes`, `comments` | `engagement_rate` | High engagement is a key indicator of content quality, influencing ad placement value and CPM. |
| `watch_time_minutes`, `views` | `avg_watch_time` | Measures viewer retention, one of the strongest predictors of ad revenue. |
| `views`, `subscribers` | `views_per_subscriber` | Provides insight into the channel’s overall vitality and subscriber effectiveness. |
| `category`, `country`, `device` | Categorical Encoding | Captures global variations in CPM and ad market value. |

## Regression Modeling: Theoretical Foundation

Regression is a fundamental supervised learning technique used to predict a continuous target variable. In this project, the target is `ad_revenue_usd`.

### Core Principle: The Linear Equation

All linear models seek to establish a relationship between features ($X_i$) and the target ($Y$) using a linear function:

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon
$$

The models are trained by finding the optimal coefficients ($\beta_i$) that minimize the overall error, typically measured by the Mean Squared Error (MSE).

$$
\text{Loss Function (OLS)} = \sum_{i=1}^{m} (Y_i - \hat{Y}_i)^2
$$

## Models and Their Project Relevance

### 1. Ridge Regression (L2 Regularization)

Ridge regression extends Linear Regression by adding an L2 penalty to the loss function. This term constrains the magnitude of the coefficients, preventing them from becoming excessively large.

$$
\text{Loss Function (Ridge)} = \sum_{i=1}^{m} (Y_i - \hat{Y}_i)^2 + \lambda \sum_{j=1}^{n} \beta_j^2
$$

* **Relevance:** Mitigates multicollinearity among features, leading to a more stable and generalized model performance.

### 2. Lasso Regression (L1 Regularization)

Lasso introduces an L1 penalty that encourages sparsity. This can force the coefficients of less influential features to zero.

$$
\text{Loss Function (Lasso)} = \sum_{i=1}^{m} (Y_i - \hat{Y}_i)^2 + \lambda \sum_{j=1}^{n} |\beta_j|
$$

* **Relevance:** Serves as an automated feature selection tool by identifying and eliminating irrelevant features.

### 3. Random Forest Regressor

A non-linear, ensemble model that constructs multiple decision trees and averages their outputs to make a prediction.

* **Process:** Recursively partitions the feature space to minimize variance, effectively capturing complex, non-linear relationships.
* **Relevance:** Provides high predictive accuracy and underpins the feature importance visualization in the dashboard.

### 4. Gradient Boosting Regressor

An iterative ensemble technique where each new tree is trained to correct the residual errors of previous trees.

* **Process:** Minimizes the loss function by fitting new models to the negative gradient of the loss function.
* **Relevance:** Often delivers the highest predictive accuracy and serves as a benchmark for optimal forecasting.

## Evaluation Metrics

Model performance is assessed using standard regression metrics:

### 1. R-squared ($R^2$)

Measures the proportion of variance in the target explained by the features.

$$
R^2 = 1 - \frac{\text{Sum of Squared Residuals (SSR)}}{\text{Total Sum of Squares (SST)}}
$$

* **Interpretation:** Values closer to 1 indicate a better model fit.

### 2. Root Mean Squared Error (RMSE)

Measures the average magnitude of errors, penalizing larger errors more heavily.

$$
\text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (Y_i - \hat{Y}_i)^2}
$$

* **Interpretation:** Lower RMSE indicates higher accuracy.

### 3. Mean Absolute Error (MAE)

Measures the average magnitude of errors using absolute differences.

$$
\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} |Y_i - \hat{Y}_i|
$$

* **Interpretation:** Lower MAE indicates better model performance in USD.

## Installation and Setup

### Prerequisites

* Python 3.8+
* Git

### Local Installation Steps

```bash
# Clone the repository
git clone https://github.com/kishore2412004/Content-Monetization.git

# Navigate to the project directory
cd Content-Monetization

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate    # Linux/macOS
.\venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Ensure all five model .pkl files are in the Models/ directory
# Ensure youtube_data_cleaned.csv is in the project root

# Run the Streamlit application
streamlit run your_main_app_file_name.py
