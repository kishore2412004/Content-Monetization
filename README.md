# Content Monetization Modeler: A Predictive Dashboard for YouTube Ad Revenue

This repository contains the source code for a comprehensive machine learning and data analysis project designed to forecast YouTube ad revenue. The project develops and evaluates a suite of five distinct regression models, which are deployed within a professionally styled, interactive dashboard built using the Streamlit framework.

The primary objective is to provide content creators and media planners with a reliable, data-driven tool to optimize content strategy, perform financial forecasting, and gain deep insights into key revenue drivers.

## Project Summary and Core Deliverables

This project focused on developing a complete, end-to-end solution, from initial data processing and feature engineering to the final deployment of a polished, user-friendly application.

### Key Achievements

* **Multi-Model Predictive Engine:** A machine learning pipeline was developed to train, compare, and allow for the dynamic selection of five regression algorithms: **Random Forest, Linear Regression, Ridge, Lasso, and Gradient Boosting**. This approach facilitates benchmarking predictive accuracy and provides flexibility.

* **Advanced Feature Engineering:** Raw video performance metrics were transformed into high-impact predictive features such as **`engagement_rate`** and **`avg_watch_time`**, which are critical for capturing the non-linear relationships that influence ad revenue (CPM).

* **Professional Streamlit Dashboard:** The dashboard was designed with a focus on aesthetics and user experience. It features a custom CSS theme that mirrors the YouTube platform's dark aesthetic, along with a responsive, flexible layout that ensures content integrity across various screen sizes.

* **Interactive Insights and Analysis:** The application includes dynamic visualizations that enable users to explore model performance, compare revenue across different countries and categories, and analyze **feature importance** to better understand the model's predictions.

## Dataset and Feature Engineering

The project relies on a cleaned dataset, `youtube_data_cleaned.csv`. The quality of the revenue predictions is directly linked to the effectiveness of the feature engineering process, which transformed raw data into meaningful and predictive variables.

| Raw Feature | Engineered Feature | Strategic Rationale |
| :--- | :--- | :--- |
| `views`, `likes`, `comments` | `engagement_rate` | High engagement is a key indicator of content quality, which influences ad placement value and justifies higher CPMs. |
| `watch_time_minutes`, `views` | `avg_watch_time` | This metric is a direct measure of viewer retention and is one of the strongest predictors of successful ad delivery and revenue. |
| `views`, `subscribers` | `views_per_subscriber` | Provides insight into the overall vitality of the channel and the effectiveness of its subscriber base. |
| `category`, `country`, `device` | Categorical Encoding | These contextual variables are crucial for capturing the global variations in cost-per-mille (CPM) and ad market value. |

## Regression Modeling: Theoretical Foundation

Regression is a fundamental supervised learning technique used to predict a continuous target variable. In this project, the target is the `ad_revenue_usd`.

### Core Principle: The Linear Equation

All linear models seek to establish a relationship between features ($X_i$) and the target ($Y$) using a linear function:

$$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_nX_n + \epsilon$$

The models are trained by finding the optimal coefficients ($\beta_i$) that minimize the overall error, typically measured by the Mean Squared Error (MSE).

$$\text{Loss Function (OLS)} = \sum_{i=1}^{m} (Y_i - \hat{Y}_i)^2$$

## Models and Their Project Relevance

The project's strength lies in its comparative analysis of different model types to identify the most suitable solution for the given dataset.

### 1. Ridge Regression (L2 Regularization)

Ridge regression extends Linear Regression by adding an L2 penalty to the loss function. This term constrains the magnitude of the coefficients, preventing them from becoming excessively large.

$$\text{Loss Function (Ridge)} = \sum_{i=1}^{m} (Y_i - \hat{Y}_i)^2 + \lambda \sum_{j=1}^{n} \beta_j^2$$

* **Project Relevance:** This model is crucial for mitigating **multicollinearity** among engineered features, leading to a more stable and generalized model performance.

### 2. Lasso Regression (L1 Regularization)

Lasso introduces an L1 penalty that encourages sparsity. This means it can force the coefficients of less influential features to become exactly zero.

$$\text{Loss Function (Lasso)} = \sum_{i=1}^{m} (Y_i - \hat{Y}_i)^2 + \lambda \sum_{j=1}^{n} |\beta_j|$$

* **Project Relevance:** This model serves as an effective **automated feature selection** tool, streamlining the predictive process by identifying and eliminating irrelevant features from the dataset.

### 3. Random Forest Regressor

As a non-linear, ensemble model, Random Forest constructs multiple decision trees and averages their outputs to make a final prediction.

* **Mathematical Process:** It recursively partitions the feature space based on minimizing variance, making it highly effective at capturing complex, non-linear relationships.

* **Project Relevance:** This model often provides high predictive accuracy and is the basis for the **Feature Importance** data displayed in the application's Insights dashboard.

### 4. Gradient Boosting Regressor

This iterative ensemble technique builds trees sequentially, with each new tree trained to correct the errors (residuals) of the previous ones.

* **Mathematical Process:** It minimizes the loss function by fitting new models to the negative gradient of the loss function, progressively reducing error.

* **Project Relevance:** Gradient Boosting typically delivers the **highest predictive accuracy**, serving as a benchmark for the best possible forecasting capability.

## Evaluation Metrics

Model reliability is assessed using standard regression metrics, which are displayed in the application's "Model Testing" section.

### 1. R-squared ($R^2$) - Coefficient of Determination

Measures the proportion of the variance in the target variable that is predictable from the features.

$$R^2 = 1 - \frac{\text{Sum of Squared Residuals (SSR)}}{\text{Total Sum of Squares (SST)}}$$

* **Interpretation:** A value closer to 1.0 indicates a superior model fit, explaining a higher percentage of the revenue's variance.

### 2. Root Mean Squared Error (RMSE)

Measures the average magnitude of the errors, with larger errors receiving a higher penalty.

$$\text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (Y_i - \hat{Y}_i)^2}$$

* **Interpretation:** The average prediction error, expressed in the same unit as the revenue (USD). A lower value indicates higher accuracy.

### 3. Mean Absolute Error (MAE)

Measures the average magnitude of the errors using the absolute difference, making it less sensitive to outliers than RMSE.

$$\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} |Y_i - \hat{Y}_i|$$

* **Interpretation:** Represents the expected magnitude of error in USD. A lower value indicates better performance.

## Installation and Setup

### Prerequisites

* Python 3.8+

* Git

### Local Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone [https://github.com/kishore2412004/Content-Monetization.git](https://github.com/kishore2412004/Content-Monetization.git)
   cd Content-Monetization
