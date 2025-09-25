# 📈🎥 Content Monetization Modeler: A Multi-Model Predictive Dashboard for YouTube Ad Revenue

This repository presents a **robust analytical framework** developed to accurately forecast YouTube ad revenue. The core of this work is a comprehensive machine learning pipeline designed to compare and leverage five distinct regression models, deployed within a professional, interactive Streamlit dashboard for real-time strategic insights.

We aim to provide content creators and media planners with a dependable, data-driven tool for optimizing content strategy and performing crucial financial forecasting.

***

## ✨ My Contribution and Project Deliverables

This project focused on developing a complete, end-to-end solution, from raw data processing to final application deployment.

### Key Achievements

1.  **Multi-Model Predictive Engine:** Developed a machine learning pipeline that trains, compares, and allows the user to dynamically select from five regression algorithms: **Random Forest, Linear Regression, Ridge, Lasso, and Gradient Boosting**. This approach ensures **benchmarking predictive accuracy** and flexibility.
2.  **Advanced Feature Engineering:** Transformed raw video metrics into sophisticated predictive features, such as **`engagement_rate`** and **`avg_watch_time`**, which capture non-obvious relationships driving ad revenue (CPM).
3.  **Professional Streamlit Dashboard:** Focused on user experience and aesthetic integrity by developing a custom, highly-themed Streamlit application. This includes:
    * A custom CSS styling that mimics the dark theme and color palette of the YouTube platform.
    * An intuitive fixed navigation menu.
    * A **YouTube-style card theme** applied to titles and key fact boxes, utilizing flexible CSS to prevent content overflow and maintain a polished look.
4.  **Interactive Insights Module:** Implemented dynamic visualizations (Plotly) that allow users to explore model performance, compare revenue across different **`country`** and **`category`** variables, and analyze **Feature Importance** from the tree-based models.

***

## 📊 Dataset and Feature Engineering

The project relies on the `youtube_data_cleaned.csv` dataset. The quality of predictions hinges on effective feature engineering to distill actionable information from raw performance data.

| Raw Feature | Engineered Feature | Strategic Rationale |
| :--- | :--- | :--- |
| `views`, `likes`, `comments` | **`engagement_rate`** | High engagement is a key factor signaling a high-quality ad inventory and justifying higher CPMs to advertisers. |
| `watch_time_minutes`, `views` | **`avg_watch_time`** | This measure of viewer retention is one of the strongest predictors of successful ad delivery and thus revenue. |
| `views`, `subscribers` | **`views_per_subscriber`** | Provides insight into channel vitality and subscriber base effectiveness. |
| `category`, `country`, `device` | Encoded Categorical Features | Contextual variables that directly influence the varying cost-per-mille (CPM) globally. |

***

## 🧠 Regression Modeling: Theoretical Foundation

Regression is a fundamental supervised learning technique used here to predict the continuous target variable, **`ad_revenue_usd`**.

### The Core Principle: Linear Equation

All linear models aim to fit a function to the data by establishing a relationship between the features ($X_i$) and the target ($Y$):

$$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_nX_n + \epsilon$$

The process involves finding the optimal **Coefficients ($\beta_i$)** that minimize the overall error, typically measured as the **Mean Squared Error (MSE)**.

$$\text{Loss Function (OLS)} = \sum_{i=1}^{m} (Y_i - \hat{Y}_i)^2$$

***

## ⚙️ Models and Their Project Relevance

The project's strength lies in comparing the stability and accuracy of different model types.

### 1. Ridge Regression (L2 Regularization)

Ridge regression enhances Linear Regression by adding an **L2 penalty** to the loss function. This term shrinks coefficient magnitudes toward zero.

$$\text{Loss Function (Ridge)} = \sum_{i=1}^{m} (Y_i - \hat{Y}_i)^2 + \lambda \sum_{j=1}^{n} \beta_j^2$$

* **Project Relevance:** Crucial for **mitigating multicollinearity** among engineered features, preventing coefficients from becoming unstable, and leading to a more generalized model performance.

### 2. Lasso Regression (L1 Regularization)

Lasso introduces an **L1 penalty** which encourages sparsity, meaning it can force the coefficients of less influential features to be **exactly zero**.

$$\text{Loss Function (Lasso)} = \sum_{i=1}^{m} (Y_i - \hat{Y}_i)^2 + \lambda \sum_{j=1}^{n} |\beta_j|$$

* **Project Relevance:** Serves as an **automated feature selection** tool. By comparing its performance, we can confirm the necessity of certain features and streamline the predictive model.

### 3. Random Forest Regressor

This non-linear, ensemble model builds multiple decision trees and averages their outputs.

* **Mathematical Process:** It minimizes variance within each recursive split of the feature space, excelling at capturing complex, non-linear feature interactions.
* **Project Relevance:** Often yields high predictive accuracy and is the source for the **Feature Importance** data displayed in the Insights dashboard.

### 4. Gradient Boosting Regressor

This iterative ensemble technique builds trees sequentially, where each new tree is trained to correct the errors (residuals) made by the combination of all previous trees.

* **Mathematical Process:** It minimizes the overall loss by iteratively fitting models to the **negative gradient** of the loss function.
* **Project Relevance:** Typically provides the **highest predictive accuracy**, making it the ideal benchmark for determining the project's best possible forecasting capability.

***

## 📐 Evaluation Metrics

Model reliability is assessed using standard regression metrics, easily viewable in the dashboard's "Model Testing" sidebar.

### 1. R-squared ($R^2$) - Coefficient of Determination

Measures the proportion of the variance in the target variable that is predictable from the features.

$$R^2 = 1 - \frac{\text{Sum of Squared Residuals (SSR)}}{\text{Total Sum of Squares (SST)}}$$

* **Interpretation:** A score closer to 1.0 indicates a superior model fit.

### 2. Root Mean Squared Error (RMSE)

Measures the average magnitude of the errors, penalizing larger errors due to the squaring operation.

$$\text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (Y_i - \hat{Y}_i)^2}$$

* **Interpretation:** The average prediction error, expressed in the same unit as the revenue (USD). Lower is better.

### 3. Mean Absolute Error (MAE)

Measures the average magnitude of the errors using the absolute difference, providing a clearer view of the typical prediction error without the high penalty for outliers.

$$\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} |Y_i - \hat{Y}_i|$$

* **Interpretation:** The expected magnitude of error in USD. Lower is better.

***

## 🛠️ Installation and Setup

### Prerequisites

* Python 3.8+
* Git

### Local Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/kishore2412004/Content-Monetization.git](https://github.com/kishore2412004/Content-Monetization.git)
    cd Content-Monetization
    ```

2.  **Activate Environment (Highly Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # For Linux/macOS
    .\venv\Scripts\activate   # For Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure all five model `.pkl` files are in the `Models/` directory and the `youtube_data_cleaned.csv` file is in the project root.)*

### Running the Application

```bash
streamlit run your_main_app_file_name.py
