import pandas as pd
import joblib
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

df = pd.read_csv("youtube_data_cleaned.csv")

target = 'ad_revenue_usd'
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

categorical_features = ['category', 'device', 'country']
numeric_features = ['views', 'likes', 'comments', 'subscribers',
                    'video_length_minutes', 'watch_time_minutes']

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

os.makedirs("Models", exist_ok=True)
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X, y)
    filename = os.path.join("Models", name.lower().replace(" ", "_") + "_pipeline.pkl")
    joblib.dump(pipeline, filename)
    print(f"*** Saved {name} pipeline to {filename} ***")