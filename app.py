import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import numpy as np
import plotly.express as px
from PIL import Image
import base64


st.markdown("""
<style>
/* ========================================= */
/* 1. GLOBAL YOUTUBE THEME STYLES */
/* ========================================= */

/* Global font */
html, body, [class*="css"] {
    font-family: 'YouTube Sans', 'Roboto', sans-serif;
}

/* App background like YouTube dark */
.main {
    background-color: #0F0F0F; /* Darker background */
    color: #fff;
    padding-top: 70px; /* IMPORTANT: Add padding to main content to clear fixed menu */
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #202020;
    color: #fff;
}

/* ========================================= */
/* 2. FIXED MENU BAR (Targeting Custom ID for Reliability) */
/* ========================================= */

#fixed-navbar-wrapper {
    position: fixed !important; /* Force fixed position */
    top: 0 !important;
    left: 0;
    right: 0;
    z-index: 1000;
    background-color: #121212; /* Menu Bar Dark Background */
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.6);
    padding: 0;
    margin: 0;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
}
/* Ensure the option_menu content itself is styled within the fixed bar */
#fixed-navbar-wrapper > div {
    width: 100%;
    display: flex;
    justify-content: center;
    padding-top: 10px;
}

/* Menu items styling */
.nav-link {
    font-size: 15px !important;
    padding: 12px 20px !important;
    margin: 0 6px !important;
    border-radius: 4px;
    color: #AAAAAA !important; /* Muted text */
    transition: all 0.2s;
}
.nav-link:hover {
    color: #fff !important;
}

.nav-link-selected {
    background-color: transparent !important;
    border-bottom: 3px solid #FF0000 !important; /* YouTube Red underline */
    font-weight: bold !important;
    color: #fff !important;
}

/* Card containers for Streamlit's default elements */
.stCard {
    background-color: #282828; /* Slightly lighter card background */
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.05);
}

/* ========================================= */
/* 3. HOME PAGE CARD STYLES (Reusable) */
/* ========================================= */

.youtube-card-wrapper {
    background: linear-gradient(135deg, #202020, #121212);
    padding: 20px;
    margin-bottom: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    border: 1px solid rgba(255,255,255,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease, border 0.3s ease;
}
.youtube-card-wrapper:hover {
    transform: translateY(-5px);
    box-shadow: 0 0 15px rgba(255, 0, 0, 0.6), 0 12px 24px rgba(0,0,0,0.5);
    border: 1px solid #FF0000;
}

/* Specific styling for fact cards (inside the wrapper) */
.fact-container {
    background: transparent; /* Use wrapper background */
    padding: 0;
    margin: 0;
    height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    border: none;
    box-shadow: none;
    transition: none;
}
.fact-container:hover {
    transform: none;
    box-shadow: none;
    border: none;
}
.fact-header {
    color: white;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 12px;
    text-shadow: 0 1px 2px rgba(0,0,0,0.4);
}
.fact-body {
    font-size: 16px;
    color: rgba(255,255,255,0.85);
    line-height: 1.4;
}
.emoji {
    font-size: 24px;
    margin-bottom: 8px;
    color: #FF0000; 
}
/* Ensure titles inside the card are styled */
.youtube-card-wrapper h1, .youtube-card-wrapper h2, .youtube-card-wrapper h3 {
    text-align: center;
    color: white !important;
    text-shadow: 0 2px 4px rgba(255,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data():
    return pd.read_csv("youtube_data_cleaned.csv")

df_clean = load_data()
target = "ad_revenue_usd"
features = [col for col in df_clean.columns if col != target]

@st.cache_resource
def load_model(model_name):
    filename = os.path.join("Models", model_name.lower().replace(" ", "_") + "_pipeline.pkl")
    return joblib.load(filename)

menu_container = st.container()
with menu_container:
    st.markdown('<div id="fixed-navbar-wrapper">', unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=["Home", "Insights", "Prediction", "Forecasting"],
        icons=["house-fill", "bar-chart-fill", "search", "graph-up-fill"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0", "background-color": "transparent", "border": "none", "width": "100%"},
            "nav-link": {"font-size": "15px", "padding": "12px 20px", "margin": "0 6px", "font-weight": "bold", "color": "#AAAAAA"},
            "nav-link-selected": {"background-color": "transparent", "border-bottom": "3px solid #FF0000", "color": "#fff"},
        }
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.header("🔧 Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a Model",
    ["Linear Regression", "Ridge", "Lasso", "Random Forest", "Gradient Boosting"]
)
pipeline = load_model(model_choice)
st.sidebar.success(f"Using {model_choice}")


with st.sidebar.expander("⚙️ Info & Model Testing", expanded=False):
    st.subheader("📊 Model Accuracy Evaluation")

    if st.button("Run Accuracy Test"):
        X = df_clean[features]
        y = df_clean[target]    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.markdown("### 🔎 Model Performance Metrics")
        st.write(f"MAE: **{mae:.2f}**")
        st.write(f"RMSE: **{rmse:.2f}**")
        st.write(f"R²: **{r2:.4f}**")

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
        ax.set_xlabel("Actual Revenue")
        ax.set_ylabel("Predicted Revenue")
        ax.set_title("Actual vs Predicted Revenue")
        st.pyplot(fig)



if selected == "Home":

    # st.markdown("""
    #     <div class="youtube-card-wrapper" style="text-align: center; padding: 10px;">
    #         <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/YouTube_Logo_%282015-2017%29.svg/500px-YouTube_Logo_%282015-2017%29.svg.png" 
    #              width=300 style="margin-top: 10px; margin-bottom: 5px;">
    #     </div>
    # """, unsafe_allow_html=True)
    
    with open("youtube-logo-png-46020.png", "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <div style="text-align: center; padding: 20px;">
            <img src="data:image/png;base64,{encoded}" width="300"/>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
        <div class="youtube-card-wrapper">
            <h1 style="text-align: center; margin-top: 0; margin-bottom: 5px;">YouTube Ad Revenue Analytics Dashboard</h1>
            <p style="text-align: center; color: rgba(255,255,255,0.7);">
                Welcome to the <b>YouTube Ad Revenue Prediction App</b>! 🎥📈 This tool leverages 
                <b>Machine Learning models</b> to help creators make <b>data-driven decisions</b>.
            </p>
        </div>
    """, unsafe_allow_html=True)
 
    st.markdown("""
        <div class="youtube-card-wrapper">
            <h2 style="text-align: center; margin-top: 0;">Key Factors Influencing Revenue</h2>
            <ul style="list-style-type: disc; padding-left: 20px; color: rgba(255,255,255,0.9);">
                <li>YouTube pays creators based on **CPM (Cost Per Mille)** and **viewer engagement**.</li>
                <li>**Watch Time & Engagement Rate** strongly influence revenue — not just views.</li>
                <li>**Geography matters**: Ads in US, UK, and Canada generate higher CPMs.</li>
                <li>**Category impact**: Tech & Finance channels often earn higher CPM compared to Music or Gaming.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
   

    st.divider()

    st.header("Ad Revenue Insights (2024-2025 Data)")
    st.markdown("""
    Based on recent industry reports, here are the top-level statistics that define today's YouTube revenue landscape:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="youtube-card-wrapper">
                <div class="fact-container">
                    <div>
                        <div class="emoji">💰</div>
                        <div class="fact-header">Highest CPM Niches</div>
                    </div>
                    <div class="fact-body">Finance and Digital Marketing command up to **$50 CPM**, significantly outperforming other categories.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="youtube-card-wrapper">
                <div class="fact-container">
                    <div>
                        <div class="emoji">🌍</div>
                        <div class="fact-header">Top Ad Market</div>
                    </div>
                    <div class="fact-body">Australia, the US, and Canada have the **highest CPM rates** (all over $29), making audience location key.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    st.header("Watch: How YouTube Ad Revenue Works 📺")
    st.markdown("""
        <div class="youtube-card-wrapper" style="padding: 0; overflow: hidden; border: 2px solid #FF0000; box-shadow: 0 0 15px rgba(255, 0, 0, 0.6), 0 12px 24px rgba(0,0,0,0.5);">
            <div style="position: relative; width: 100%; padding-top: 56.25%;">
                <iframe 
                    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
                    src="https://www.youtube.com/embed/8h_2oVspFYw" 
                    frameborder="0" 
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                    allowfullscreen
                ></iframe>
            </div>
            <div style="padding: 15px; background-color: #121212;">
                <h4 style="color: white; margin: 0; font-size: 18px;">YouTube Ads Revenue Analytics Explained</h4>
                <p style="color: #AAAAAA; margin: 5px 0 0 0; font-size: 14px;">A detailed look at how to optimize ad placements and understand your analytics for maximum earnings.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    *Note: These statistics are based on 2024-2025 industry estimates and highlight the factors your machine learning model uses for prediction.*
    """)


    

# if selected == "Prediction":
#     # st.header("Predict Revenue for a Single Video")

#     st.markdown("""
#             <div style='background-color: rgba(255,255,255,0.04); 
#                         border: 1px solid #FF0000; 
#                         padding: 16px; 
#                         border-radius: 12px;
#                         margin-bottom: 25px;
#                         box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
#                 <h3 style='color: white ; text-align: center;'>Predict Revenue for a Single Video</h3>
#         """, unsafe_allow_html=True)

#     input_dict = {}
#     for col in features:
#         if col in ["views", "likes", "comments", "subscribers", "video_length_minutes", "watch_time_minutes"]:
#             input_dict[col] = st.number_input(
#                 f"{col}", min_value=0, value=int(df_clean[col].median()), step=100
#             )
#         elif col in ["category", "device", "country"]: 
#             input_dict[col] = st.selectbox(f"{col}", df_clean[col].unique())
#         else:
#             if pd.api.types.is_numeric_dtype(df_clean[col]):
#                 input_dict[col] = st.number_input(
#                     f"{col}", min_value=0, value=int(df_clean[col].median()), step=100
#                 )
#             else:
#                 input_dict[col] = st.selectbox(f"{col}", df_clean[col].unique())

#     input_data = pd.DataFrame([input_dict])

#     if st.button("Predict Single Video Revenue"):
#         prediction = pipeline.predict(input_data)[0]
#         st.success(f"Predicted Ad Revenue: **${prediction:.2f}**")

if selected == "Prediction":
    st.markdown("""
        <div style='background-color: rgba(255,255,255,0.04); 
                    border: 1px solid #FF0000; 
                    padding: 16px; 
                    border-radius: 12px;
                    margin-bottom: 25px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
            <h3 style='color: white ; text-align: center;'>Predict Revenue for a Single Video</h3>
        """, unsafe_allow_html=True)

    raw_features = ["views", "likes", "comments", "subscribers",
                    "video_length_minutes", "watch_time_minutes",
                    "category", "device", "country"]

    input_dict = {}
    for col in raw_features:
        if col in ["views", "likes", "comments", "subscribers", "video_length_minutes", "watch_time_minutes"]:
            input_dict[col] = st.number_input(
                f"{col.capitalize()}",
                min_value=0,
                value=int(df_clean[col].median()),
                step=100
            )
        else:  # categorical
            input_dict[col] = st.selectbox(f"{col.capitalize()}", df_clean[col].unique())

    # Create DataFrame
    input_data = pd.DataFrame([input_dict])
    input_data["engagement_rate"] = (input_data["likes"] + input_data["comments"]) / input_data["views"].replace(0, 1)
    input_data["views_per_subscriber"] = input_data["views"] / input_data["subscribers"].replace(0, 1)
    input_data["avg_watch_time"] = input_data["watch_time_minutes"] / input_data["views"].replace(0, 1)
    input_data["log_views"] = np.log1p(input_data["views"])
    input_data["dayofweek"] = 0
    input_data["is_weekend"] = 0
    input_data["month"] = 1
    input_data["year"] = 2025

    if st.button("Predict Single Video Revenue"):
        prediction = pipeline.predict(input_data)[0]
        st.success(f"Predicted Ad Revenue: **${prediction:.2f}**")




# if selected == "Forecasting":
#     # st.header("Future Revenue Forecasting")
#     st.markdown("""
#             <div style='background-color: rgba(255,255,255,0.04); 
#                         border: 1px solid #FF0000; 
#                         padding: 16px; 
#                         border-radius: 12px;
#                         margin-bottom: 25px;
#                         box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
#                 <h3 style='color: white ; text-align: center;'>Future Revenue Forecasting</h3>
#         """, unsafe_allow_html=True)

#     num_videos = st.number_input("Number of Future Videos per Month", min_value=1, value=5, step=1)

#     st.markdown("### Set Expected Averages for Future Videos")

#     future_input_dict = {}
#     for col in features:
#         if col in ["views", "likes", "comments", "subscribers", "video_length_minutes", "watch_time_minutes"]:
#             future_input_dict[col] = st.number_input(
#                 f"Expected Avg {col}", 
#                 min_value=0, 
#                 value=int(df_clean[col].median()), 
#                 step=100
#             )
#         elif col in ["category", "device", "country"]:
#             future_input_dict[col] = st.selectbox(f"Expected Avg {col}", df_clean[col].unique())
#         else:
#             if pd.api.types.is_numeric_dtype(df_clean[col]):
#                 future_input_dict[col] = st.number_input(
#                     f"Expected Avg {col}", 
#                     min_value=0, 
#                     value=int(df_clean[col].median()), 
#                     step=100
#                 )
#             else:
#                 future_input_dict[col] = st.selectbox(f"Expected Avg {col}", df_clean[col].unique())

#     future_data = pd.DataFrame([future_input_dict])


#     if st.button("Forecast Future Monthly Revenue"):
#         revenue_per_video = pipeline.predict(future_data)[0]
#         total_revenue = revenue_per_video * num_videos

#         st.success(f"Expected Revenue per Future Video: **${revenue_per_video:.2f}**")
#         st.success(f"Forecasted Monthly Revenue (Future): **${total_revenue:.2f}**")

if selected == "Forecasting":
    st.markdown("""
        <div style='background-color: rgba(255,255,255,0.04); 
                    border: 1px solid #FF0000; 
                    padding: 16px; 
                    border-radius: 12px;
                    margin-bottom: 25px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
            <h3 style='color: white ; text-align: center;'>Future Revenue Forecasting</h3>
        """, unsafe_allow_html=True)

    num_videos = st.number_input("Number of Future Videos per Month", min_value=1, value=5, step=1)

    st.markdown("### Set Expected Averages for Future Videos")

    raw_features = ["views", "likes", "comments", "subscribers",
                    "video_length_minutes", "watch_time_minutes",
                    "category", "device", "country"]

    future_input_dict = {}
    for col in raw_features:
        if col in ["views", "likes", "comments", "subscribers", "video_length_minutes", "watch_time_minutes"]:
            future_input_dict[col] = st.number_input(
                f"Expected Avg {col.capitalize()}",
                min_value=0,
                value=int(df_clean[col].median()),
                step=100
            )
        else:
            future_input_dict[col] = st.selectbox(f"Expected Avg {col.capitalize()}", df_clean[col].unique())

    future_data = pd.DataFrame([future_input_dict])

    future_data["engagement_rate"] = (future_data["likes"] + future_data["comments"]) / future_data["views"].replace(0, 1)
    future_data["views_per_subscriber"] = future_data["views"] / future_data["subscribers"].replace(0, 1)
    future_data["avg_watch_time"] = future_data["watch_time_minutes"] / future_data["views"].replace(0, 1)
    future_data["log_views"] = np.log1p(future_data["views"])

    future_data["dayofweek"] = 0
    future_data["is_weekend"] = 0
    future_data["month"] = 1
    future_data["year"] = 2025

    if st.button("Forecast Future Monthly Revenue"):
        revenue_per_video = pipeline.predict(future_data)[0]
        total_revenue = revenue_per_video * num_videos

        st.success(f"📹 Expected Revenue per Future Video: **${revenue_per_video:.2f}**")
        st.success(f"📈 Forecasted Monthly Revenue: **${total_revenue:.2f}**")



# if selected == "Bulk Predictions":
#     st.header(" Upload CSV for Bulk Revenue Prediction")

#     uploaded_file = st.file_uploader("Upload CSV with the same feature columns", type=["csv"])
#     if uploaded_file is not None:
#         input_df = pd.read_csv(uploaded_file)
#         missing_cols = set(features) - set(input_df.columns)
#         if missing_cols:
#             st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
#         else:
#             st.write("### Preview of Uploaded Data")
#             st.dataframe(input_df.head())

#             if st.button("Predict Revenue for Uploaded Data"):
#                 predictions = pipeline.predict(input_df)
#                 input_df["Predicted_Ad_Revenue"] = predictions
#                 st.success("Predictions Completed!")
#                 st.dataframe(input_df.head())
#                 csv_download = input_df.to_csv(index=False).encode("utf-8")
#                 st.download_button(
#                     label="Download Predictions as CSV",
#                     data=csv_download,
#                     file_name="predicted_revenue.csv",
#                     mime="text/csv"
#                 )


if selected == "Insights":
    st.header("Interactive Insights & Business Use Cases")

    st.markdown("Use the controls below to explore how different features affect ad revenue.")

    if "category" in df_clean.columns:
        # st.subheader("Revenue by Category")
        st.markdown("""
            <div style='background-color: rgba(255,255,255,0.04); 
                        border: 1px solid #FF0000; 
                        padding: 16px; 
                        border-radius: 12px;
                        margin-bottom: 25px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3 style='color: white ; text-align: center;'>Revenue by Category</h3>
        """, unsafe_allow_html=True)

        selected_cats = st.multiselect(
            "Select Categories to Compare:",
            options=df_clean["category"].unique(),
            default=list(df_clean["category"].unique())[:5]  # first 5 as default
        )
        cat_df = df_clean[df_clean["category"].isin(selected_cats)]
        cat_rev = cat_df.groupby("category")[target].mean().reset_index()

        fig_cat = px.bar(
            cat_rev,
            x="category",
            y=target,
            title="Average Revenue by Selected Categories",
            text_auto=".2f",
            color="category",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_cat.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_cat, use_container_width=True)

    st.divider()

    if "country" in df_clean.columns:
        # st.subheader("Revenue by Country")
        st.markdown("""
            <div style='background-color: rgba(255,255,255,0.04); 
                        border: 1px solid #FF0000; 
                        padding: 16px; 
                        border-radius: 12px;
                        margin-bottom: 25px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3 style='color: white ; text-align: center;'>Revenue by Country</h3>
        """, unsafe_allow_html=True)

        top_n = st.slider("Show Top N Countries", min_value=3, max_value=5, value=3)
        country_rev = df_clean.groupby("country")[target].mean().reset_index().sort_values(by=target, ascending=False).head(top_n)

        fig_country = px.bar(
            country_rev,
            x="country",
            y=target,
            title=f"Top {top_n} Countries by Average Revenue",
            text_auto=".2f",
            color="country",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_country.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_country, use_container_width=True)

    st.divider()

    if "device" in df_clean.columns:
        # st.subheader("Revenue by Device")
        st.markdown("""
            <div style='background-color: rgba(255,255,255,0.04); 
                        border: 1px solid #FF0000; 
                        padding: 16px; 
                        border-radius: 12px;
                        margin-bottom: 25px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3 style='color: white ; text-align: center;'>Revenue by Device</h3>
        """, unsafe_allow_html=True)

        show_as = st.radio("Display As:", ["Bar Chart", "Pie Chart"], horizontal=True)

        device_rev = df_clean.groupby("device")[target].mean().reset_index()

        if show_as == "Bar Chart":
            fig_device = px.bar(
                device_rev,
                x="device",
                y=target,
                title="Average Revenue by Device",
                text_auto=".2f",
                color="device",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
        else:
            fig_device = px.pie(
                device_rev,
                values=target,
                names="device",
                title="Revenue Share by Device",
                color_discrete_sequence=px.colors.sequential.RdBu
            )

        fig_device.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_device, use_container_width=True)

    st.divider()

    if model_choice == "Random Forest":
        # st.subheader("Feature Importance (Random Forest)")
        st.markdown("""
            <div style='background-color: rgba(255,255,255,0.04); 
                        border: 1px solid #FF0000; 
                        padding: 16px; 
                        border-radius: 12px;
                        margin-bottom: 25px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3 style='color: white ; text-align: center;'>Feature Importance (Random Forest)</h3>
        """, unsafe_allow_html=True)
        model = pipeline.named_steps["regressor"]
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)

        top_features = st.slider("Show Top N Features", min_value=3, max_value=len(features), value=8)
        importances = importances.tail(top_features)

        fig_imp = px.bar(
            importances,
            x=importances.values,
            y=importances.index,
            orientation="h",
            title=f"Top {top_features} Influential Features",
            color=importances.values,
            color_continuous_scale="reds"
        )
        fig_imp.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_imp, use_container_width=True)


    # st.subheader("Revenue Distribution by Category")
    st.markdown("""
            <div style='background-color: rgba(255,255,255,0.04); 
                        border: 1px solid #FF0000; 
                        padding: 16px; 
                        border-radius: 12px;
                        margin-bottom: 25px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3 style='color: white ; text-align: center;'>Revenue Distribution by Category</h3>
        """, unsafe_allow_html=True)
    fig_box = px.box(
        df_clean,
        x="category" if "category" in df_clean.columns else None,
        y=target,
        color="category" if "category" in df_clean.columns else None,
        points="all",
        title="Revenue Distribution Across Categories"
    )
    fig_box.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(fig_box, use_container_width=True)

    if "device" in df_clean.columns and "country" in df_clean.columns:
        # st.subheader("Device Usage by Country")

        st.markdown("""
            <div style='background-color: rgba(255,255,255,0.04); 
                        border: 1px solid #FF0000; 
                        padding: 16px; 
                        border-radius: 12px;
                        margin-bottom: 25px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3 style='color: white ; text-align: center;'>Device Usage by Country</h3>
        """, unsafe_allow_html=True)
        device_country = df_clean.groupby(["country", "device"])[target].mean().reset_index()
        fig_combo = px.bar(
            device_country,
            x="country",
            y=target,
            color="device",
            barmode="group",
            title="Revenue by Device in Each Country"
        )
        fig_combo.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_combo, use_container_width=True)

    if "video_length_minutes" in df_clean.columns:
        # st.subheader("Video Length Impact")
        st.markdown("""
            <div style='background-color: rgba(255,255,255,0.04); 
                        border: 1px solid #FF0000; 
                        padding: 16px; 
                        border-radius: 12px;
                        margin-bottom: 25px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3 style='color: white ; text-align: center;'>Video Length Impact</h3>
        """, unsafe_allow_html=True)
        fig_len = px.scatter(
            df_clean,
            x="video_length_minutes",
            y=target,
            color="category" if "category" in df_clean.columns else None,
            title="Video Length vs Revenue",
            opacity=0.6
        )
        fig_len.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_len, use_container_width=True) 

    # st.subheader("Engagement Impact on Revenue")
    st.markdown("""
            <div style='background-color: rgba(255,255,255,0.04); 
                        border: 1px solid #FF0000; 
                        padding: 16px; 
                        border-radius: 12px;
                        margin-bottom: 25px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3 style='color: white ; text-align: center;'>Engagement Impact on Revenue</h3>
        """, unsafe_allow_html=True)
    x_axis = st.selectbox("Choose Engagement Metric:", ["views", "likes", "comments", "watch_time_minutes"])
    fig_eng = px.scatter(
        df_clean,
        x=x_axis,
        y=target,
        size="subscribers" if "subscribers" in df_clean.columns else None,
        color="category" if "category" in df_clean.columns else None,
        title=f"{x_axis.capitalize()} vs Ad Revenue",
        opacity=0.6
    )
    fig_eng.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(fig_eng, use_container_width=True)   

    

# Footer
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        # border-top: 1px solid #ddd;
    }
    </style>

    <div class="footer">
        Made with ❤️
    </div>
""", unsafe_allow_html=True)    