import streamlit as st
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import dagshub
import shap
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import metrics


dagshub.init(repo_owner='dx2119', repo_name='dsfe-final', mlflow=True)

st.set_page_config(
    page_title="Air Quality",
    layout="centered",
)

## Step 01 - Setup
st.sidebar.title("Air Quality")
st.sidebar.image("air.jpeg")
page = st.sidebar.selectbox("Select Page",["Introduction","Visualization", "Prediction",
        "Explainability",
        "MLflow Runs",
])



st.write("   ")
st.write("   ")
st.write("   ")
df = pd.read_csv("Air_Quality.csv")

# CO2 is missing 43056 entries so we dropped it from all visulizations and models
if 'CO2' in df.columns:
    df.drop(['CO2'], axis=1, inplace=True)


## Step 02 - Load dataset
# only include this if you want to turn the city into numbers
dfPrediction = df.copy()
dfPrediction['City_Name'] = dfPrediction['City']  # Keep original names for display
le = LabelEncoder()
dfPrediction['City'] = le.fit_transform(dfPrediction['City'])  # Numerical version for modeling

if page == "Introduction":
    df2 = pd.read_csv("Air_Quality.csv")

    st.subheader("01 Introduction")
    st.markdown("Air pollution causes approximately 7 million premature deaths annually (WHO). This dataset contains records of pollutants and European Air Quality Index (AQI) through January to December 2024 and includes cities from all inhabited continents. Our goal is to predict AQI based on pollutant levels, identify which pollutants most impact AQI, and compare air quality trends across continents.")


    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display",5,20,5)
    st.dataframe(df2.head(rows))

    st.markdown("##### Missing values")
    missing = df2.isnull().sum()
    st.write(missing)

    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ you have missing values")

    st.markdown("##### Summary Statistics")
    if st.toggle("Show Describe Table"):
        st.dataframe(df2.describe())

elif page == "Visualization":

    ## Step 03 - Data Viz
    st.subheader("02 Air Quality Index (AQI) Distribution by City")

    cities = df['City'].unique()
    selected_city = st.selectbox("Select a City", cities)

    # Filter data for the selected city
    city_data = df[df['City'] == selected_city]

    tab1, tab2, tab3 = st.tabs(["Histogram","Bar Chart","Correlation Heatmap"])

    with tab1:
        st.subheader("Distribution of AQI")
        # Plotting
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(city_data['AQI'], bins=30, kde=True, ax=ax)
        ax.set_title(f'Distribution of AQI Values in {selected_city}')
        ax.set_xlabel('Air Quality Index (AQI)')
        ax.set_ylabel('Frequency')
        ax.grid(True)

        st.pyplot(fig)

    with tab2:
        st.subheader("Average AQI by Month")

        # Convert 'Date' to datetime and extract month name
        city_data['Date'] = pd.to_datetime(city_data['Date'], errors='coerce')
        city_data = city_data.dropna(subset=['Date'])

        # Extract month names (e.g., Jan, Feb) and create a new column
        city_data['Month'] = city_data['Date'].dt.strftime('%b')  # 'Jan', 'Feb', etc.
        city_data['Month_Num'] = city_data['Date'].dt.month       # Numeric month for sorting

        # Group by Month_Num and Month for average AQI
        monthly_avg = city_data.groupby(['Month_Num', 'Month'])['AQI'].mean().reset_index()
        monthly_avg = monthly_avg.sort_values('Month_Num')  # Ensure months are in calendar order

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(monthly_avg['Month'], monthly_avg['AQI'], width=0.6)
        ax.set_title(f"Average Monthly AQI in {selected_city}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Average AQI")
        ax.grid(axis='y')

        st.pyplot(fig)


    with tab3:
        st.subheader("City-wise Correlation Matrix of AQI Dataset")
        # Filter and copy data for the selected city
        city_df_corr = df[df['City'] == selected_city].copy()

        # Drop 'City' and 'Date' columns if they exist
        drop_cols = [col for col in ['City', 'Date'] if col in city_df_corr.columns]
        city_df_corr.drop(columns=drop_cols, inplace=True)

        # Plot correlation matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(city_df_corr.corr().round(2), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
        ax.set_title(f'Correlation Matrix for {selected_city}')
        st.pyplot(fig)

elif page == "Prediction":
    st.subheader("03 Prediction with Model Comparison")

    # Sidebar: model selection
    model_name = st.sidebar.selectbox(
        "Choose Regression Model",
        ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"]
    )

    # Sidebar: hyperparameters
    params = {}
    if model_name == "Decision Tree":
        params['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
    elif model_name == "Random Forest":
        params['n_estimators'] = st.sidebar.slider("Number of Estimators", 10, 200, 100)
        params['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
    elif model_name == "XGBoost":
        params['n_estimators'] = st.sidebar.slider("Number of Estimators", 10, 200, 100)
        params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01)

    # Model instantiation function
    def get_model(name, params):
        if name == "Linear Regression":
            return LinearRegression()
        elif name == "Decision Tree":
            return DecisionTreeRegressor(**params, random_state=42)
        elif name == "Random Forest":
            return RandomForestRegressor(**params, random_state=42)
        elif name == "XGBoost":
            return XGBRegressor(objective="reg:squarederror", **params, random_state=42)

    # Store results per city
    results = {}
    for city_name in dfPrediction["City_Name"].unique():
        city_df = dfPrediction[dfPrediction["City_Name"] == city_name]
        X = city_df.drop(["AQI", "Date", "City_Name"], axis=1)
        y = city_df["AQI"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = get_model(model_name, params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[city_name] = {
            'model': model,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        }

    # Plotting function
    def plot_city(city_name):
        y_test = results[city_name]["y_test"]
        y_pred = results[city_name]["y_pred"]
        plot_df = y_test.copy().to_frame(name="Actual")
        plot_df["Predicted"] = y_pred
        plot_df["Error"] = abs(plot_df["Actual"] - plot_df["Predicted"])

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=plot_df,
            x="Actual",
            y="Predicted",
            hue="Error",
            palette="plasma",
            legend=True
        )
        plt.plot(
            [plot_df["Actual"].min(), plot_df["Actual"].max()],
            [plot_df["Actual"].min(), plot_df["Actual"].max()],
            "--",
            color="gray"
        )
        plt.xlabel("Actual AQI")
        plt.ylabel("Predicted AQI")
        plt.title(f"{city_name}: Actual vs Predicted AQI ({model_name})")
        st.pyplot(plt.gcf())

    # Stats display
    def display_city_stats(city_name):
        stats = results[city_name]
        st.markdown(f"#### Model Performance for {city_name}:")
        col1, col2, col3 = st.columns(3)
        col1.metric("MSE", f"{stats['MSE']:.2f}")
        col2.metric("MAE", f"{stats['MAE']:.2f}")
        col3.metric("R²", f"{stats['R2']:.3f}")

    # Tabs for each city
    cities = list(results.keys())
    tabs = st.tabs(cities)
    for i, city in enumerate(cities):
        with tabs[i]:
            st.subheader(f"{city}")
            plot_city(city)
            display_city_stats(city)

elif page == "Explainability":
    st.subheader("04 Explainability")
    st.markdown("Use SHAP (SHapley Additive exPlanations) to understand which features most influence AQI predictions for each city.")

    city_names = dfPrediction["City_Name"].unique().tolist()
    selected_city = st.selectbox("Select a City", city_names)

    df_exp = dfPrediction[dfPrediction["City_Name"] == selected_city].copy()

    df_exp = df_exp.sample(min(1000, len(df_exp)), random_state=42)

    X = df_exp.drop(columns=["AQI", "Date", "City_Name"])
    y = df_exp["AQI"]

    model = XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    model.fit(X, y)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    top_features = shap_values.abs.mean(0).values.argsort()[::-1][:5]
    top_feature_names = X.columns[top_features].tolist()

    st.markdown("### SHAP Results for: " + selected_city)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Top 5 Features by Mean Impact:**")
        st.dataframe(pd.DataFrame({
            "Feature": top_feature_names,
            "Mean |SHAP|": shap_values.abs.mean(0).values[top_features].round(4)
        }).reset_index(drop=True))

    with col2:
        st.markdown("#### SHAP Waterfall Plot (First Prediction)")
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(plt.gcf())

    st.divider()
    st.markdown("#### SHAP Beeswarm Plot (Distribution of SHAP Values)")
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(plt.gcf())

    st.divider()
    scatter_features = [col for col in X.columns if col != "City"]
    feature_for_scatter = st.selectbox("Select feature for SHAP scatter plot", scatter_features)
    st.markdown(f"#### SHAP Scatter Plot for '{feature_for_scatter}'")
    shap.plots.scatter(shap_values[:, feature_for_scatter], color=shap_values, show=False)
    st.pyplot(plt.gcf())


elif page == "MLflow Runs":
    st.subheader("05 MLflow Runs")

    # Load MLflow runs
    runs = mlflow.search_runs(order_by=["start_time desc"])

    # Optional raw data viewer
    with st.expander("Show Raw Run Table"):
        st.dataframe(runs)

    # Clean and select relevant columns
    df_clean = runs[[
        "tags.mlflow.runName", "params.model", "metrics.mse", "metrics.mae", "metrics.r2"
    ]].rename(columns={
        "tags.mlflow.runName": "City",
        "params.model": "Model",
        "metrics.mse": "MSE",
        "metrics.mae": "MAE",
        "metrics.r2": "R²"
    })

    # List unique model types
    model_types = df_clean["Model"].dropna().unique().tolist()
    tabs = st.tabs(model_types)

    for i, model_name in enumerate(model_types):
        with tabs[i]:
            st.markdown(f"### Results for `{model_name}`")

            model_df = df_clean[df_clean["Model"] == model_name].sort_values("R²", ascending=False)

            # Display summary table
            st.dataframe(model_df.reset_index(drop=True))

            # Plot MAE and R²
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            sns.barplot(data=model_df, y="City", x="MAE", ax=axes[0], palette="crest")
            axes[0].set_title("Mean Absolute Error (MAE)")
            axes[0].invert_yaxis()

            sns.barplot(data=model_df, y="City", x="R²", ax=axes[1], palette="flare")
            axes[1].set_title("R² Score")
            axes[1].invert_yaxis()

            st.pyplot(fig)

    st.markdown(
        "View detailed runs on DagsHub: [dx2119/dsfe-final MLflow](https://dagshub.com/dx2119/dsfe-final.mlflow)"
    )