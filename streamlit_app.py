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
st.sidebar.markdown("By Julia Huang and Derek Xia")
st.sidebar.image("air.jpeg")
page = st.sidebar.selectbox("Select Page",[
        "Introduction",
        "Visualization", 
        "Prediction",
        "Explainability",
        "MLflow Runs",
        "Conclusion"
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

with st.container():
    if page == "Introduction":
        df2 = pd.read_csv("Air_Quality.csv")

        st.subheader("01 Introduction")
        st.markdown("Air pollution causes approximately 7 million premature deaths annually (WHO). This dataset contains records of pollutants and the European Air Quality Index (AQI) from January to December 2024 and includes cities from all inhabited continents. Our goal is to predict AQI based on pollutant levels, identify which pollutants most impact AQI, and compare air quality trends across continents.")


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

        def display_city_stats(city_name):
            stats = results[city_name]
            st.markdown(f"#### Model Performance for {city_name}:")
            col1, col2, col3 = st.columns(3)
            col1.metric("MSE", f"{stats['MSE']:.2f}")
            col2.metric("MAE", f"{stats['MAE']:.2f}")
            col3.metric("R²", f"{stats['R2']:.3f}")

            # 👇👇 Add markdown explanation here for Linear Regression
            if model_name == "Linear Regression":
                st.markdown(f"**Explanation for {city_name} (Linear Regression):**")
                if city_name == "Brasilia":
                    st.markdown("###### The model’s performance in Brasília reflects moderate predictive accuracy, with a mean squared error of 23.11 and a mean absolute error of 3.92, indicating reasonably close predictions with occasional larger deviations. The R² score of 0.749 shows that the model explains a substantial portion of the variation in AQI, suggesting that Brasília’s pollution patterns are relatively consistent and partially linear—possibly due to its planned urban design, lower industrial density, and controlled traffic flow. However, the residual error still highlights some unpredictable fluctuations, likely caused by sporadic environmental events or emissions not captured in the dataset. To improve the model, incorporating features like topographical influences, forest burning activity in nearby regions, or seasonal wind behavior could help better account for these irregularities.")
                if city_name == "Cairo":
                    st.markdown("###### The model performs poorly in Cairo, with a high mean squared error of 142.41 and a mean absolute error of 10.21, indicating frequent and sizable prediction errors. The R² score of 0.357 shows that the model struggles to capture more than a third of AQI variance, which aligns with the reality of Cairo’s complex pollution environment—marked by high population density, heavy traffic, industrial activity, and frequent desert dust intrusions. The non-linear spread in the predictions suggests that the city’s air quality is influenced by a combination of abrupt and overlapping factors that a simple linear model cannot effectively model. Improving the model would likely require integrating more granular inputs, such as real-time traffic flow, humidity levels, or particulate matter composition, to better reflect Cairo’s chaotic and multifactorial pollution landscape.")
                if city_name == "Dubai":
                    st.markdown("###### The model’s performance in Dubai is notably poor, with a mean squared error of 192.36 and a mean absolute error of 11.24, indicating consistently large gaps between predicted and actual AQI values. Most critically, the R² score is -1.108, meaning the model performs worse than simply predicting the average AQI every time—suggesting that the regression line fits the data very poorly. This failure likely reflects the complex and irregular pollution patterns in Dubai, where air quality is influenced by a mix of urban emissions, construction dust, industrial activity, and frequent desert sandstorms, none of which are well captured by a simple linear model. To meaningfully improve predictive accuracy, the model would need to incorporate domain-specific variables such as dust storm alerts, humidity, and construction activity levels, and likely use nonlinear or ensemble methods better suited to such erratic environmental conditions.")
                if city_name == "London":
                    st.markdown("###### The model’s performance in London is weak, with a high mean squared error of 177.82 and a mean absolute error of 8.40, indicating frequent and sizable deviations from actual AQI values. The low R² score of 0.141 shows that the model captures only a small fraction of the variation in air quality, failing to represent London’s AQI dynamics effectively. This poor fit may be due to London’s highly variable pollution sources, which include fluctuating traffic congestion, weather-driven dispersion, and pollution drift from surrounding areas. The systematic underprediction of higher AQI values suggests the model is not equipped to handle pollution spikes; incorporating variables like traffic intensity, wind direction, or temperature inversions may improve performance and make the model more suitable for capturing London’s complex urban air quality behavior.")
                if city_name == "New York":
                    st.markdown("###### The model’s performance in New York is poor, with a mean squared error of 130.02 and a mean absolute error of 6.96, indicating that predictions frequently deviate from actual AQI values by a moderate to large margin. The extremely low R² score of 0.042 shows that the model explains virtually none of the variation in air quality, suggesting a weak relationship between the inputs and AQI outcomes. This could reflect the complexity of New York’s pollution patterns, which are shaped by dense traffic, seasonal weather shifts, and building-induced microclimates that likely introduce nonlinearity and variability not captured by the model. Enhancing prediction accuracy would require incorporating temporal and spatial variables—such as traffic congestion, temperature fluctuations, or building density—to account for the unique urban dynamics that influence AQI in the city.")
                if city_name == "Sydney":
                    st.markdown("###### The model performs well in Sydney, with a low mean squared error of 30.51 and a mean absolute error of 4.21, indicating that predictions are generally accurate with only small deviations. The high R² score of 0.756 suggests that the model captures a strong majority of AQI variation, making it effective at modeling Sydney’s air quality. This success may be due to Sydney’s relatively stable pollution patterns, which are influenced by consistent weather systems, effective emission controls, and a lower frequency of extreme pollution events compared to more industrialized or densely populated cities. To further refine the model, especially in high AQI cases where residuals still widen, incorporating variables such as bushfire proximity or wind direction could improve responsiveness to short-term spikes.")

        # Create tabs for each city
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
    
    elif page == "Conclusion":
        st.markdown("## Conclusion")
        st.markdown("###### By modeling AQI using pollutant levels, we successfully addressed the problem of predicting air quality across global cities. To enhance model accuracy and capture local environmental differences, we chose to segment the dataset by city, which allows us to tailor models to each city's unique pollution profile. Through our analysis, we found that the most influential pollutants varied not only in magnitude but also in direction of their relationship with AQI.")
        st.image("cleanair.jpg", width=None)
        st.markdown("###### Future improvements could include hyperparameter tuning (e.g., grid search for tree depth or learning rate) and experimenting with deep learning models like fully connected neural networks. These could potentially capture complex, nonlinear pollutant-AQI relationships that tree-based models might miss.")
        
