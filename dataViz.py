import base64
import re
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Data Visualization App", layout="wide")
st.title('Interactive Data Visualization App')

# Sidebar for global settings
with st.sidebar:
    st.header("Settings")
    theme = st.selectbox("Color Theme", ["default", "plotly_dark", "plotly_white", "seaborn"])
    if theme != "default":
        plt.style.use(theme)

def prepare_for_plotting(df, x_var, y_vars=None):
    """Prepare data for plotting by ensuring proper numeric conversion"""
    df = df.copy()
    
    def to_numeric_if_possible(series):
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        try:
            return pd.to_numeric(series, errors='coerce')
        except:
            if not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series):
                return series.astype('category').cat.codes
            return series
    
    df[x_var] = to_numeric_if_possible(df[x_var])
    if y_vars:
        for y_var in y_vars:
            df[y_var] = to_numeric_if_possible(df[y_var])
    
    return df

def handle_missing_values(df):
    """Handle missing values in the dataframe"""
    st.subheader("Missing Values Treatment")
    
    # Display missing values information
    missing_stats = df.isnull().sum()
    if missing_stats.any():
        st.write("Missing values found in:")
        st.write(missing_stats[missing_stats > 0])
        
        for column in all_cols[df.isnull().any()]:
            st.write(f"\nHandling missing values in {column}:")
            method = st.selectbox(
                f"Choose method for {column}",
                ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with custom value"],
                key=f"missing_{column}"
            )
            
            if method == "Drop rows":
                df = df.dropna(subset=[column])
            elif method == "Fill with mean" and pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].mean())
            elif method == "Fill with median" and pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].median())
            elif method == "Fill with mode":
                df[column] = df[column].fillna(df[column].mode()[0])
            elif method == "Fill with custom value":
                custom_value = st.text_input(f"Enter value for {column}", key=f"custom_{column}")
                if custom_value:
                    df[column] = df[column].fillna(custom_value)
    else:
        st.write("No missing values found in the dataset.")
    
    return df

def data_transformation(df):
    """Apply data transformations to numeric columns"""
    st.subheader("Data Transformation")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        transform_cols = st.multiselect(
            "Select columns for transformation",
            numeric_columns
        )
        
        if transform_cols:
            transform_method = st.selectbox(
                "Select transformation method",
                ["StandardScaler", "MinMaxScaler", "Log Transform", "Square Root Transform"]
            )
            
            df_transformed = df.copy()
            
            if transform_method == "StandardScaler":
                scaler = StandardScaler()
                df_transformed[transform_cols] = scaler.fit_transform(df[transform_cols])
            elif transform_method == "MinMaxScaler":
                scaler = MinMaxScaler()
                df_transformed[transform_cols] = scaler.fit_transform(df[transform_cols])
            elif transform_method == "Log Transform":
                for col in transform_cols:
                    if (df[col] > 0).all():
                        df_transformed[f"{col}_log"] = np.log(df[col])
                    else:
                        st.warning(f"Log transform not possible for {col} due to non-positive values")
            elif transform_method == "Square Root Transform":
                for col in transform_cols:
                    if (df[col] >= 0).all():
                        df_transformed[f"{col}_sqrt"] = np.sqrt(df[col])
                    else:
                        st.warning(f"Square root transform not possible for {col} due to negative values")
            
            return df_transformed
    
    return df

def filter_data(df):
    """Add data filtering options"""
    st.subheader("Data Filtering")
    
    filtered_df = df.copy()
    
    # Numeric columns filtering
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if st.checkbox(f"Filter {col}", key=f"filter_{col}"):
            min_val, max_val = st.slider(
                f"Select range for {col}",
                float(df[col].min()),
                float(df[col].max()),
                (float(df[col].min()), float(df[col].max())),
                key=f"range_{col}"
            )
            filtered_df = filtered_df[filtered_df[col].between(min_val, max_val)]
    
    # Categorical columns filtering
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if st.checkbox(f"Filter {col}", key=f"filter_cat_{col}"):
            selected_values = st.multiselect(
                f"Select values for {col}",
                df[col].unique(),
                default=list(df[col].unique()),
                key=f"select_{col}"
            )
            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
    
    return filtered_df

def get_download_link(df, filename):
    """Generate a download link for the dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
    return href

def add_statistical_summary(df):
    """Add statistical summary of the data"""
    st.subheader("Statistical Summary")
    
    # Numeric summary
    numeric_summary = df.describe()
    st.write("Numeric Summary:")
    st.dataframe(numeric_summary)
    
    # Categorical summary
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        st.write("Categorical Summary:")
        for col in categorical_cols:
            st.write(f"\nValue counts for {col}:")
            st.write(df[col].value_counts())

def save_plot_settings():
    """Save current plot settings to session state"""
    plot_settings = {
        'plot_type': st.session_state.get('plot_type'),
        'x_var': st.session_state.get('x_var'),
        'y_var': st.session_state.get('y_var'),
        'hue_var': st.session_state.get('hue_var'),
        'other_settings': st.session_state.get('other_settings', {})
    }
    return plot_settings

# File upload section
uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])
st.write("Supported file types: CSV (.csv) and Excel (.xlsx)")

if uploaded_file:
    try:
        # File loading with configuration
        if uploaded_file.name.endswith('.csv'):
            separator = st.selectbox("Select separator", [",", ";", "\t", "|"])
            encoding = st.selectbox("Select encoding", ["utf-8", "ISO-8859-1", "latin1"])
            df = pd.read_csv(uploaded_file, sep=separator, encoding=encoding)
        else:
            sheet_name = None
            xls = pd.ExcelFile(uploaded_file)
            if len(xls.sheet_names) > 1:
                sheet_name = st.selectbox("Select sheet", xls.sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)

        # Define all_cols after loading the dataframe
        all_cols = df.columns.tolist()

        # Data preprocessing steps
        st.header("Data Preprocessing")
        
        # Handle missing values
        df = handle_missing_values(df)
        
        # Data transformation
        df = data_transformation(df)
        
        # Data filtering
        df = filter_data(df)
        
        # Statistical summary
        add_statistical_summary(df)
        
        # Data preview
        st.header("Data Preview")
        st.dataframe(df.head())
        
        # Download processed data
        st.markdown(get_download_link(df, "processed_data.csv"), unsafe_allow_html=True)
        
        # Visualization section
        st.header("Data Visualization")
        
        # Plot settings
        plot_settings = {
            "Line Plot": {"library": "plotly", "interactive": True},
            "Scatter Plot": {"library": "plotly", "interactive": True},
            "Bar Plot": {"library": "plotly", "interactive": True},
            "Box Plot": {"library": "plotly", "interactive": True},
            "Violin Plot": {"library": "seaborn", "interactive": False},
            "Histogram": {"library": "plotly", "interactive": True},
            "Count Plot": {"library": "seaborn", "interactive": False},
            "Heat Map": {"library": "seaborn", "interactive": False},
            "Pair Plot": {"library": "seaborn", "interactive": False}
        }
        
        plot_type = st.selectbox("Select plot type", list(plot_settings.keys()))
        
        # Plot customization options
        st.subheader("Plot Customization")
        col1, col2 = st.columns(2)
        with col1:
            plot_title = st.text_input("Plot title", "My Plot")
            x_axis_label = st.text_input("X-axis label")
            y_axis_label = st.text_input("Y-axis label")
        with col2:
            plot_width = st.slider("Plot width", 400, 1200, 800)
            plot_height = st.slider("Plot height", 300, 800, 500)
        
        # Create visualization based on type
        if plot_type == "Line Plot":
            fig = px.line(df, x=st.selectbox("X-axis", all_cols),
                         y=st.multiselect("Y-axis", all_cols),
                         title=plot_title,
                         width=plot_width,
                         height=plot_height)
            st.plotly_chart(fig)
        
        elif plot_type == "Scatter Plot":
            fig = px.scatter(df, x=st.selectbox("X-axis", all_cols),
                           y=st.selectbox("Y-axis", all_cols),
                           color=st.selectbox("Color by", ["None"] + list(all_cols)),
                           title=plot_title,
                           width=plot_width,
                           height=plot_height)
            st.plotly_chart(fig)

        elif plot_type == "Bar Plot":
            col1, col2, col3 = st.columns(3)
            with col1:
                x_var = st.selectbox("Select X variable", all_cols)
            with col2:
                y_var = st.selectbox("Select Y variable", all_cols)
            with col3:
                hue_var = st.selectbox("Select group variable (optional)", ["None"] + all_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            if hue_var != "None":
                sns.barplot(data=df, x=x_var, y=y_var, hue=hue_var)
            else:
                sns.barplot(data=df, x=x_var, y=y_var)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        elif plot_type == "Box Plot":
            col1, col2, col3 = st.columns(3)
            with col1:
                x_var = st.selectbox("Select X variable", all_cols)
            with col2:
                y_var = st.selectbox("Select Y variable", all_cols)
            with col3:
                hue_var = st.selectbox("Select group variable (optional)", ["None"] + all_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            if hue_var != "None":
                sns.boxplot(data=df, x=x_var, y=y_var, hue=hue_var)
            else:
                sns.boxplot(data=df, x=x_var, y=y_var)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        elif plot_type == "Violin Plot":
            col1, col2, col3 = st.columns(3)
            with col1:
                x_var = st.selectbox("Select X variable", all_cols)
            with col2:
                y_var = st.selectbox("Select Y variable", all_cols)
            with col3:
                hue_var = st.selectbox("Select group variable (optional)", ["None"] + all_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            if hue_var != "None":
                sns.violinplot(data=df, x=x_var, y=y_var, hue=hue_var)
            else:
                sns.violinplot(data=df, x=x_var, y=y_var)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        elif plot_type == "Histogram":
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Select variable", all_cols)
            with col2:
                bins = st.slider("Number of bins", min_value=5, max_value=100, value=30)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x=x_var, bins=bins)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        elif plot_type == "Count Plot":
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Select variable", all_cols)
            with col2:
                hue_var = st.selectbox("Select group variable (optional)", ["None"] + all_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            if hue_var != "None":
                sns.countplot(data=df, x=x_var, hue=hue_var)
            else:
                sns.countplot(data=df, x=x_var)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        elif plot_type == "Heat Map":
            # Let user select variables for correlation
            selected_cols = st.multiselect("Select variables for correlation heatmap", all_cols)
            
            if selected_cols:
                # Prepare numeric data for correlation
                numeric_df = df[selected_cols].copy()
                
                # Convert all columns to numeric where possible
                for col in numeric_df.columns:
                    if not pd.api.types.is_numeric_dtype(numeric_df[col]):
                        try:
                            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
                        except:
                            numeric_df[col] = numeric_df[col].astype('category').cat.codes
                
                # Create correlation matrix
                correlation_matrix = numeric_df.corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Heatmap')
                plt.tight_layout()
                st.pyplot(fig)

        elif plot_type == "Pair Plot":
            selected_cols = st.multiselect(
                "Select variables for pair plot (max 4)", 
                all_cols,
                default=all_cols[:2] if len(all_cols) >= 2 else all_cols,
                max_selections=4
            )
            
            if selected_cols:
                hue_var = st.selectbox("Select grouping variable (optional)", ["None"] + all_cols)
                if hue_var != "None":
                    fig = sns.pairplot(df[selected_cols + [hue_var]], hue=hue_var)
                else:
                    fig = sns.pairplot(df[selected_cols])
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading or processing the file: {str(e)}")
        st.write("Please check your file format and try again.")
        st.write("Error details:", str(e))  #for debugging

# Add help section
with st.expander("Help & Documentation"):
    st.markdown("""
    ### How to use this app:
    1. Upload your data file (CSV or Excel)
    2. Configure data preprocessing options
    3. Apply transformations if needed
    4. Create visualizations
    5. Customize and save your plots
    
    ### Features:
    - Missing value handling
    - Data transformation
    - Interactive filtering
    - Multiple visualization options
    - Plot customization
    - Settings save/load
    """)
