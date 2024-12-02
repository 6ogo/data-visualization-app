import base64
import re
import json
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy import stats
from datetime import datetime
import statsmodels.api as sm
from statsmodels.stats.diagnostic import lilliefors

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
        
        # Get columns with missing values
        columns_with_missing = df.columns[df.isnull().any()].tolist()
        
        for column in columns_with_missing:
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
                    # Try to convert to numeric if the column is numeric
                    if pd.api.types.is_numeric_dtype(df[column]):
                        try:
                            custom_value = float(custom_value)
                        except ValueError:
                            st.error(f"Please enter a numeric value for {column}")
                            continue
                    df[column] = df[column].fillna(custom_value)
    else:
        st.write("No missing values found in the dataset.")
    
    return df

def save_config_to_file(config, filename):
    """Save visualization configuration to a file"""
    config_str = json.dumps(config)
    b64 = base64.b64encode(config_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}">Download Configuration</a>'
    return href

def load_config_from_file(uploaded_file):
    """Load visualization configuration from a file"""
    content = uploaded_file.read().decode()
    return json.loads(content)

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

# Advanced statistical analysis function
def perform_advanced_statistics(df, column):
    """Perform advanced statistical analysis on a numeric column"""
    stats_results = {}
    
    # Basic statistics
    stats_results['mean'] = df[column].mean()
    stats_results['median'] = df[column].median()
    stats_results['std'] = df[column].std()
    stats_results['skewness'] = stats.skew(df[column].dropna())
    stats_results['kurtosis'] = stats.kurtosis(df[column].dropna())
    
    # Normality tests
    try:
        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(df[column].dropna())
        stats_results['shapiro_test'] = {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'is_normal': shapiro_p > 0.05
        }
        
        # Lilliefors test
        lillie_stat, lillie_p = lilliefors(df[column].dropna())
        stats_results['lilliefors_test'] = {
            'statistic': lillie_stat,
            'p_value': lillie_p,
            'is_normal': lillie_p > 0.05
        }
    except Exception as e:
        stats_results['normality_test_error'] = str(e)
    
    # Outlier detection
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    stats_results['outliers'] = {
        'count': len(df[(df[column] < lower_bound) | (df[column] > upper_bound)]),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }
    
    return stats_results

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

def create_interactive_bar_plot(df, x_var, y_var, hue_var=None, title="Interactive Bar Plot"):
    """Create an interactive bar plot using Plotly"""
    if hue_var and hue_var != "None":
        fig = px.bar(df, x=x_var, y=y_var, color=hue_var, title=title,
                    barmode='group', template='plotly_white')
    else:
        fig = px.bar(df, x=x_var, y=y_var, title=title, template='plotly_white')
    
    fig.update_layout(
        hoverlabel=dict(bgcolor="white"),
        hovermode='closest'
    )
    return fig

def create_interactive_box_plot(df, x_var, y_var, hue_var=None, title="Interactive Box Plot"):
    """Create an interactive box plot using Plotly"""
    if hue_var and hue_var != "None":
        fig = px.box(df, x=x_var, y=y_var, color=hue_var, title=title,
                    template='plotly_white')
    else:
        fig = px.box(df, x=x_var, y=y_var, title=title, template='plotly_white')
    
    fig.update_layout(
        hoverlabel=dict(bgcolor="white"),
        hovermode='closest'
    )
    return fig

def create_interactive_violin_plot(df, x_var, y_var, hue_var=None, title="Interactive Violin Plot"):
    """Create an interactive violin plot using Plotly"""
    if hue_var and hue_var != "None":
        fig = px.violin(df, x=x_var, y=y_var, color=hue_var, title=title,
                       box=True, points="all", template='plotly_white')
    else:
        fig = px.violin(df, x=x_var, y=y_var, title=title,
                       box=True, points="all", template='plotly_white')
    
    fig.update_layout(
        hoverlabel=dict(bgcolor="white"),
        hovermode='closest'
    )
    return fig

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

        # Plot settings - Define this before configuration management
        plot_settings = {
            "Line Plot": {"library": "plotly", "interactive": True},
            "Scatter Plot": {"library": "plotly", "interactive": True},
            "Bar Plot": {"library": "plotly", "interactive": True},
            "Box Plot": {"library": "plotly", "interactive": True},
            "Violin Plot": {"library": "plotly", "interactive": True},
            "Histogram": {"library": "plotly", "interactive": True},
            "Count Plot": {"library": "plotly", "interactive": True},
            "Heat Map": {"library": "plotly", "interactive": True},
            "Pair Plot": {"library": "plotly", "interactive": True}
        }
        
        # Plot type selection
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

        # Now add configuration management UI after variables are defined
        st.sidebar.header("Configuration Management")
        save_config = st.sidebar.button("Save Current Configuration")
        if save_config:
            current_config = {
                'plot_type': plot_type,
                'plot_settings': plot_settings,
                'customization': {
                    'title': plot_title,
                    'x_label': x_axis_label,
                    'y_label': y_axis_label,
                    'width': plot_width,
                    'height': plot_height
                }
            }
            st.sidebar.markdown(save_config_to_file(current_config, "viz_config.json"), unsafe_allow_html=True)

        config_file = st.sidebar.file_uploader("Load Configuration", type=['json'])
        if config_file:
            loaded_config = load_config_from_file(config_file)
            st.session_state.update(loaded_config)
            st.success("Configuration loaded successfully!")

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
        
        # Add advanced statistics section
        st.header("Advanced Statistical Analysis")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column for advanced analysis", numeric_cols)
            if selected_col:
                stats_results = perform_advanced_statistics(df, selected_col)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Basic Statistics")
                    st.write(f"Mean: {stats_results['mean']:.2f}")
                    st.write(f"Median: {stats_results['median']:.2f}")
                    st.write(f"Standard Deviation: {stats_results['std']:.2f}")
                    st.write(f"Skewness: {stats_results['skewness']:.2f}")
                    st.write(f"Kurtosis: {stats_results['kurtosis']:.2f}")
                
                with col2:
                    st.subheader("Normality Tests")
                    if 'shapiro_test' in stats_results:
                        st.write("Shapiro-Wilk Test:")
                        st.write(f"p-value: {stats_results['shapiro_test']['p_value']:.4f}")
                        st.write(f"Normal distribution: {stats_results['shapiro_test']['is_normal']}")
                    
                    if 'lilliefors_test' in stats_results:
                        st.write("Lilliefors Test:")
                        st.write(f"p-value: {stats_results['lilliefors_test']['p_value']:.4f}")
                        st.write(f"Normal distribution: {stats_results['lilliefors_test']['is_normal']}")

                st.subheader("Outlier Analysis")
                st.write(f"Number of outliers: {stats_results['outliers']['count']}")
                st.write(f"Lower bound: {stats_results['outliers']['lower_bound']:.2f}")
                st.write(f"Upper bound: {stats_results['outliers']['upper_bound']:.2f}")
        
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
            "Violin Plot": {"library": "plotly", "interactive": True},
            "Histogram": {"library": "plotly", "interactive": True},
            "Count Plot": {"library": "plotly", "interactive": True},
            "Heat Map": {"library": "plotly", "interactive": True},
            "Pair Plot": {"library": "plotly", "interactive": True}
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
            x_col = st.selectbox("X-axis", all_cols)
            y_cols = st.multiselect("Y-axis", all_cols)
            if x_col and y_cols:
                fig = px.line(df, x=x_col, y=y_cols,
                            title=plot_title,
                            width=plot_width,
                            height=plot_height)
                fig.update_layout(
                    xaxis_title=x_axis_label if x_axis_label else x_col,
                    yaxis_title=y_axis_label if y_axis_label else ', '.join(y_cols)
                )
                st.plotly_chart(fig)
        
        elif plot_type == "Scatter Plot":
            x_col = st.selectbox("X-axis", all_cols)
            y_col = st.selectbox("Y-axis", all_cols)
            color_col = st.selectbox("Color by", ["None"] + all_cols)
            
            if x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col,
                               color=None if color_col == "None" else color_col,
                               title=plot_title,
                               width=plot_width,
                               height=plot_height)
                fig.update_layout(
                    xaxis_title=x_axis_label if x_axis_label else x_col,
                    yaxis_title=y_axis_label if y_axis_label else y_col
                )
                st.plotly_chart(fig)

        elif plot_type == "Bar Plot":
            x_var = st.selectbox("Select X variable", all_cols)
            y_var = st.selectbox("Select Y variable", all_cols)
            hue_var = st.selectbox("Select group variable (optional)", ["None"] + all_cols)
            
            fig = create_interactive_bar_plot(
                df, x_var, y_var, 
                None if hue_var == "None" else hue_var,
                plot_title
            )
            fig.update_layout(
                width=plot_width,
                height=plot_height,
                xaxis_title=x_axis_label if x_axis_label else x_var,
                yaxis_title=y_axis_label if y_axis_label else y_var
            )
            st.plotly_chart(fig)

        elif plot_type == "Box Plot":
            x_var = st.selectbox("Select X variable", all_cols)
            y_var = st.selectbox("Select Y variable", all_cols)
            hue_var = st.selectbox("Select group variable (optional)", ["None"] + all_cols)
            
            fig = create_interactive_box_plot(
                df, x_var, y_var,
                None if hue_var == "None" else hue_var,
                plot_title
            )
            fig.update_layout(
                width=plot_width,
                height=plot_height,
                xaxis_title=x_axis_label if x_axis_label else x_var,
                yaxis_title=y_axis_label if y_axis_label else y_var
            )
            st.plotly_chart(fig)

        elif plot_type == "Violin Plot":
            x_var = st.selectbox("Select X variable", all_cols)
            y_var = st.selectbox("Select Y variable", all_cols)
            hue_var = st.selectbox("Select group variable (optional)", ["None"] + all_cols)
            
            fig = create_interactive_violin_plot(
                df, x_var, y_var,
                None if hue_var == "None" else hue_var,
                plot_title
            )
            fig.update_layout(
                width=plot_width,
                height=plot_height,
                xaxis_title=x_axis_label if x_axis_label else x_var,
                yaxis_title=y_axis_label if y_axis_label else y_var
            )
            st.plotly_chart(fig)

        elif plot_type == "Histogram":
            x_var = st.selectbox("Select variable", all_cols)
            bins = st.slider("Number of bins", min_value=5, max_value=100, value=30)
            
            fig = px.histogram(df, x=x_var, nbins=bins,
                             title=plot_title,
                             width=plot_width,
                             height=plot_height)
            fig.update_layout(
                xaxis_title=x_axis_label if x_axis_label else x_var,
                yaxis_title=y_axis_label if y_axis_label else "Count"
            )
            st.plotly_chart(fig)

        elif plot_type == "Count Plot":
            x_var = st.selectbox("Select variable", all_cols)
            hue_var = st.selectbox("Select group variable (optional)", ["None"] + all_cols)
            
            if hue_var != "None":
                fig = px.histogram(df, x=x_var, color=hue_var,
                                 title=plot_title,
                                 width=plot_width,
                                 height=plot_height)
            else:
                fig = px.histogram(df, x=x_var,
                                 title=plot_title,
                                 width=plot_width,
                                 height=plot_height)
            
            fig.update_layout(
                xaxis_title=x_axis_label if x_axis_label else x_var,
                yaxis_title=y_axis_label if y_axis_label else "Count"
            )
            st.plotly_chart(fig)

        elif plot_type == "Heat Map":
            selected_cols = st.multiselect("Select variables for correlation heatmap", all_cols)
            
            if selected_cols:
                corr_matrix = df[selected_cols].corr()
                
                fig = px.imshow(corr_matrix,
                              title=plot_title,
                              width=plot_width,
                              height=plot_height,
                              color_continuous_scale="RdBu_r")
                
                fig.update_layout(
                    xaxis_title="Variables",
                    yaxis_title="Variables"
                )
                
                st.plotly_chart(fig)

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
                    fig = px.scatter_matrix(df, dimensions=selected_cols, color=hue_var,
                                          title=plot_title,
                                          width=plot_width,
                                          height=plot_height)
                else:
                    fig = px.scatter_matrix(df, dimensions=selected_cols,
                                          title=plot_title,
                                          width=plot_width,
                                          height=plot_height)
                
                st.plotly_chart(fig)

        # Save plot settings
        if st.button("Save Plot Settings"):
            st.session_state['plot_settings'] = save_plot_settings()
            st.success("Plot settings saved!")
        
        # Load saved settings
        if 'plot_settings' in st.session_state and st.button("Load Saved Settings"):
            st.info("Loading saved settings...")

    except Exception as e:
        st.error(f"Error loading or processing the file: {str(e)}")
        st.write("Please check your file format and try again.")
        st.write("Error details:", str(e))

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
