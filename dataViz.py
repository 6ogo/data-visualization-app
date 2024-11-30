import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Data Visualization App", layout="wide")
st.title('Interactive Data Visualization App')

# File upload section
uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Detect file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            # CSV import options
            separator = st.selectbox("Select separator", [",", ";", "\t", "|"])
            encoding = st.selectbox("Select encoding", ["utf-8", "ISO-8859-1", "latin1"])
            df = pd.read_csv(uploaded_file, sep=separator, encoding=encoding)
        else:
            # Excel file
            sheet_name = None
            xls = pd.ExcelFile(uploaded_file)
            if len(xls.sheet_names) > 1:
                sheet_name = st.selectbox("Select sheet", xls.sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)

        st.success("Data loaded successfully!")
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Data info
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Number of rows: {df.shape[0]}")
            st.write(f"Number of columns: {df.shape[1]}")
        with col2:
            st.write(f"Column names: {', '.join(df.columns)}")
        
        # Visualization section
        st.subheader("Create Visualization")
        
        # Select plot type
        plot_type = st.selectbox(
            "Select the type of plot",
            ["Line Plot", "Scatter Plot", "Bar Plot", "Box Plot", "Violin Plot", 
             "Histogram", "Count Plot", "Heat Map", "Pair Plot"]
        )
        
        # All columns are available for selection
        all_cols = df.columns.tolist()
        
        if plot_type == "Line Plot":
            st.write("Select variables for the line plot:")
            
            # X-axis selection
            x_var = st.selectbox("Select X-axis variable", all_cols)
            
            # Multiple Y-axis selection
            y_vars = st.multiselect("Select Y-axis variables (multiple selection possible)", all_cols)
            
            if y_vars:
                # Optional grouping variable
                group_var = st.selectbox("Select group variable (optional)", ["None"] + all_cols)
                
                # Plot options
                col1, col2 = st.columns(2)
                with col1:
                    plot_title = st.text_input("Plot title", "Line Plot")
                with col2:
                    line_style = st.selectbox("Line style", ["-", "--", "-.", ":"])
                
                # Create the plot
                fig, ax = plt.subplots(figsize=(12, 6))
                
                if group_var != "None":
                    # Plot lines for each group
                    for group in df[group_var].unique():
                        group_data = df[df[group_var] == group]
                        for y_var in y_vars:
                            plt.plot(group_data[x_var], group_data[y_var], 
                                   label=f'{y_var} - {group}', linestyle=line_style)
                else:
                    # Plot each selected y variable
                    for y_var in y_vars:
                        plt.plot(df[x_var], df[y_var], label=y_var, linestyle=line_style)
                
                plt.xlabel(x_var)
                plt.ylabel("Values")
                plt.title(plot_title)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

        elif plot_type == "Scatter Plot":
            col1, col2, col3 = st.columns(3)
            with col1:
                x_var = st.selectbox("Select X variable", all_cols)
            with col2:
                y_var = st.selectbox("Select Y variable", all_cols)
            with col3:
                hue_var = st.selectbox("Select color variable (optional)", ["None"] + all_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            if hue_var != "None":
                sns.scatterplot(data=df, x=x_var, y=y_var, hue=hue_var)
            else:
                sns.scatterplot(data=df, x=x_var, y=y_var)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

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
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df[selected_cols].corr(), annot=True, cmap='coolwarm', center=0)
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