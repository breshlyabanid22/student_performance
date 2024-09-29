import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# App Title
st.title("Student Performance Data Exploration")

# Introduction Section
st.header("Introduction")
st.write("""
This dataset, sourced from Kaggle, contains information about student performance, including various factors such as attendance rate, study hours per week, previous grades, extracurricular activities, and parental support.
The purpose of this exploration is to gain insights into the key factors influencing students' final grades and identify any potential correlations between these variables.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display dataset preview
    st.subheader("Dataset Preview")
    st.write("Here is a quick preview of the dataset:")
    st.dataframe(df)

    # Select numeric columns for analysis
    numeric_cols = df.select_dtypes(include=[int, float])

    # Visualizations Section
    st.header("Visualizations")

    # Histograms
    st.subheader("Histograms of Numeric Columns")
    st.write("These histograms show the distribution of various numeric variables in the dataset.")
    fig, axes = plt.subplots(len(numeric_cols.columns), 1, figsize=(8, 4 * len(numeric_cols.columns)))
    for i, col in enumerate(numeric_cols.columns):
        axes[i].hist(df[col], bins=10, color='skyblue')
        axes[i].set_title(f"Histogram of {col}", fontsize=14)
        axes[i].set_xlabel(col, fontsize=12)  # X-axis label
        axes[i].set_ylabel("Frequency", fontsize=12)  # Y-axis label

    plt.tight_layout()
    st.pyplot(fig)

    # Box plots
    st.subheader("Box Plots of Numeric Columns")
    st.write("These box plots help visualize the spread and outliers for the numeric columns.")
    fig, axes = plt.subplots(len(numeric_cols.columns), 1, figsize=(8, 4 * len(numeric_cols.columns)))
    for i, col in enumerate(numeric_cols.columns):
        sns.boxplot(data=numeric_cols, y=col, ax=axes[i], color='lightgreen')
        axes[i].set_title(f"Box Plot of {col}")
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    st.write("This heatmap displays the correlation between numeric variables in the dataset.")
    corr_matrix = numeric_cols.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Conclusion Section
    st.header("Conclusion")
    st.write("""
    Based on the analysis of the student performance dataset, we can observe several key insights:
    1. Variables such as **AttendanceRate**, **StudyHoursPerWeek**, and **PreviousGrade** seem to have a strong correlation with **FinalGrade**.
    2. The correlation heatmap shows that students with higher parental support tend to perform better.
    3. Outliers can be identified using box plots, providing insights into extreme cases of student performance.

    These insights can help educators focus on key areas to improve student outcomes.
    """)

else:
    st.write("Please upload a CSV file to proceed with the analysis.")
