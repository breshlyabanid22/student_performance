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

df = pd.read_csv('student_performance.csv')

if 'StudentID' in df.columns:
    df = df.drop('StudentID', axis=1)

if 'show_preview' not in st.session_state:
    st.session_state.show_preview = False

# Button to toggle Dataset Preview visibility
if st.sidebar.button("Show/Hide Dataset Preview"):
    st.session_state.show_preview = not st.session_state.show_preview

    # # Display dataset preview if toggle is set to True
if st.session_state.show_preview:
    st.subheader("Dataset Preview")
    st.write("Here is a quick preview of the dataset:")
    st.dataframe(df)

# Select numeric columns for analysis
numeric_cols = df.select_dtypes(include=[int, float])

    # Visualization selection dropdown
st.sidebar.title("Visualizations")
graph_type = st.sidebar.selectbox("Choose the type of graph:", ["Histogram", "Box Plot", "Correlation Heatmap"  ])

    # Display visualizations based on selected type
    
if graph_type == "Histogram":
    st.subheader("Histograms of Numeric Columns")
    st.write("These histograms show the distribution of various numeric variables in the dataset.")
    fig, axes = plt.subplots(len(numeric_cols.columns), 1, figsize=(8, 4 * len(numeric_cols.columns)))
    for i, col in enumerate(numeric_cols.columns):
        axes[i].hist(df[col], bins=5, color='skyblue')
        axes[i].set_title(f"Histogram of {col}", fontsize=16)
        axes[i].set_xlabel(col, fontsize=12)  # X-axis label
        axes[i].set_ylabel("Frequency", fontsize=12)  # Y-axis label
    plt.tight_layout()
    st.pyplot(fig)
    
    for col in numeric_cols.columns:
        st.subheader(f"Histogram of {col}")
        if col == "AttendanceRate":
            st.write(f"""
                Attendance rates are clustered in the high range (mostly 80-95), You might see a peak in the 85-95 range, indicating that many students have good attendance.
            """)   
        elif col == "StudyHoursPerWeek":
            st.write(f"""
                The histogram could show that most students study between 15 and 20 hours per week, with a few outliers like (30 hours) and (8 hours).
                The graph might have a peak around 15–20 hours, which would indicate that the majority of students study a moderate amount of time.
            """)
        elif col == "PreviousGrade":
            st.write(f"""
                The grades are spread across a range, with most students falling between 75 and 90, you might see a few peaks for students who consistently score in the mid-to-high range (e.g., 78, 85, 88, 90).
            """)
        elif col == "ExtracurricularActivities":
            st.write(f"""
                There will likely be multiple peaks at levels 1 and 2, showing that many students are involved in some extracurricular activities.
                Lower involvement (level 0) and higher involvement (level 3) might show smaller bars, meaning fewer students are at those extremes.
            """)    
        elif col == "FinalGrade":
            st.write(f"""
                A left-skewed distribution could occur here, with most students scoring between 80 and 92. 
                There will likely be a peak in the high-performance range (e.g., 85 to 92), meaning many students scored well in their final grades.
            """)            

elif graph_type == "Box Plot":
    st.subheader("Box Plots of Numeric Columns")
    st.write("These box plots help visualize the spread and outliers for the numeric columns.")
    fig, axes = plt.subplots(len(numeric_cols.columns), 1, figsize=(8, 4 * len(numeric_cols.columns)))
    for i, col in enumerate(numeric_cols.columns):
        sns.boxplot(data=numeric_cols, y=col, ax=axes[i], color='lightgreen')
        axes[i].set_title(f"Box Plot of {col}")
    st.pyplot(fig)

elif graph_type == "Correlation Heatmap":
    st.subheader("Correlation Heatmap")
    st.write("This heatmap displays the correlation between numeric variables in the dataset.")
    corr_matrix = numeric_cols.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    
    conclusion = st.sidebar.toggle("Conclusion")
    if conclusion:
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
