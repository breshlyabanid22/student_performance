import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('student_performance.csv')

# Button to toggle Dataset Preview visibility
if st.sidebar.button("Show/Hide Dataset Preview"):
    st.session_state.show_preview = not st.session_state.show_preview

if 'StudentID' in df.columns:
    df = df.drop('StudentID', axis=1)

# Initial state for showing dataset preview
if 'show_preview' not in st.session_state:
    st.session_state.show_preview = False

# Display dataset preview if toggle is set to True
if st.session_state.show_preview:
    st.subheader("Dataset Preview")
    st.write("Here is a quick preview of the dataset:")
    st.dataframe(df)

# Select numeric columns for analysis
numeric_cols = df.select_dtypes(include=[int, float])

# Sidebar - Choose Page
st.sidebar.header("Visualization")
page = st.sidebar.radio("Navigate to a page:", ["Home", "Histograms", "Box Plot", "Correlation Heatmap", "Grade Prediction"])

if page == "Home":
    st.title("Student Performance Data Exploration")
    container = st.container(border=True)
    container.header("Introduction")
    container.write("""
    This dataset, sourced from Kaggle, contains information about student performance, including various factors such as attendance rate, study hours per week, previous grades, extracurricular activities, and parental support.
    The purpose of this exploration is to gain insights into the key factors influencing students' final grades and identify any potential correlations between these variables.
    """)

elif page == "Histograms":
    st.subheader("Histograms of Numeric Columns")
    st.write("These histograms show the distribution of various numeric variables in the dataset.")
    fig, axes = plt.subplots(len(numeric_cols.columns), 1, figsize=(8, 4 * len(numeric_cols.columns)))
    for i, col in enumerate(numeric_cols.columns):
        axes[i].hist(df[col], bins=5, color='skyblue')
        axes[i].set_title(f"Histogram of {col}", fontsize=16)
        axes[i].set_xlabel(col, fontsize=12)
        axes[i].set_ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    for col in numeric_cols.columns:
        st.subheader(f"Histogram of {col}")
        expander = st.expander("See Interpretation")
        if col == "AttendanceRate":
            expander.write(f"""
                Attendance rates are clustered in the high range (mostly 80-95). You might see a peak in the 85-95 range, indicating that many students have good attendance.
            """)   
        elif col == "StudyHoursPerWeek":
            expander.write(f"""
                The histogram could show that most students study between 15 and 20 hours per week, with a few outliers like (30 hours) and (8 hours).
                The graph might have a peak around 15-20 hours, which would indicate mostly of students study a moderate amount of time.
            """)
        elif col == "PreviousGrade":
            expander.write(f"""
                The grades are spread across a range, with most students falling between 75 and 90. You might see a few peaks for students who consistently score in the mid-to-high range (e.g., 78, 85, 88, 90).
            """)
        elif col == "ExtracurricularActivities":
            expander.write(f"""
                There will likely be multiple peaks at levels 1 and 2, showing that many students are involved in some extracurricular activities.
                Lower involvement (level 0) and higher involvement (level 3) might show smaller bars, meaning fewer students are at those extremes.
            """)    
        elif col == "FinalGrade":
            expander.write(f"""
                A left-skewed distribution could occur here, with most students scoring between 80 and 92. 
                There will likely be a peak in the high-performance range (e.g., 85 to 92), meaning many students scored well in their final grades.
            """)

elif page == "Box Plot":
    st.subheader("Box Plots of Numeric Columns")
    st.write("These box plots help visualize the spread and outliers for the numeric columns.")
    fig, axes = plt.subplots(len(numeric_cols.columns), 1, figsize=(8, 4 * len(numeric_cols.columns)))
    for i, col in enumerate(numeric_cols.columns):
        sns.boxplot(data=numeric_cols, y=col, ax=axes[i], color='lightgreen')
        axes[i].set_title(f"Box Plot of {col}")
    st.pyplot(fig)
    
    for col in numeric_cols.columns:
        st.subheader(f"Box Plot of {col}")
        expander = st.expander("See Interpretation")
        if col == "AttendanceRate":
            expander.write(f"""
                The box will show the range of attendance rates (likely between 70 and 95).
                The median (middle 50%) is likely around 85-90, indicating that most students have a high attendance rate.
            """)   
        elif col == "StudyHoursPerWeek":
            expander.write(f"""
                The box plot will show a range from the minimum (8 hours) to the maximum (30 hours).
                The median is likely around 15-20 hours, meaning that half of the students study within this range.
                The lower quartile (Q1) might indicate students who study around 10 hours or fewer, while the upper quartile (Q3) may include students studying around 20+ hours.
            """)
        elif col == "PreviousGrade":
            expander.write(f"""
                The box plot will show previous grades ranging from 60 (minimum) to 90 (maximum).
                The median might fall around 80, indicating that most students previously performed well.
            """)
        elif col == "ExtracurricularActivities":
            expander.write(f"""
                The box plot here will show levels from 0 (no extracurricular involvement) to 3 (high involvement).
                The median might fall around 1 or 2, indicating that most students are moderately involved in extracurricular activities.
                The box plot might have a wide spread, as some students are not involved at all (0), while others are very involved (3).
            """)    
        elif col == "FinalGrade":
            expander.write(f"""
                The box plot will show final grades ranging from around 62 (minimum) to 92 (maximum).
                The median is likely around 85-90, suggesting that many students perform well.
            """)


elif page == "Correlation Heatmap":
    st.subheader("Correlation Heatmap")
    st.write("This heatmap displays the correlation between numeric variables in the dataset.")
    corr_matrix = numeric_cols.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    def interpret_correlation(var1, var2, corr_value):
        if corr_value > 0.7:
            return f"**{var1} vs {var2}:** Strong positive correlation ({corr_value:.2f}). This indicates that as {var1} increases, {var2} tends to increase significantly."
        elif 0.3 < corr_value <= 0.7:
            return f"**{var1} vs {var2}:** Moderate positive correlation ({corr_value:.2f}). This suggests that there is a positive relationship between {var1} and {var2}, but it's not very strong."
        elif 0 < corr_value <= 0.3:
            return f"**{var1} vs {var2}:** Weak positive correlation ({corr_value:.2f}). There is a slight positive relationship between {var1} and {var2}."
        elif corr_value < -0.7:
            return f"**{var1} vs {var2}:** Strong negative correlation ({corr_value:.2f}). This suggests that as {var1} increases, {var2} tends to decrease significantly."
        elif -0.7 <= corr_value < -0.3:
            return f"**{var1} vs {var2}:** Moderate negative correlation ({corr_value:.2f}). This suggests a negative relationship between {var1} and {var2}, but not very strong."
        elif -0.3 <= corr_value < 0:
            return f"**{var1} vs {var2}:** Weak negative correlation ({corr_value:.2f}). There's a slight negative relationship between {var1} and {var2}."
        else:
            return f"**{var1} vs {var2}:** No significant correlation ({corr_value:.2f}). These variables do not seem to be linearly related."

    # Automatic Interpretation of Correlations
    st.subheader("Automatic Interpretation of Correlations")
    for i in range(len(numeric_cols.columns)):
        for j in range(i + 1, len(numeric_cols.columns)):
            var1 = numeric_cols.columns[i]
            var2 = numeric_cols.columns[j]
            corr_value = corr_matrix.loc[var1, var2]
            interpretation = interpret_correlation(var1, var2, corr_value)
            expander = st.expander(f"See Interpretation for {var1} vs {var2}")
            expander.write(interpretation)

elif page == "Grade Prediction":
    st.header("Predict Student Final Grade Based on Key Factors")
    
    # Input features for grade prediction
    st.write("Enter the values for the features below to predict the final grade:")
    
    attendance_rate = st.number_input("Attendance Rate", min_value=0, max_value=100, value=85)
    study_hours = st.number_input("Study Hours per Week", min_value=0, max_value=40, value=15)
    prev_grade = st.number_input("Previous Grade", min_value=0, max_value=100, value=80)
    extracurricular = st.number_input("Extracurricular Activities (0-3 scale)", min_value=0, max_value=3, value=2)
    
    # Prepare data for prediction
    X = df[['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'ExtracurricularActivities']]
    y = df['FinalGrade']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict final grade based on input
    input_features = pd.DataFrame([[attendance_rate, study_hours, prev_grade, extracurricular]], 
                                  columns=['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'ExtracurricularActivities'])
    predicted_grade = model.predict(input_features)[0]
    
    # Display predicted grade
    st.write(f"**Predicted Final Grade**: {predicted_grade:.2f}")
    
    # Show model performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"**Model Mean Squared Error**: {mse:.2f}")

# Conclusion Section
conclusion = st.sidebar.checkbox("Conclusion")
if conclusion:
    st.header("Conclusion")
    st.write("""
    Based on the analysis of the student performance dataset, we can observe several key insights:
    1. Variables such as **AttendanceRate**, **StudyHoursPerWeek**, and **PreviousGrade** seem to have a strong correlation with **FinalGrade**.
    2. The correlation heatmap shows that students with higher parental support tend to perform better.
    3. Outliers can be identified using box plots, providing insights into extreme cases of student performance.

    These insights can help educators focus on key areas to improve student outcomes.
    """)