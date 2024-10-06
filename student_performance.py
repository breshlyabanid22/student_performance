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

if 'show_preview' not in st.session_state:
    st.session_state.show_preview = False



numeric_cols = df.select_dtypes(include=[int, float])


st.sidebar.header("Visualization")
page = st.sidebar.radio("Navigate to a page:", ["Home", "Histograms", "Box Plot", "Correlation Heatmap", "Line Chart", "Grade Prediction", "Conclusion"])

if page == "Home":
    st.title("Student Performance Data Exploration")
    container = st.container(border=True)
    container.header("Introduction")
    container.write("""
    This dataset, sourced from Kaggle, contains information about student performance, including various factors such as attendance rate, study hours per week, previous grades, extracurricular activities, and parental support.
    The purpose of this exploration is to gain insights into the key factors influencing students' final grades and identify any potential correlations between these variables.
    """)
    if st.session_state.show_preview:
        st.subheader("Dataset Preview")
        st.write("Here is a quick preview of the dataset:")
        st.dataframe(df)

elif page == "Histograms":
    if st.session_state.show_preview:
        st.subheader("Dataset Preview")
        st.write("Here is a quick preview of the dataset:")
        st.dataframe(df)

    st.subheader("Histograms of Numeric Columns")
    st.write("These histograms show the distribution of various numeric variables in the dataset.")
    fig, axes = plt.subplots(len(numeric_cols.columns), 1, figsize=(8, 4 * len(numeric_cols.columns)))
    for i, col in enumerate(numeric_cols.columns):
        axes[i].bar(df['Name'], df[col], color='skyblue')
        axes[i].set_title(f"Histogram of {col}", fontsize=16)
        axes[i].set_xlabel('Student Name', fontsize=12)
        axes[i].set_ylabel(col, fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    for col in numeric_cols.columns:
        st.subheader(f"Histogram of {col}")
        expander = st.expander("See Interpretation")
        if col == "AttendanceRate":
            expander.write(f"""
                Olivia has the highest attendance rate at 95%, followed by Michael at 92% and Isabella at 91%.
                Daniel has the lowest attendance rate at 70%. This could indicate that his lower grades are partly influenced by a lack of attendance.
                Overall, most students have attendance rates above 80%, indicating that attendance is not a major issue for the majority.
            """)   
        elif col == "StudyHoursPerWeek":
            expander.write(f"""
                Olivia studies the most with 30 hours per week, followed by Michael with 25 hours and Isabella with 22 hours.
                Daniel studies the least at only 8 hours per week, which could also contribute to his lower grades.
                Most other students have study hours between 10 to 20 hours, which seems to correspond reasonably well with their final grades, showing that study hours tend to have a positive impact.
            """)
        elif col == "PreviousGrade":
            expander.write(f"""
                Michael has the highest previous grade of 90, followed by Olivia at 88 and Isabella at 86. These students consistently perform well academically.
                Daniel has the lowest previous grade at 60, which correlates with his lower attendance rate and fewer study hours.
                Overall, the previous grades align with students' attendance rates and study hours, reinforcing that commitment and time spent studying are key factors in academic performance.
            """)
        elif col == "ExtracurricularActivities":
            expander.write(f"""
                Michael and Isabella are involved in the most extracurricular activities, with 3 activities each, and they both have very high final grades (92 and 88, respectively).
                Sarah and Emma participate in 2 extracurricular activities, and they also maintain high final grades (87 and 85).
                John, Olivia, and Sophia are involved in 1 extracurricular activity each, and their grades are relatively high as well.
                Alex and Daniel are not involved in any extracurricular activities, and they have some of the lowest final grades (68 and 62, respectively).
            """)    
        elif col == "FinalGrade":
            expander.write(f"""
                Michael again has the highest final grade of 92, showing consistent performance and slight improvement over his previous grade of 90.
                Isabella and Olivia both have strong final grades of 88 and 90, indicating steady improvement from their previous grades.
                Daniel has the lowest final grade of 62, showing that his performance hasn’t improved much and may even be a concern.
                Alex and James also have lower final grades (68 and 72 respectively), though both showed some improvement from their previous grades. This suggests that while they may have made some progress, they could still benefit from increased study time or attendance.
            """)

elif page == "Box Plot":
    if st.session_state.show_preview:
        st.subheader("Dataset Preview")
        st.write("Here is a quick preview of the dataset:")
        st.dataframe(df)
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
    if st.session_state.show_preview:
        st.subheader("Dataset Preview")
        st.write("Here is a quick preview of the dataset:")
        st.dataframe(df)
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
    
    st.write("Enter the values for the features below to predict the final grade:")
    
    attendance_rate = st.number_input("Attendance Rate", min_value=0, max_value=100, value=85)
    study_hours = st.number_input("Study Hours per Week", min_value=0, max_value=40, value=15)
    prev_grade = st.number_input("Previous Grade", min_value=0, max_value=100, value=80)
    extracurricular = st.number_input("Extracurricular Activities (0-3 scale)", min_value=0, max_value=3, value=2)
    
    X = df[['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'ExtracurricularActivities']]
    y = df['FinalGrade']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    input_features = pd.DataFrame([[attendance_rate, study_hours, prev_grade, extracurricular]], 
                                  columns=['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'ExtracurricularActivities'])
    predicted_grade = model.predict(input_features)[0]
    
    st.write(f"**Predicted Final Grade**: {predicted_grade:.2f}")
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"**Model Mean Squared Error**: {mse:.2f}")


elif page == "Line Chart":
    st.title("Comparison of Previous Grades and Final Grades")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Name'], df['PreviousGrade'], label='Previous Grade', marker='o')
    ax.plot(df['Name'], df['FinalGrade'], label='Final Grade', marker='o')
    ax.set_xlabel("Students")
    ax.set_ylabel("Grades")
    ax.set_title("Previous Grades vs Final Grades")
    ax.legend()
    st.pyplot(fig)

    def interpret_grades(previous, final, name):
        improvement = final - previous
        
        if improvement > 3:
            interpretation = f"{name} showed a strong improvement of {improvement} points in the final grade, indicating significant academic growth."
        elif 1 <= improvement <= 3:
            interpretation = f"{name} improved by {improvement} points, showing steady progress."
        elif improvement == 0:
            interpretation = f"{name}'s grades remained the same, indicating consistent performance but no growth."
        elif improvement < 0:
            interpretation = f"{name}'s final grade dropped by {abs(improvement)} points, which suggests a decline in performance and may need attention."
        return interpretation

    st.write("The above analysis gives a detailed look into the performance of each student based on their previous and final grades.")

    half = len(df) // 2
    df_left = df.iloc[:half]
    df_right = df.iloc[half:]

    col1, col2 = st.columns(2)

    with col1:
        for index, row in df_left.iterrows():
            name = row['Name']
            previous_grade = row['PreviousGrade']
            final_grade = row['FinalGrade']
            
            interpretation = interpret_grades(previous_grade, final_grade, name)
            with st.expander(f"{name}'s Grades "):
                st.write(f"**Previous Grade**: {previous_grade}")
                st.write(f"**Final Grade**: {final_grade}")
                st.write(interpretation)

    with col2:
        for index, row in df_right.iterrows():
            name = row['Name']
            previous_grade = row['PreviousGrade']
            final_grade = row['FinalGrade']
            
            interpretation = interpret_grades(previous_grade, final_grade, name)
            with st.expander(f"{name}'s Grades"):
                st.write(f"**Previous Grade**: {previous_grade}")
                st.write(f"**Final Grade**: {final_grade}")
                st.write(interpretation)
    


elif page == "Conclusion":
    if st.session_state.show_preview:
        st.subheader("Dataset Preview")
        st.write("Here is a quick preview of the dataset:")
        st.dataframe(df)
    st.title("Conclusion")
    st.write("""
    Based on the analysis of the student performance dataset, we can observe several key insights:
    1. Attendance and Final Grade Correlation:
        Students with higher attendance rates generally have higher final grades (88-92).
        Lower attendance rates tend to be associated with lower final grades (62 and 68).
        This suggests a positive correlation between attendance and final academic performance.
             
    2. Study Hours and Final Grade:
        Students who study more hours per week have higher final grades (90 and 92).
        Conversely, students with lower study hours have lower final grades (62 and 68).
        This indicates that increased study time likely contributes to better academic performance.
             
    3. Previous Grades as Indicators:
        Students with high previous grades continue to perform well in their final grades .
        However, students who had lower previous grades show similarly lower final grades (62 and 68), indicating that past performance can be an indicator of current academic success.
        
    4. Impact of Extracurricular Activities:
        Students involved in more extracurricular activities with 3 also show strong academic performance.
        However, students with no extracurricular activities suggesting that engagement outside of academics may contribute positively to student outcomes.
             
    5. Parental Support:
        High parental support correlates with higher final grades, as seen in students having "High" support. 
        Lower levels of parental support, such as "Low" are associated with lower final grades (62, 68, 72).
    """)

    st.header("Insights")
    st.write(f"""
        The female students in the dataset generally have slightly better final grades (87 to 90 range) than their male counterparts (62 to 92 range). However, this could be influenced by other factors like attendance, study hours, and support.
        These insights can help educators focus on key areas to improve student outcomes.""")
