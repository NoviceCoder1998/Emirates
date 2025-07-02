# Create the Data Visualization page for Streamlit
viz_code = """
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.title("📊 Data Visualization")

@st.cache_data
def load_data():
    return pd.read_csv("data/emirates_synthetic_survey_data.csv")

df = load_data()
st.sidebar.header("Filter Options")

# Sidebar filters
genders = st.sidebar.multiselect("Select Gender(s):", options=df["Gender"].unique(), default=df["Gender"].unique())
classes = st.sidebar.multiselect("Preferred Flight Class:", options=df["Preferred Flight Class"].unique(), default=df["Preferred Flight Class"].unique())

filtered_df = df[df["Gender"].isin(genders) & df["Preferred Flight Class"].isin(classes)]

st.subheader("1. Distribution of Age")
fig1, ax1 = plt.subplots()
sns.histplot(filtered_df["Age"], kde=True, ax=ax1, color="skyblue")
st.pyplot(fig1)

st.subheader("2. Annual Income Distribution")
fig2 = px.box(filtered_df, y="Annual Income", points="all", color_discrete_sequence=["orange"])
st.plotly_chart(fig2)

st.subheader("3. Travel Frequency by Gender")
fig3 = px.histogram(filtered_df, x="Flight Frequency (Annual)", color="Gender", barmode="group")
st.plotly_chart(fig3)

st.subheader("4. Preferred Flight Class by Employment")
fig4 = px.histogram(filtered_df, x="Employment", color="Preferred Flight Class", barmode="group")
st.plotly_chart(fig4)

st.subheader("5. Interest in Personalized App by Class")
fig5 = px.histogram(filtered_df, x="Preferred Flight Class", color="Personalized App Improves Experience", barmode="group")
st.plotly_chart(fig5)

st.subheader("6. Willingness to Pay for Sustainability vs Income")
fig6 = px.box(filtered_df, x="Willing to Pay for Sustainability", y="Annual Income", color="Willing to Pay for Sustainability")
st.plotly_chart(fig6)

st.subheader("7. Heatmap: Correlation (Numerical Columns)")
numeric_cols = filtered_df.select_dtypes(include="number")
fig7, ax7 = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax7)
st.pyplot(fig7)

st.subheader("8. Food Importance by Class")
fig8 = px.box(filtered_df, x="Preferred Flight Class", y="Food Importance (1-5)", color="Preferred Flight Class")
st.plotly_chart(fig8)

st.subheader("9. App Interest vs Likelihood of Use")
fig9 = px.scatter(filtered_df, x="Likelihood to Use Personalized App (1-5)", y="Annual Income",
                  color="Personalized App Improves Experience", size="Age", hover_data=["Preferred Flight Class"])
st.plotly_chart(fig9)

st.subheader("10. Top Frustrations (Word Cloud)")
from wordcloud import WordCloud

text_data = ", ".join(filtered_df["Top Frustrations"].dropna().astype(str))
wc = WordCloud(width=800, height=400, background_color="white").generate(text_data)

fig10, ax10 = plt.subplots()
ax10.imshow(wc, interpolation="bilinear")
ax10.axis("off")
st.pyplot(fig10)

st.success("Explore filters on the sidebar to see different visualizations!")
"""

# Write the page to the correct path
page_path = "/mnt/data/emirates_streamlit_dashboard/pages/1_Data_Visualization.py"
with open(page_path, "w") as f:
    f.write(viz_code)

page_path
