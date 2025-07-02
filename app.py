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

# Code for the Classification tab (Streamlit page)
classification_code = """
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import io

st.title("🤖 Classification Models")

@st.cache_data
def load_data():
    return pd.read_csv("data/emirates_synthetic_survey_data.csv")

df = load_data()

# Preprocess
df = df.dropna()
df_model = df.copy()

# Encode categorical columns
label_encoders = {}
for col in df_model.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

# Define features and target
features = df_model.drop(columns=["Personalized App Improves Experience"])
target = df_model["Personalized App Improves Experience"]

# Binarize target for classification
target = target.apply(lambda x: 1 if x == 1 else 0)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = []

st.subheader("🔍 Model Performance Summary")

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results).round(3)
st.dataframe(results_df)

# Confusion matrix
st.subheader("📊 Confusion Matrix")
selected_model = st.selectbox("Select model to display confusion matrix", list(models.keys()))
model_cm = models[selected_model]
model_cm.fit(X_train_scaled, y_train)
y_pred_cm = model_cm.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_cm)

fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Not Interested", "Interested"], yticklabels=["Not Interested", "Interested"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig_cm)

# ROC curve
st.subheader("📈 ROC Curve")
fig_roc = go.Figure()
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash'), name='Random'))
fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
st.plotly_chart(fig_roc)

# Upload new data and predict
st.subheader("📤 Upload Data for Prediction")
uploaded_file = st.file_uploader("Upload CSV (same structure as original without target):", type="csv")
if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    for col in new_data.select_dtypes(include='object').columns:
        if col in label_encoders:
            new_data[col] = label_encoders[col].transform(new_data[col].astype(str))
    new_scaled = scaler.transform(new_data)
    best_model = RandomForestClassifier()
    best_model.fit(X_train_scaled, y_train)
    predictions = best_model.predict(new_scaled)
    output_df = new_data.copy()
    output_df["Predicted Label"] = predictions

    st.write("Prediction Results:")
    st.dataframe(output_df.head())

    buffer = io.StringIO()
    output_df.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button("📥 Download Predictions", buffer, file_name="predictions.csv", mime="text/csv")
"""

# Save classification code to pages directory
classification_path = "/mnt/data/emirates_streamlit_dashboard/pages/2_Classification.py"
with open(classification_path, "w") as f:
    f.write(classification_code)

classification_path

