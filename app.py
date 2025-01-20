import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data.csv")  # Update with your dataset path
data.fillna("Unknown", inplace=True)

# Encode categorical features
label_encoders = {}
for column in ['Reactant_1', 'Reactant_2', 'Reactant_3', 'Reaction_Type', 'Product', 'By_Product_1', 'By_Product_2']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Splitting features and target for different models
X = data[['Reactant_1', 'Reactant_2', 'Reactant_3']]
y_reaction_type = data['Reaction_Type']
y_product = data['Product']
y_by_product_1 = data['By_Product_1']
y_by_product_2 = data['By_Product_2']

# Train Models for Prediction
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X, y_reaction_type)

logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X, y_product)

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X, y_by_product_1)

# Streamlit App UI
st.set_page_config(page_title="Chemical Reaction Predictor", layout="wide")
st.title("🔬 **Chemical Reaction Predictor**")

st.sidebar.header("Select Reactants and Temperature")

# User Inputs for Prediction
reactant_1 = st.sidebar.selectbox("Select Reactant 1", label_encoders['Reactant_1'].classes_)
reactant_2 = st.sidebar.selectbox("Select Reactant 2", label_encoders['Reactant_2'].classes_)
temperature = st.sidebar.slider("Input Temperature (°C):", 0, 600, 150)

# Encode user inputs for prediction
reactant_1_encoded = label_encoders['Reactant_1'].transform([reactant_1])[0]
reactant_2_encoded = label_encoders['Reactant_2'].transform([reactant_2])[0]
user_input = np.array([[reactant_1_encoded, reactant_2_encoded, reactant_1_encoded]])

# Predictions
predicted_reaction_type = knn_model.predict(user_input)
predicted_product = logistic_model.predict(user_input)
predicted_by_product_1 = decision_tree_model.predict(user_input)

# Decode Predictions
reaction_type_decoded = label_encoders['Reaction_Type'].inverse_transform(predicted_reaction_type)[0]
product_decoded = label_encoders['Product'].inverse_transform(predicted_product)[0]
by_product_1_decoded = label_encoders['By_Product_1'].inverse_transform(predicted_by_product_1)[0]

# Display Prediction Results in an Organized Interface
st.header("🧪 **Predicted Reaction Details**")
st.info(f"**Reaction:** {reactant_1} + {reactant_2} → {reaction_type_decoded}")
st.write(f"- **Main Product Formed:** {product_decoded}")
st.write(f"- **Primary By-Product:** {by_product_1_decoded}")
st.write(f"- **Suggested Temperature Range:** 150°C - 400°C (example)")

# Visualization: Reaction Temperature Input
st.subheader("🌡️ Reaction Temperature Analysis")
fig, ax = plt.subplots()
ax.bar(["Input Temperature"], [temperature], color='skyblue')
ax.set_ylabel("Temperature (°C)")
ax.set_title("Reaction Input Temperature Visualization")
st.pyplot(fig)

# Sidebar Notes for Guidance
st.sidebar.subheader("📋 **Notes for Users**")
st.sidebar.write("""
- Select reactants and input temperature to generate a prediction.
- Outputs include reaction type, primary product, and by-product.
- Temperature slider helps simulate varying conditions for reactions.
""")
