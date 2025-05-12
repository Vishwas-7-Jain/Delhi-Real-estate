import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from sklearn.linear_model import LinearRegression
import joblib

# Load data
df = pd.read_csv("delhi_property_data.csv")

# Sidebar filters
st.sidebar.title("Filter Properties")
location = st.sidebar.selectbox("Select Location", sorted(df['Location'].unique()))
bhk = st.sidebar.selectbox("Select BHK", sorted(df['BHK'].unique()))
ptype = st.sidebar.selectbox("Property Type", sorted(df['Property_Type'].unique()))
size = st.sidebar.slider("Size (sqft)", 400, 3500, 1000, 100)

# Prepare model
df_model = pd.get_dummies(df, columns=['Location', 'Property_Type', 'Builder'], drop_first=True)
X = df_model.drop(['Total_Price'], axis=1)
y = df_model['Total_Price']
model = LinearRegression().fit(X, y)

# Create input for prediction
input_dict = {
    'BHK': bhk,
    'Size_sqft': size,
    'Price_per_sqft': int(df[df['Location'] == location]['Price_per_sqft'].mean()),
    'Age_of_Property': int(df[df['Location'] == location]['Age_of_Property'].mean()),
}
for col in X.columns:
    if col.startswith('Location_'):
        input_dict[col] = 1 if col == f'Location_{location}' else 0
    elif col.startswith('Property_Type_'):
        input_dict[col] = 1 if col == f'Property_Type_{ptype}' else 0
    elif col.startswith('Builder_'):
        input_dict[col] = 0  # set to 0 as builder input is skipped for simplicity

input_df = pd.DataFrame([input_dict])
predicted_price = model.predict(input_df)[0]

st.title("🏠 Delhi Property Finder & Price Predictor")
st.markdown(f"### Estimated Price: 💰 ₹{int(predicted_price):,}")

# Display top 5 listings
st.subheader("Top Matching Listings")
top_matches = df[(df['Location'] == location) &
                 (df['BHK'] == bhk) &
                 (df['Property_Type'] == ptype) &
                 (df['Size_sqft'].between(size - 200, size + 200))]

st.dataframe(top_matches[['Location', 'BHK', 'Size_sqft', 'Price_per_sqft', 'Total_Price', 'Property_Type']].head(5))

# Heatmap of average prices
st.subheader("🌍 Price Heatmap (Mock Coordinates)")

# Assign mock lat/lng for demonstration
location_coords = {
    'Saket': [28.5222, 77.2100],
    'Dwarka': [28.5921, 77.0460],
    'Rohini': [28.7500, 77.0500],
    'Lajpat Nagar': [28.5677, 77.2433],
    'Karol Bagh': [28.6519, 77.1909],
    'Connaught Place': [28.6315, 77.2167],
    'Nehru Place': [28.5487, 77.2513],
    'Vasant Kunj': [28.5205, 77.1635]
}

avg_prices = df.groupby('Location')['Price_per_sqft'].mean().reset_index()
avg_prices['lat'] = avg_prices['Location'].map(lambda x: location_coords[x][0])
avg_prices['lon'] = avg_prices['Location'].map(lambda x: location_coords[x][1])

m = folium.Map(location=[28.61, 77.23], zoom_start=11)
heat_data = [[row['lat'], row['lon'], row['Price_per_sqft']] for index, row in avg_prices.iterrows()]
HeatMap(heat_data).add_to(m)
folium_static(m)

st.markdown("---")
st.markdown("Made with ❤️ for Delhi Real Estate")
