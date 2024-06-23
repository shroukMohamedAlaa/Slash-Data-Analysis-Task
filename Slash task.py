#!/usr/bin/env python
# coding: utf-8

# # Slash internship Task

# ## 1-Exploratory Data Analysis 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[363]:


df=pd.read_csv("Amazon Sale Report .csv")
df


# In[269]:


df.head()


# In[270]:


df.dtypes


# In[271]:


df.info()


# In[158]:


df.describe(include=['object'])


# In[159]:


df.describe()


# ### Visualize the distribution of key features to identify trends and patterns

# In[160]:



# Distribution of Order Status
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Status')
plt.title('Distribution of Order Status')
plt.xlabel('Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[161]:


#
# Aggregating the data to get the sum of 'Amount' by 'Category'
category_sales = df.groupby('Category')['Amount'].sum().reset_index()

# Plotting the bar chart
plt.figure(figsize=(12, 6))
sns.barplot(data=category_sales, x='Category', y='Amount')
plt.title('Sales Amount by Category')
plt.xlabel('Category')
plt.ylabel('Amount (INR)')
plt.xticks(rotation=45)
plt.show()


# In[162]:


# Amount Distribution by Fulfilment
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Fulfilment', y='Amount')
plt.title('Sales Amount by Fulfilment')
plt.xlabel('Fulfilment')
plt.ylabel('Amount (INR)')
plt.xticks(rotation=45)
plt.show()


# In[163]:


# Sales by Region (Ship City)
plt.figure(figsize=(12, 6))
top_cities = df['ship-city'].value_counts().head(10).index
sns.countplot(data=df[df['ship-city'].isin(top_cities)], x='ship-city', order=top_cities)
plt.title('Top 10 Cities by Number of Orders')
plt.xlabel('City')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# ## 2-Data Preprocessing

# ### Removing Duplicates

# In[335]:


df.drop_duplicates()


# In[364]:


# drop unused column
df.drop(columns="Unnamed: 22",axis=1,inplace=True)


# ### Handling Missing Values

# In[365]:


df.isnull().sum()


# In[366]:


df.isnull().sum().sum()


# In[367]:


df.fillna(method='ffill', inplace=True)
df


# In[368]:


df["currency"].fillna(value="INR")


# In[369]:


df["ship-country"].replace(to_replace=np.nan,value="IN")


# In[370]:


df.isnull().sum().sum()


# In[371]:


df.isnull().sum()


# In[372]:


#Drop rows with NaN values that remain 
df.dropna(inplace=True)


# In[373]:


df.isnull().sum().sum()


# In[374]:


# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y', errors='coerce')


# ### Determine outliers using Box Plot

# In[176]:


plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Amount'])
plt.title('Box plot of Amount')
plt.show()


# In[177]:


plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Qty'])
plt.title('Box plot of Qty')
plt.show()


# In[178]:


plt.figure(figsize=(8, 5))
sns.boxplot(x=df['ship-postal-code'])
plt.title('Box plot of ship-postal-code')
plt.show()


# ## 3-Data Visualization

# In[179]:


df['Date'] = pd.to_datetime(df['Date'])

df['MonthYear'] = df['Date'].dt.to_period('M')
monthly_sales = df.groupby('MonthYear')['Amount'].sum().reset_index()
monthly_sales['MonthYear'] = monthly_sales['MonthYear'].astype(str)


# Plotting the monthly sales trends
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['MonthYear'].values , monthly_sales['Amount'].values,  marker='o',linestyle='-')
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Total Sales Amount (INR)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[147]:


# Top-Selling categories
# Aggregate sales data by category
top_categories = df.groupby('Category')['Amount'].sum().sort_values(ascending=False).reset_index()

# Plotting the top-selling categories
plt.figure(figsize=(12, 6))
plt.bar(top_categories['Category'], top_categories['Amount'], color='blue')
plt.title('Top-Selling Categories')
plt.xlabel('Category')
plt.ylabel('Total Sales Amount (INR)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[57]:


# Top-Selling Products
top_products = df.groupby('SKU')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_products, x='Amount', y='SKU',color='lightblue' )
plt.title('Top 10 Selling Products')
plt.xlabel('Total Sales Amount')
plt.ylabel('Product SKU')
plt.tight_layout()
plt.show()
J0230-SKD-M


# In[58]:


top_states = df.groupby('ship-state')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_states, x='Amount', y='ship-state', color='blue')
plt.title('Top 10 States by Sales')
plt.xlabel('Total Sales Amount')
plt.ylabel('State')
plt.tight_layout()
plt.show()


# ### 4: Predictive Modeling

# #### Model Building

# #### Decision Tree

# In[361]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np


# Ensure the 'Date' column is present and convert it to datetime
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y')
else:
    raise KeyError("The 'Date' column is missing from the dataframe")

# Drop rows with NaN values that remain (if any)
df.dropna(inplace=True)

# Feature engineering - extract date components
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Drop the 'Date' column as it's now redundant
df.drop(columns=['Date'], inplace=True)

# Select features and target
target = 'Status'
features = df.drop(columns=[target, 'B2B'])

# Encode categorical features
label_encoders = {}
for column in features.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    features[column] = le.fit_transform(features[column])
    label_encoders[column] = le

# Separate features and target
X = features
y = df[target]

# Encode target labels
status_label_encoder = LabelEncoder()
y = status_label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)

# Predict on the test set
y_pred = decision_tree_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Decision Tree Accuracy: {accuracy:.2f}")
print(f"Decision Tree Precision: {precision:.2f}")
print(f"Decision Tree Recall: {recall:.2f}")
print("Decision Tree Confusion Matrix:")
print(conf_matrix)

# Example new data
new_data = {
    'Order ID': ['405-12345', '678-98765'],  
    'Date': ['24-06-2024', '25-06-2024'],  
    'Amount': [100.50, 250.00],  
    'index': [1, 2],
    'Fulfilment': ['FBM', 'FBA'],
    'Sales Channel': ['Online', 'Store'],
    'ship-service-level': ['Standard', 'Express'],
    'Style': ['Casual', 'Formal'],
    'SKU': ['A123', 'B456'],
    'Category': ['Electronics', 'Clothing'],
    'Size': ['M', 'L'],
    'ASIN': ['ASIN123', 'ASIN456'],
    'Courier Status': ['Delivered', 'Shipped'],
    'Qty': [1, 2],
    'currency': ['USD', 'EUR'],
    'ship-city': ['New York', 'Berlin'],
    'ship-state': ['NY', 'BE'],
    'ship-postal-code': ['10001', '10115'],
    'ship-country': ['USA', 'Germany'],
    'promotion-ids': ['PROMO1', 'PROMO2'],
    'fulfilled-by': ['Amazon', 'Seller'],
}

# Convert to pandas DataFrame
new_data_df = pd.DataFrame(new_data)

# Preprocess new data
new_data_df['Date'] = pd.to_datetime(new_data_df['Date'], format='%d-%m-%Y')  # Adjust format 
new_data_df['Year'] = new_data_df['Date'].dt.year
new_data_df['Month'] = new_data_df['Date'].dt.month
new_data_df['Day'] = new_data_df['Date'].dt.day
new_data_df['Amount'] = pd.to_numeric(new_data_df['Amount'], errors='coerce')

# Drop the 'Date' column
new_data_df.drop(columns=['Date'], inplace=True)

# Create a template DataFrame with the same columns as 'features'
template_df = pd.DataFrame(columns=features.columns)

# Append the new data to the template DataFrame
new_data_encoded = pd.concat([template_df, new_data_df], ignore_index=True)

# Fill any missing columns with zeros or appropriate default values
new_data_encoded = new_data_encoded.fillna(0)

# Encode new data with the same encoders used for training
def encode_new_data(new_data, label_encoders):
    for column in new_data.select_dtypes(include=['object']).columns:
        if column in label_encoders:
            le = label_encoders[column]
            # Handle unseen labels
            new_data[column] = new_data[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    return new_data

new_data_encoded = encode_new_data(new_data_encoded, label_encoders)

# Ensure the new data has the same feature columns as the training data
new_data_encoded = new_data_encoded[X.columns]

# Make predictions
predictions = decision_tree_model.predict(new_data_encoded)

# Convert numerical predictions back to categorical labels
categorical_predictions = status_label_encoder.inverse_transform(predictions)

print(categorical_predictions)


# #### Random Forest

# In[120]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print(f"Random Forest Accuracy: {accuracy_rf:.2f}")
print(f"Random Forest Precision: {precision_rf:.2f}")
print(f"Random Forest Recall: {recall_rf:.2f}")
print("Random Forest Confusion Matrix:")
print(conf_matrix_rf)


# #### 5-Dashboard

# In[189]:


from dash import Dash, dcc, html, Input, Output
import plotly.express as px

df['MonthYear'] = df['Date'].dt.to_period('M').astype(str)
df['SKU'] = df['SKU'].astype(str)

monthly_sales = df.groupby('MonthYear')['Amount'].sum().reset_index()
top_products = df.groupby('SKU')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False).head(10)
top_categories = df.groupby('Category')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False).head(10)
regional_sales = df.groupby('ship-state')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False)

# Initialize the Dash app
app = Dash(__name__)

# Layout of the Dash app
app.layout = html.Div(
    style={'font-family': 'Arial, sans-serif', 'backgroundColor': '#f8f9fa', 'padding': '20px'},
    children=[
        html.H1("Amazon Sales Analysis Dashboard", style={'text-align': 'center', 'margin-bottom': '40px'}),

        # Filters
        html.Div([
            html.Div([
                html.Label("Select Date Range"),
                dcc.DatePickerRange(
                    id='date-picker-range',
                    start_date=df['Date'].min(),
                    end_date=df['Date'].max(),
                    display_format='YYYY-MM-DD'
                ),
            ], style={'padding': '10px'}),

            html.Div([
                html.Label("Select Status"),
                dcc.Dropdown(
                    id='status-dropdown',
                    options=[{'label': status, 'value': status} for status in df['Status'].unique()],
                    multi=True,
                    placeholder="Select Status"
                ),
            ], style={'padding': '10px'}),

            html.Div([
                html.Label("Select Fulfilment"),
                dcc.Dropdown(
                    id='fulfilment-dropdown',
                    options=[{'label': fulfilment, 'value': fulfilment} for fulfilment in df['Fulfilment'].unique()],
                    multi=True,
                    placeholder="Select Fulfilment"
                ),
            ], style={'padding': '10px'}),

            html.Div([
                html.Label("Select Sales Channel"),
                dcc.Dropdown(
                    id='sales-channel-dropdown',
                    options=[{'label': channel, 'value': channel} for channel in df['Sales Channel'].unique()],
                    multi=True,
                    placeholder="Select Sales Channel"
                ),
            ], style={'padding': '10px'})
        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '40px', 'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)'}),

        # Graphs
        html.Div([
            html.Div([
                html.H2("Monthly Sales Trends", style={'text-align': 'center'}),
                dcc.Graph(id="monthly-sales-trends")
            ], style={'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)'})
        ], style={"display": "inline-block", "width": "40%", "vertical-align": "top"}),

        html.Div([
            html.H2("Top Selling Categories", style={'text-align': 'center'}),
            dcc.Graph(id="top-selling-categories")
        ], style={"display": "inline-block", "width": "40%", "margin-left": "4%", 'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)'}),

        html.Div([
            html.Div([
                html.H2("Top Selling Products", style={'text-align': 'center'}),
                dcc.Graph(id="top-selling-products")
            ], style={'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)'})
        ], style={"display": "inline-block", "width": "40%", "vertical-align": "top", 'margin-top': '20px'}),

        html.Div([
            html.H2("Regional Sales Distribution", style={'text-align': 'center'}),
            dcc.Graph(id="regional-sales-distribution")
        ], style={"display": "inline-block", "width": "40%", "margin-left": "4%", 'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 'margin-top': '20px'})
    ]
)

# Callback to update graphs based on filters
@app.callback(
    [
        Output('monthly-sales-trends', 'figure'),
        Output('top-selling-categories', 'figure'),
        Output('top-selling-products', 'figure'),
        Output('regional-sales-distribution', 'figure')
    ],
    [
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
        Input('status-dropdown', 'value'),
        Input('fulfilment-dropdown', 'value'),
        Input('sales-channel-dropdown', 'value')
    ]
)
def update_graphs(start_date, end_date, statuses, fulfilments, sales_channels):
    filtered_df = df[
        (df['Date'] >= pd.to_datetime(start_date)) &
        (df['Date'] <= pd.to_datetime(end_date))
    ]

    if statuses:
        filtered_df = filtered_df[filtered_df['Status'].isin(statuses)]
    if fulfilments:
        filtered_df = filtered_df[filtered_df['Fulfilment'].isin(fulfilments)]
    if sales_channels:
        filtered_df = filtered_df[filtered_df['Sales Channel'].isin(sales_channels)]

    monthly_sales = filtered_df.groupby('MonthYear')['Amount'].sum().reset_index()
    monthly_sales['MonthYear'] = monthly_sales['MonthYear'].astype(str)

    top_products = filtered_df.groupby('SKU')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False).head(10)
    top_categories = filtered_df.groupby('Category')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False).head(10)
    regional_sales = filtered_df.groupby('ship-state')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False)

    fig_monthly_sales = {
        "data": [{
            "x": monthly_sales['MonthYear'],
            "y": monthly_sales['Amount'],
            "type": "lines+markers",
            "name": "Sales"
        }],
        "layout": {
            "margin": {"l": 30, "r": 30, "t": 30, "b": 30},
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#ffffff"
        }
    }

    fig_top_categories = {
        "data": [{
            "x": top_categories["Category"],
            "y": top_categories["Amount"],
            "type": "bar",
            "name": "Top Categories",
            "marker": {"color": "#ff6361"}
        }],
        "layout": {
            "margin": {"l": 30, "r": 30, "t": 30, "b": 30},
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#ffffff"
        }
    }

    fig_top_products = {
        "data": [{
            "x": top_products["SKU"],
            "y": top_products["Amount"],
            "type": "bar",
            "name": "Top Products",
            "marker": {"color": "#ff6361"}
        }],
        "layout": {
            "margin": {"l": 30, "r": 30, "t": 30, "b": 40},
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#ffffff"
        }
    }

    fig_regional_sales = px.bar(regional_sales, x='ship-state', y='Amount')
    fig_regional_sales.update_layout(
        margin={"l": 30, "r": 30, "t": 30, "b": 30},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff"
    )

    return fig_monthly_sales, fig_top_categories, fig_top_products, fig_regional_sales

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)


# In[ ]:




