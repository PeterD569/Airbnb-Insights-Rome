

# This script analyzes Airbnb listings in Rome to uncover trends and insights. Key features include:
   # Data preprocessing and cleaning (e.g., handling missing values, calculating derived columns).
    # Exploratory Data Analysis (EDA) using visualizations and descriptive statistics.
    # Calculation of metrics such as average reviews per month and review scores for top hosts.
    # Geospatial and categorical analysis of neighborhoods and room types.
    # Statistical modeling and relationships:
        # Correlations between price, review scores, and number of reviews.
        # Regression Analysis for price
        
#Dataset provided by 'https://insideairbnb.com'

import zipfile
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from NeighbourhoodGroups import assign_neighbourhoods, assign_quarters
from MapCreation import plot_listings_with_filters
from scipy.stats import spearmanr
import RegressionModel as rm


#Load data
file_path = '' 
neighbourhoods_geojson_path = ''
quarters_geojson_path = ''

# Open the listings zip file
with zipfile.ZipFile(file_path, 'r') as z:
    # Open the actual CSV file
    with z.open('listings.csv') as f:
        df = pd.read_csv(f)

print("\nData frame information:")
print(df.info())

#Select columns of interest
columns_of_interest = ['id', 'host_id', 'host_name', 'latitude', 'longitude', 'room_type', 'accommodates', 
                       'bathrooms', 'bedrooms', 'amenities', 'price', 'number_of_reviews', 'first_review', 'last_review', 
                       'review_scores_rating','review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 
                       'review_scores_communication', 'review_scores_location', 'review_scores_value']
df = df.filter(columns_of_interest)

#Dataframe overview
print("\nUpdated data frame information:")
print(df.info())
pd.set_option('display.max_columns', None)
print(df.head())
pd.reset_option('display.max_columns')

#Preprocessing the Data
    #Step 1: Transform columns for visualization/analysis
df['id'] = df['id'].astype(str)
df['host_id'] = df['host_id'].astype(str)
df['price'] = df['price'].replace(r'[\$,]', '', regex=True)
df['price'] = df['price'].astype(float)
df['first_review'] = pd.to_datetime(df['first_review'], errors='coerce')
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

   #Step 2: Assign to listings neighbourhood name and quarter name using latitude and longitude values and digital geoJSON shapefiles  
#function assigns neighbourhood names
neighbourhood_series = assign_neighbourhoods(df, neighbourhoods_geojson_path)
df['neighbourhood'] = neighbourhood_series.fillna('Unknown')

#function assigns quarter names for listings in central rome
quarter_series = assign_quarters(df, quarters_geojson_path)
df['quarter'] = quarter_series.fillna('Unknown/NotCentralArea')

print(df[['id', 'neighbourhood', 'quarter']].head())

    #Step 3: Visual inspection with heatmap of missing values in selected columns
missing_data = df.isnull().astype(int)
fig = go.Figure(
    data=go.Heatmap(
        z=missing_data.values,
        x=missing_data.columns,
        y=missing_data.index,
        colorscale='Viridis',
        showscale=False)
)
fig.update_layout(
    title="Missing Values in Columns of Interest",
    xaxis_title="Columns of Interest",
    yaxis_title="Rows"
)
fig.show()

    # Step 4: Dropping rows if main listing identifier 'id' missing
initial_row_count = len(df)
df_cleaned_basic = df.dropna(subset=['id'])
final_row_count = len(df_cleaned_basic)
n_rows_dropped = initial_row_count - final_row_count
print(f"Number of rows dropped due to missing identifier: {n_rows_dropped}")

# df_basic_cleaned will be used were having the complete number of listings is of importance
print("\nData frame summary after initial cleaning:")
print(df_cleaned_basic.info())

# The following section will deal with Outliers and Missing Values for the remaining Variables

    # Step 5: Drop rows with missing 'price' values
df_cleaned_complete = df_cleaned_basic.dropna(subset=['price'])

    # Step 6: Handle missing values for selected variables by imputing median/mode
# Missing calues in revew_scores columns are related to number of reviews 0. Dropping these rows would create a distortion in the representation of number_of_reviews.
#A warning is created if more then 5% of values are created through imputation
numerical_columns = ['bathrooms', 'bedrooms', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 
                     'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']  #insert numerical variables here

for column in numerical_columns:
    initial_missing_count = df_cleaned_complete[column].isnull().sum()
    df_cleaned_complete.loc[:, column] = df_cleaned_complete[column].fillna(df_cleaned_complete[column].median())
    percentage_imputed = (initial_missing_count / len(df_cleaned_complete)) * 100
    print(f"Percentage of Rows with Imputed '{column}' Values: {percentage_imputed:.2f}%")
    if percentage_imputed > 5:
        print(f"Warning: More than 5% of '{column}' values were imputed.")

categorical_columns = []    #insert categorical variables here

for column in categorical_columns:
    initial_missing_count = df_cleaned_complete[column].isnull().sum()
    mode_value = df_cleaned_complete[column].mode()[0]
    df_cleaned_complete.loc[:, column] = df_cleaned_complete[column].fillna(mode_value)
    percentage_imputed = (initial_missing_count / len(df_cleaned_complete)) * 100
    print(f"Percentage of Rows with Imputed '{column}' Values: {percentage_imputed:.2f}%")
    if percentage_imputed > 5:
        print(f"Warning: More than 5% of '{column}' values were imputed.")

#ALTERNATIVE cleaning option if number of reviews not of interest: drop first rows with value 0 in number_of_reviews, this will remove most missing values in review score columns

    # Step 7: Checking for outlier and removing when necessary
categorical_columns = ['room_type', 'first_review', 'last_review', 'neighbourhood', 'quarter']
for col in categorical_columns:
    print(f"\nValue Counts for {col}:")
    print(df_cleaned_complete[col].value_counts())

# Box plots for numerical values
numeric_columns = ['accommodates', 'bathrooms', 'bedrooms', 'price', 'number_of_reviews',
                   'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                   'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
                   'review_scores_value']

fig = make_subplots(rows=4, cols=3, subplot_titles=numeric_columns)
for i, column in enumerate(numeric_columns):
    row = i // 3 + 1
    col = i % 3 + 1
    fig.add_trace(
        go.Box(y=df_cleaned_complete[column].dropna(), name=column),
        row=row, col=col
    )
fig.update_layout(height=800, width=1200, title_text="Boxplots for Numeric Variables", showlegend=False)
fig.show()

#Conservative outlier removal as high variability is expected
#Outlier removal using cut off value for "bathrooms"
df_cleaned_complete = df_cleaned_complete[df_cleaned_complete['bathrooms'] <= 13]
#Outlier removal using cut off value for "price"
df_cleaned_complete = df_cleaned_complete[df_cleaned_complete['price'] <= 25000]
#Outlier removal using cut off value for "number of reviews"
df_cleaned_complete = df_cleaned_complete[df_cleaned_complete['number_of_reviews'] <= 1000]

# Step 7: Add cost_per_person column, after having clean data for price and accommodates
df_cleaned_complete['cost_per_person'] = df_cleaned_complete['price'] / df_cleaned_complete['accommodates']
print("\n'cost_per_person' column added to data frame:")
print(df_cleaned_complete[['price', 'accommodates', 'cost_per_person']].head())

    # Step 8: Cleaning and transforming "amenities" into seperated counts for amenity frequency analysis
df_cleaned_complete['amenities'] = df_cleaned_complete['amenities'].fillna('')
df_cleaned_complete['amenities_list'] = df_cleaned_complete['amenities'].apply(lambda x: [amenity.strip() for amenity in x.split(',')])
all_amenities = [amenity for sublist in df_cleaned_complete['amenities_list'] for amenity in sublist]

amenity_counts = pd.Series(all_amenities).value_counts()

    # Step 9: Print final confirmation of completed preprocessing
print("\nData cleaning and imputation complete. The final DataFrame is stored as 'df_cleaned_complete'.")
print(df_cleaned_complete.info())




# 1. Total number of individual listings and unique hosts
total_listings = len(df_cleaned_basic)
unique_hosts = df_cleaned_basic['host_id'].nunique()
print("\nTotal Number of Listings:")
print(total_listings)
print("\nTotal Number of Hosts:")
print(unique_hosts)

# 2. Distribution of Listings per Host and top 10 performers (hosts)
host_counts = df_cleaned_basic.groupby(['host_id', 'host_name']).size().sort_values(ascending=False)

# Plotting listing per host distribution
bins = [0, 1, 2, 3, 10, 50, 100, float("inf")]
labels = ["1", "2", "3", "4-10", "11-50", "51-100", "100+"]
host_counts_binned = pd.cut(host_counts.copy(), bins=bins, labels=labels, right=True)

fig_distribution_listings = px.histogram(
    host_counts_binned,
    labels={'value': 'Listings per Host'},
    title="Distribution of Listings per Host"
)
fig_distribution_listings.update_xaxes(categoryorder='array', categoryarray=labels)
fig_distribution_listings.show()

# Plotting the top 10 hosts by number of listings
top_hosts = host_counts.head(10)
fig_top_hosts = go.Figure(go.Bar(
    x=[f"{host_id} ({host_name})" for host_id, host_name in top_hosts.index],
    y=top_hosts.values,
    text=top_hosts.values,
    textposition='auto'
))
fig_top_hosts.update_layout(title="Top 10 Hosts by Number of Listings", xaxis_title="Host ID (Host Name)", yaxis_title="Number of Listings")
fig_top_hosts.show()

# 3. Total percentage of types of Rooms and Room Type Count for Top 10 hosts
#Percentage of Types of Rooms (Pie Chart)
room_type_counts = df_cleaned_basic['room_type'].value_counts(normalize=True).copy() * 100

# Distribution of Room Types for Top 10 Hosts (Grouped Bar Chart with Host ID and Name)
top_hosts_data = df_cleaned_basic['host_id'].value_counts().nlargest(10).index
top_hosts_info = (
    df_cleaned_basic[df_cleaned_basic['host_id'].isin(top_hosts_data)].copy()
    .groupby(['host_id', 'host_name', 'room_type']).size().unstack(fill_value=0)
)

#Prepare labels that combine host_id and host_name
top_hosts_info = top_hosts_info.rename(
    index=lambda idx: f"{idx[0]} ({idx[1]})"
)

room_type_colors = {
    "Entire home/apt": "#1f77b4",  # Blue
    "Private room": "#ff7f0e",     # Orange
    "Shared room": "#2ca02c",      # Green
    "Hotel room": "#d62728"       # Red
}

#Creating subplots
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'pie'}, {'type': 'bar'}]],
    subplot_titles=("Room Type Distribution", "Room Types for Top 10 Hosts"))

#Add pie chart for Room Type Distribution
fig.add_trace(
    go.Pie(labels=room_type_counts.index, 
           values=room_type_counts.values, 
           hole=0.4, 
           marker=dict(colors=[room_type_colors[room]for room in room_type_counts.index])),
    row=1, col=1)

#Add grouped bar chart for Room Types for Top 10 Hosts
for room_type in top_hosts_info.columns:
    fig.add_trace(
        go.Bar(
            x=top_hosts_info.index,
            y=top_hosts_info[room_type],
            name=room_type, marker_color=room_type_colors[room_type]),
        row=1, col=2)

fig.update_layout(
    title="Room Type Analysis",
    barmode='group',  # Grouped bars to avoid overlap
    legend_title="Room Type",)
fig.update_xaxes(title_text="Host ID (Host Name)", row=1, col=2)
fig.update_yaxes(title_text="Number of Listings", row=1, col=2)
fig.show()

# 4. Listings Count per Neighborhood and Count of Listings per Neighbourhood for top 10 Hosts
#Listings Count per Neighborhood
neighborhood_counts = df_cleaned_basic['neighbourhood'].value_counts().reset_index().copy()
neighborhood_counts.columns = ['Neighborhood', 'Listing Count']

#Defining consistent color mapping for neighborhoods
unique_neighborhoods = neighborhood_counts['Neighborhood'].unique()
color_palette = px.colors.qualitative.Plotly  # Using Plotly's default color palette
color_map = {neighborhood: color_palette[i % len(color_palette)] for i, neighborhood in enumerate(unique_neighborhoods)}

pie_colors = [color_map[neighborhood] for neighborhood in neighborhood_counts['Neighborhood']]

# Add pie chart for listings count per neighbourhood
fig_pie = go.Figure(
    go.Pie(
        labels=neighborhood_counts['Neighborhood'],
        values=neighborhood_counts['Listing Count'],
        marker=dict(colors=pie_colors),
        showlegend=True))
fig_pie.update_layout(
    title="Listings per Neighborhood",
    legend=dict(
        title="Neighborhoods",
        orientation="v",
        x=1.05,
        y=0.5))

# Distribution of Listings for Top 10 Hosts by Neighborhood
top_hosts = df_cleaned_basic['host_id'].value_counts().nlargest(10).index
filtered_df = df_cleaned_basic[df_cleaned_basic['host_id'].isin(top_hosts)].copy()
top_hosts_neighborhoods = (
    filtered_df.groupby(['host_id', 'host_name', 'neighbourhood']).size().unstack(fill_value=0))

# Prepare labels that combine host_id and host_name for top hosts
top_hosts_neighborhoods.index = [f"{host_id} ({host_name})" for host_id, host_name in top_hosts_neighborhoods.index]

# Create bar chart for top 10 hosts: listings per neighbourhood
fig_bar = go.Figure()
for neighborhood in top_hosts_neighborhoods.columns:
    fig_bar.add_trace(
        go.Bar(
            x=top_hosts_neighborhoods.index,
            y=top_hosts_neighborhoods[neighborhood],
            name=neighborhood,
            marker_color=color_map[neighborhood]))
fig_bar.update_layout(
    title="Top 10 Hosts' Listings per Neighborhood",
    barmode='group',
    xaxis_title="Host ID (Host Name)",
    yaxis_title="Number of Listings",
    legend=dict(
        title="Neighborhoods",
        orientation="v",
        x=1.05,
        y=0.5))

fig_pie.show()
fig_bar.show()

# 5. Listings Count per Quarter and Count of Listings per Neighbourhood for top 10 Hosts
# Listings Count per Quarter
quarter_counts = df_cleaned_basic['quarter'].value_counts().reset_index().copy()
quarter_counts.columns = ['Quarter', 'Listing Count']

# Define consistent color mapping for quarters
unique_quarters = quarter_counts['Quarter'].unique()
color_palette_quarters = px.colors.qualitative.Plotly  # Using Plotly's default color palette
color_map_quarters = {quarter: color_palette_quarters[i % len(color_palette_quarters)] for i, quarter in enumerate(unique_quarters)}

# Pie Chart: Listings per Quarter
pie_colors_quarters = [color_map_quarters[quarter] for quarter in quarter_counts['Quarter']]
fig_pie = go.Figure(
    go.Pie(
        labels=quarter_counts['Quarter'],
        values=quarter_counts['Listing Count'],
        marker=dict(colors=pie_colors_quarters),
        showlegend=True
    )
)
fig_pie.update_layout(
    title="Listings per Quarter",
    legend_title="Quarters"
)

# Create bar chart: Top 10 Hosts' Listings per Quarter
top_hosts = df_cleaned_basic['host_id'].value_counts().nlargest(10).index
filtered_df = df_cleaned_basic[df_cleaned_basic['host_id'].isin(top_hosts)].copy()
top_hosts_quarters = (
    filtered_df.groupby(['host_id', 'host_name', 'quarter']).size().unstack(fill_value=0)
)

# Prepare labels that combine host_id and host_name for top hosts
top_hosts_quarters.index = [f"{host_id} ({host_name})" for host_id, host_name in top_hosts_quarters.index]

fig_bar = go.Figure()
for quarter in top_hosts_quarters.columns:
    fig_bar.add_trace(
        go.Bar(
            x=top_hosts_quarters.index,
            y=top_hosts_quarters[quarter],
            name=quarter,
            marker_color=color_map_quarters[quarter]
        )
    )
fig_bar.update_layout(
    title="Top 10 Hosts' Listings per Quarter",
    barmode='group',
    xaxis_title="Host ID (Host Name)",
    yaxis_title="Number of Listings",
    legend_title="Quarters"
)

# Show the two separate plots
fig_pie.show()
fig_bar.show()

# 6. This function creates an interactive map for all listings with filters for top 25 hosts and quarters within central area
plot_listings_with_filters(df)

#   7. Number of reviews and average rating scores
# Calculate review months and reviews per month
df_cleaned_complete['review_months'] = (
    (df_cleaned_complete['last_review'] - df_cleaned_complete['first_review']).dt.days / 30.44
)
df_cleaned_complete['reviews_per_month_calculated'] = (
    df_cleaned_complete['number_of_reviews'] / df_cleaned_complete['review_months']
)

# Calculate Overall Average Reviews Per Month
overall_avg_reviews_per_month = df_cleaned_complete['reviews_per_month_calculated'].mean()

# Identify top 10 hosts by the total number of listings
top_10_hosts = (
    df_cleaned_complete.groupby(['host_id', 'host_name'])
    .size()
    .nlargest(10)
    .index
)

# Filter data for top 10 hosts
top_10_hosts_data = df_cleaned_complete[
    df_cleaned_complete['host_id'].isin([host_id for host_id, _ in top_10_hosts])
]

# Calculate the average reviews per listing per month for top 10 hosts
top_10_hosts_avg_reviews = (
    top_10_hosts_data.groupby(['host_id', 'host_name'])['reviews_per_month_calculated']
    .mean()
    .reset_index()
)

# Add total number of listings for sorting
top_10_hosts_avg_reviews['total_listings'] = top_10_hosts_avg_reviews['host_id'].map(
    df_cleaned_complete.groupby('host_id').size()
)

# Sort by total number of listings in descending order
top_10_hosts_avg_reviews = top_10_hosts_avg_reviews.sort_values(by='total_listings', ascending=False)

# Print the information
print(f"\nOverall Average Reviews Per Month: {overall_avg_reviews_per_month:.2f}")
print("\nAverage Reviews Per Listing Per Month for Top 10 Hosts (Ordered by Total Listings):")
print(top_10_hosts_avg_reviews)

# --- Calculate Overall Average Review Scores ---
# List of individual review score columns
review_columns = [
    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
    'review_scores_value'
]

# Calculate overall averages for all review columns
overall_avg_review_scores = {col: df_cleaned_complete[col].mean() for col in review_columns}

# Calculate average scores for all review columns for top 10 hosts
top_10_hosts_avg_review_scores = (
    top_10_hosts_data.groupby(['host_id', 'host_name'])[review_columns]
    .mean()
    .reset_index()
)

# Add total number of listings for sorting
top_10_hosts_avg_review_scores['total_listings'] = top_10_hosts_avg_review_scores['host_id'].map(
    df_cleaned_complete.groupby('host_id').size()
)

# Sort by total number of listings in descending order
top_10_hosts_avg_review_scores = top_10_hosts_avg_review_scores.sort_values(by='total_listings', ascending=False)

# Print the information
print("\nOverall Average Review Scores:")
for score, avg in overall_avg_review_scores.items():
    print(f"{score.replace('_', ' ').title()}: {avg:.2f}")

print("\nAverage Review Scores for Top 10 Hosts (Ordered by Total Listings):")
print(top_10_hosts_avg_review_scores)

# 8. Amenity Popularity Analysis
# Preprocessing the variable
top_amenities = amenity_counts.head(20).reset_index()
top_amenities.columns = ['Amenity', 'Frequency']

fig = px.bar(
    top_amenities,
    x='Amenity',
    y='Frequency',
    title="Top 20 Most Common Amenities",
    labels={'Amenity': 'Amenity', 'Frequency': 'Frequency'},)
fig.update_layout(
    xaxis_tickangle=45,
    height=800,
    width=1200,
    title_x=0.5)
fig.show()

# 9. Average Price and cost per person for each room type
# Calculate the average price and accommodates per room type
avg_price_per_room_type = df_cleaned_complete.groupby('room_type')['price'].mean()

# Calculate the average accommodates per room type
avg_accommodates_per_room_type = df_cleaned_complete.groupby('room_type')['accommodates'].mean()
cost_per_person = avg_price_per_room_type / avg_accommodates_per_room_type

# Combined Plot for Average Price and Cost per Person per Room Type
fig_combined_price_cost = make_subplots(rows=1, cols=1, subplot_titles=["Average Price and Cost per Person per Room Type"])
fig_combined_price_cost.add_trace(
    go.Bar(x=avg_price_per_room_type.index, y=avg_price_per_room_type.values, name="Avg Price"),
    row=1, col=1)
fig_combined_price_cost.add_trace(
    go.Bar(x=cost_per_person.index, y=cost_per_person.values, name="Cost per Person"),
    row=1, col=1)
fig_combined_price_cost.update_layout(
    title="Average Price and Cost per Person per Room Type",
    xaxis_title="Room Type",
    yaxis_title="Cost",
    barmode='group')
fig_combined_price_cost.show()

# 10 Plot Average Accommodates per Room Type
fig_avg_accommodates = go.Figure(go.Bar(
    x=avg_accommodates_per_room_type.index,
    y=avg_accommodates_per_room_type.values,
    name="Avg Accommodates"))
fig_avg_accommodates.update_layout(
    title="Average Accommodates per Room Type",
    xaxis_title="Room Type",
    yaxis_title="Average Accommodates")
fig_avg_accommodates.show()

# 11. Assessing Relationships between number of reviews and Price and Review Scores and Price
# Create and defined rating categories based on review_scores_rating: Low (1-3), Medium (3-4), High (4-5)
df_cleaned_complete['rating_category'] = pd.cut(
    df_cleaned_complete['review_scores_rating'],
    bins=[0, 3, 4, 5],
    labels=['Low (1-3)', 'Medium (3-4)', 'High (4-5)']
)

    # Create and Define popularity categories based on number_of_reviews: Few Reviews (1-10), Some Reviews (11-50), Popular (51-100), Highly Popular (100+)
df_cleaned_complete['popularity'] = pd.cut(
    df_cleaned_complete['number_of_reviews'],
    bins=[0, 10, 50, 100, float("inf")],
    labels=['Few Reviews (1-10)', 'Some Reviews (11-50)', 'Popular (51-100)', 'Highly Popular (100+)']
)

    #Scatterplots to visualize relationship
fig1 = px.scatter(
    df_cleaned_complete,
    x='review_scores_rating',
    y='price',
    color='popularity',
    title="Price vs. Review Scores Rating by Popularity",
    labels={'review_scores_rating': 'Review Scores Rating', 'price': 'Price'}
)
fig1.show()

fig2 = px.scatter(
    df_cleaned_complete,
    x='number_of_reviews',
    y='price',
    color='rating_category',
    title="Price vs. Number of Reviews by Rating Category",
    labels={'number_of_reviews': 'Number of Reviews', 'price': 'Price'}
)
fig2.show()

# Spearman correlation for Price vs Number of Reviews with signifiance testing
spearman_corr_price_reviews, p_value_reviews = spearmanr(df_cleaned_complete['price'], df_cleaned_complete['number_of_reviews'])
print(f"Spearman Correlation (Price vs. Number of Reviews): {spearman_corr_price_reviews:.2f} (p-value: {p_value_reviews:.5f})")

# Spearman correlation for Price vs Review Scores Rating with significance testing
spearman_corr_price_rating, p_value_rating = spearmanr(df_cleaned_complete['price'], df_cleaned_complete['review_scores_rating'])
print(f"Spearman Correlation (Price vs. Review Scores Rating): {spearman_corr_price_rating:.2f} (p-value: {p_value_rating:.5f})")

# 12. Explore correlation between number of reviews and review_scores_rating
# Create scatterplot for visual analysis
fig = px.scatter(
    df_cleaned_complete,
    x='number_of_reviews',
    y='review_scores_rating',
    trendline="ols",  # Adds a linear trendline
    title="Correlation between Number of Reviews and Review Scores Rating",
    labels={'number_of_reviews': 'Number of Reviews', 'review_scores_rating': 'Review Scores Rating'}
)
fig.update_layout(
    xaxis_title="Number of Reviews",
    yaxis_title="Review Scores Rating",
    height=600,
    width=800
)
fig.show()

#Spearman correlation with significance testing for number of reviews vs review scores rating
spearman_corr_reviews_rating, p_val_reviews_rating = spearmanr(df_cleaned_complete['number_of_reviews'], df_cleaned_complete['review_scores_rating'])
print(f"Spearman correlation (number of reviews vs. review_scores_rating): {spearman_corr_reviews_rating:.2f} (p-value: {p_val_reviews_rating:.5f})")

# 13. This function creates a regression model with dependent variable price.
rm.unified_model(df_cleaned_complete)



