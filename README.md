# Airbnb-Insights-Rome
This repository contains a comprehensive analysis of Airbnb listings in Rome using a dataset sourced from Inside Airbnb. The project explores various aspects of Airbnb data, including pricing, host activity, reviews, geographical distribution, and amenities. Advanced statistical modeling, spatial data integration, and interactive visualizations are used to uncover insights about the Airbnb market in Rome, benefiting hosts, travelers, and researchers.
The Dataset includes listing information (e.g., associated host, location, room/apartment metrics, price per night, review metrics, amenities).


Key Features:
- Data preprocessing and cleaning
  - transforming necessary columns
  - assigning neighbourhood and central quarters to listings using geoJSON files
  - missing values and outlier are inspected and removed where necessary
- Exploratory Data Analysis
  - distribution of listings by room type, host activity and neighbourhoods
  - insights into pricing trends, review scores and amenities
- Geospatial Analysis
  - mapping of Airbnb listings across Rome
  - filtered maps for top 25 hosts with most listings with interactive dropdowns
  - neighbourhood and quarter level distribution visualizations
- Statistical Modeling
  - unified regression model ro predict prices using host, review and geographic features
  - residual analysis for model evaluation
  - R-squared, RMSE and MAE metrics
- Interactive visualizations
  - box plots, scatterplots and pie charts using Plotly
  - heatmap to identify missing data patterns

 
Data Source:
The main data file is sourced from Inside Airbnb (https://insideairbnb.com), an open platform providing Airbnb data to empower communities with insights into Airbnb's impact on residential areas.
The neighbourhood and central quarter area are geocoded using the latitude and longitude against neighbourhoods and central quarter areas in Rome as defined by open digital shapefiles, sourced from OpenStreetMap (https://www.openstreetmap.org).


Tech Stack:

Programming Language:
  - Python

Libraries and Framework:
  - pandas, numpy: Data manipulation and analysis.
  - geopandas: Spatial Data integration.
  - plotly, matplotlib: Interactive and static visualizations.
  - scipy, statsmodels: Statistical computation and regression modeling.
  - sklearn: preprocessing and evaluation metrics


Setup and Installation:
  - Python 3.8 or higher
  - Clone the repository
  - Make sure the 3 seperate files with required functions are included ('NeighbourhoodGroups.py' 'MapCreation.py', 'RegressionModel.py')
  - Install required dependencies (pip install -r requirements.txt)
  - Run 'Main.py'

Input Files:
  - listings.csv: Airbnb data file.
  - neighbourhoods.geojson: GeoJSON file for neighbourhood boundaries in Rome
  - Quarteri Roma.geoJSON: GeoJSON file for central quarters in Rome


Insights and Results
Key findings:


Functions and Modules
Preprocessing:
  - assign_neighbourhoods: Adds a neighbourhood column using spatial data
  - assign_quarter: Assigns listings in central areas to quarters
Exploratory Analysis:
  - plot_listings_with_filters: Generates interactive maps filtered by top hosts.
Statistical Modeling:
  - unified_model: Fits a regression model to predict prices using log-transformed price, location, host activity and review metrics


Acknowledgments:
  - Inside Airbnb: For providing the dataset used in this analysis
  - Open-Source Libraries: For enabling advanced analysis and visualization.



