import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def assign_neighbourhoods(airbnb_data, geojson_path):
    """
    Add a column 'neighbourhood' to the Airbnb dataset with the name of the neighborhood each listing belongs to.

    Parameters:
        airbnb_data (pd.DataFrame): DataFrame with Airbnb listings, containing 'latitude' and 'longitude'.
        geojson_path (str): Path to the GeoJSON file with neighborhood borders.

    Returns:
        pd.DataFrame: Updated DataFrame with a 'neighbourhood' column.
    """
    # Step 1: Load the GeoJSON file with neighborhood borders
    neighbourhoods_gdf = gpd.read_file(geojson_path)
    neighbourhoods_gdf = neighbourhoods_gdf.to_crs("EPSG:4326")  # Ensure CRS is WGS84 (lat/lon)
    print(f"Loaded {len(neighbourhoods_gdf)} neighborhoods from GeoJSON.")

    # Step 2: Convert Airbnb listings to a GeoDataFrame
    airbnb_gdf = gpd.GeoDataFrame(
        airbnb_data,
        geometry=[Point(lon, lat) for lon, lat in zip(airbnb_data['longitude'], airbnb_data['latitude'])],
        crs="EPSG:4326"
    )

    # Step 3: Perform spatial join to associate listings with neighborhoods
    listings_with_neighbourhoods = gpd.sjoin(airbnb_gdf, neighbourhoods_gdf, how="left", predicate="within")

    return listings_with_neighbourhoods['neighbourhood']



def assign_quarters(airbnb_data, geojson_path):
    """
    Add a column 'quarter' to the Airbnb dataset with the name of the quarter each listing belongs to.

    Parameters:
        airbnb_data (pd.DataFrame): DataFrame with Airbnb listings, containing 'latitude' and 'longitude'.
        geojson_path (str): Path to the GeoJSON file with quarter borders.

    Returns:
        pd.Series: Series containing the quarter names for each listing.
    """
    # Step 1: Load the GeoJSON file with quarter borders
    quarters_gdf = gpd.read_file(geojson_path)
    quarters_gdf = quarters_gdf.to_crs("EPSG:4326")  # Ensure CRS is WGS84 (lat/lon)
    print(f"Loaded {len(quarters_gdf)} quarters from GeoJSON.")

    # Step 2: Convert Airbnb listings to a GeoDataFrame
    airbnb_gdf = gpd.GeoDataFrame(
        airbnb_data,
        geometry=[Point(lon, lat) for lon, lat in zip(airbnb_data['longitude'], airbnb_data['latitude'])],
        crs="EPSG:4326"
    )


    # Step 3: Perform spatial join to associate listings with quarters
    listings_with_quarters = gpd.sjoin(airbnb_gdf, quarters_gdf, how="left", predicate="within")

    # Return the column that contains quarter names
    return listings_with_quarters['name']



