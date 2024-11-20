import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def assign_geospatial_column(airbnb_data, geojson_path, column_name, fill_value="Unknown"):
    """
    Assign a geospatial attribute (e.g., neighborhood or quarter) to Airbnb listings.

    Parameters:
        airbnb_data (pd.DataFrame): DataFrame with Airbnb listings, containing 'latitude' and 'longitude'.
        geojson_path (str): Path to the GeoJSON file with polygon boundaries.
        column_name (str): The name of the column in the GeoJSON file that contains the desired attribute.
        fill_value (str): Value to assign for missing matches (default: "Unknown").

    Returns:
        pd.Series: A Series containing the assigned geospatial attribute for each listing.
    """
    if 'latitude' not in airbnb_data.columns or 'longitude' not in airbnb_data.columns:
        raise ValueError("Input DataFrame must contain 'latitude' and 'longitude' columns.")

    # Load GeoJSON and ensure CRS is WGS84 (EPSG:4326)
    geodata_gdf = gpd.read_file(geojson_path)
    if geodata_gdf.crs != "EPSG:4326":
        geodata_gdf = geodata_gdf.to_crs("EPSG:4326")
    print(f"Loaded {len(geodata_gdf)} polygons from GeoJSON.")

    # Convert Airbnb listings to GeoDataFrame
    airbnb_gdf = gpd.GeoDataFrame(
        airbnb_data,
        geometry=gpd.points_from_xy(airbnb_data['longitude'], airbnb_data['latitude']),
        crs="EPSG:4326"
    )

    # Spatial join to assign geospatial attribute
    joined_gdf = gpd.sjoin(airbnb_gdf, geodata_gdf, how="left", predicate="within")

    # Return the specified column with missing values filled
    return joined_gdf[column_name].fillna(fill_value)


# Specific functions for neighborhoods and quarters
def assign_neighbourhoods(airbnb_data, geojson_path):
    """
    Assign neighborhoods to Airbnb listings.

    Parameters:
        airbnb_data (pd.DataFrame): DataFrame with Airbnb listings.
        geojson_path (str): Path to the GeoJSON file with neighborhood boundaries.

    Returns:
        pd.Series: A Series containing assigned neighborhoods for each listing.
    """
    return assign_geospatial_column(airbnb_data, geojson_path, column_name="neighbourhood", fill_value="Unknown")


def assign_quarters(airbnb_data, geojson_path):
    """
    Assign quarters to Airbnb listings.

    Parameters:
        airbnb_data (pd.DataFrame): DataFrame with Airbnb listings.
        geojson_path (str): Path to the GeoJSON file with quarter boundaries.

    Returns:
        pd.Series: A Series containing assigned quarters for each listing.
    """
    return assign_geospatial_column(airbnb_data, geojson_path, column_name="name", fill_value="Unknown/NotCentralArea")
