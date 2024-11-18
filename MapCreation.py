
import pandas as pd
import plotly.express as px

def plot_listings_with_filters(df):
    """
    Plot all Airbnb listings with filtering options for the top 25 hosts.

    Parameters:
        df (pd.DataFrame): DataFrame containing Airbnb listings with latitude, longitude, host_id, etc.

    Returns:
        None
    """
    # Identify top 25 hosts by number of listings
    host_counts = df['host_id'].value_counts().nlargest(25)
    top_hosts = host_counts.index

    # Ensure 'host_name' and 'room_type' columns are strings for hover information
    df['host_name'] = df['host_name'].astype(str)
    df['room_type'] = df['room_type'].astype(str)

    # Create a scatter mapbox plot
    fig = px.scatter_mapbox(
        df,
        lat='latitude',
        lon='longitude',
        hover_name='host_name',
        hover_data={
            'host_id': True, 'room_type': True, 'price': True
        },
        color='host_id',  # Color by host_id to differentiate top hosts
        zoom=8,
        height=600,
        opacity=0.75,
        color_discrete_sequence=px.colors.qualitative.Set2  # Use a distinct color palette
    )

    # Customize marker size to make dots more visible
    fig.update_traces(marker=dict(size=6))

    # Set map style and center on Lazio, Italy
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": 41.9, "lon": 12.6},
        title="Airbnb Listings with Filters for Top 25 Hosts",
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    # Dropdown filter options
    dropdown_buttons = [
        {
            "label": "All Listings",
            "method": "update",
            "args": [{"lat": [df['latitude']], "lon": [df['longitude']], "marker.color": [df['host_id']]}]
        }
    ]

    # Add filtering by top 25 hosts
    for host in top_hosts:
        host_data = df[df['host_id'] == host]
        host_name = host_data['host_name'].iloc[0]
        listing_count = len(host_data)
        dropdown_buttons.append({
            "label": f"Host: {host_name} ({host}) - {listing_count} Listings",
            "method": "update",
            "args": [{"lat": [host_data['latitude']], "lon": [host_data['longitude']],
                      "marker.color": [host_data['host_id']]}]
        })

    # Add dropdown menu to layout
    fig.update_layout(
        updatemenus=[{
            "buttons": dropdown_buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.17,
            "xanchor": "left",
            "y": 1.1,
            "yanchor": "top",
            "pad": {"r": 10, "t": 10},
        }]
    )

    # Show the figure
    fig.show()

    # Prompt user to save as HTML
    save_html = input("Do you want to save this map as an HTML file? (Yes or No): ").strip().lower()
    if save_html == 'yes':
        fig.write_html("top_hosts_map.html")
        print("Map saved as 'top_hosts_map.html'.")


