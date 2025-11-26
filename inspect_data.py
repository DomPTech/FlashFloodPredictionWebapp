from data_fetcher import fetch_streamflow_data
import json

try:
    data = fetch_streamflow_data('TN')
    # Print keys of the top level object
    print("Top level keys:", data.keys())
    
    # It seems to be a USGS NWIS response. Usually it has 'value' -> 'timeSeries'
    if 'value' in data and 'timeSeries' in data['value']:
        time_series = data['value']['timeSeries']
        print(f"Number of sites found: {len(time_series)}")
        if len(time_series) > 0:
            first_site = time_series[0]
            # Print structure of the first site to find name and site code
            print("First site structure keys:", first_site.keys())
            print("SourceInfo:", json.dumps(first_site.get('sourceInfo'), indent=2))
            print("Variable:", json.dumps(first_site.get('variable'), indent=2))
    else:
        print("Unexpected structure:", json.dumps(data, indent=2)[:500])

except Exception as e:
    print(f"Error: {e}")
