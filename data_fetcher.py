import requests
import pandas as pd

def fetch_streamflow_data(state_code='TN'):
    """
    Fetch streamflow data from the USGS NWIS service.

    Parameters:
    - state_code: State abbreviation (default is 'CA' for California)

    Returns:
    - List containing streamflow data.
    """
    # Base URL for the NWIS API
    base_url = "https://waterservices.usgs.gov/nwis/iv/"

    # Parameters for the API request
    params = {
        'format': 'json',  # Response format
        'stateCd': state_code,  # State Code
        'siteStatus': 'active',  # Only fetch active sites
        'parameterCd': '00060',  # Parameter code for streamflow
    }

    # Send the request to the USGS NWIS service
    response = requests.get(base_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")

def fetch_historical_streamflow_data(site_number, start_date, end_date):
    """
    Fetch historical streamflow data from the USGS NWIS service.

    Parameters:
    - site_number: USGS site number for the streamgage.
    - start_date: Start date in the format 'YYYY-MM-DD'.
    - end_date: End date in the format 'YYYY-MM-DD'.

    Returns:
    - DataFrame containing historical streamflow data.
    """
    base_url = "https://waterservices.usgs.gov/nwis/dv/"

    # Parameters for the API request
    params = {
        'format': 'json',
        'sites': site_number,
        'startDT': start_date,
        'endDT': end_date,
        'parameterCd': '00060',
        'siteStatus': 'active'
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        # Parse the JSON data into a DataFrame
        time_series = data['value']['timeSeries']
        records = []

        # How the json is formatted inside each series
        """
        "values": [
            {
                "value": [
                    {
                        "value": "335",
                        "qualifiers": [
                            "A"
                        ],
                        "dateTime": "2020-01-01T00:00:00.000"
                    },
        """
        for series in time_series:
            values = series['values'][0]["value"]
            for occurrence in values:
                record = {
                    'date_time': occurrence["dateTime"],
                    'streamflow_cfs': occurrence["value"]
                }
                records.append(record)

        df = pd.DataFrame(records)
        if len(records) != 0:
            df['date_time'] = pd.to_datetime(df['date_time'])
        else:
            # Ensure columns exist even if empty
            df = pd.DataFrame(columns=['date_time', 'streamflow_cfs'])
        return df
    else:
        raise Exception(f"Error fetching historical data: {response.status_code} - {response.text}")