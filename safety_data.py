import requests

def fetch_nws_alerts(state_code=None, lat=None, lon=None):
    """
    Fetch active alerts from the National Weather Service API.
    
    Args:
        state_code (str): Two-letter state code (e.g., 'TN').
        lat (float): Latitude for point-based alert search.
        lon (float): Longitude for point-based alert search.
        
    Returns:
        list: A list of alert dictionaries.
    """
    base_url = "https://api.weather.gov/alerts/active"
    headers = {
        "User-Agent": "(FlashFloodChatbot, contact@example.com)",
        "Accept": "application/geo+json"
    }
    
    params = {}
    if lat is not None and lon is not None:
        params['point'] = f"{lat},{lon}"
    elif state_code:
        params['area'] = state_code
    else:
        # Default to national or handle error? For now, let's require at least one.
        return []

    try:
        response = requests.get(base_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        alerts = []
        if 'features' in data:
            for feature in data['features']:
                props = feature.get('properties', {})
                alerts.append({
                    'event': props.get('event', 'Unknown Event'),
                    'headline': props.get('headline', ''),
                    'description': props.get('description', ''),
                    'severity': props.get('severity', 'Unknown'),
                    'urgency': props.get('urgency', 'Unknown'),
                    'areaDesc': props.get('areaDesc', ''),
                    'effective': props.get('effective', ''),
                    'expires': props.get('expires', ''),
                    'instruction': props.get('instruction', '')
                })
        return alerts
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching NWS alerts: {e}")
        return []

def get_red_cross_safety_tips():
    """
    Returns a dictionary of static Red Cross flood safety tips.
    Source: American Red Cross Flood Safety
    """
    return {
        "Prepare": [
            "**Build an Emergency Kit**: Include water (1 gallon per person per day), non-perishable food, flashlight, battery-powered radio, first aid kit, medications, multi-purpose tool, sanitation items, copies of personal documents, cell phone with chargers, family and emergency contact information, and extra cash.",
            "**Make a Plan**: Discuss with your family where you will go if you need to evacuate. Plan how you will communicate if separated.",
            "**Know Your Risk**: Check if you live in a flood plain. Sign up for your community's warning system. The Emergency Alert System (EAS) and National Oceanic and Atmospheric Administration (NOAA) Weather Radio also provide emergency alerts.",
            "**Protect Your Home**: Elevate the furnace, water heater, and electric panel if susceptible to flooding. Install check valves in sewer traps to prevent floodwater from backing up into the drains of your home."
        ],
        "Respond (During a Flood)": [
            "**Listen to Authorities**: If told to evacuate, do so immediately. Never drive around barricades. Local responders use them to safely direct traffic out of flooded areas.",
            "**Turn Around, Don't Drown**: Do not walk, swim, or drive through floodwaters. Just 6 inches of moving water can knock you down, and one foot of moving water can sweep your vehicle away.",
            "**Stay off Bridges**: Stay off bridges over fast-moving water.",
            "**Move to Higher Ground**: If trapped in a building, move to the highest level. Do not climb into a closed attic; you may become trapped by rising floodwater. Go on the roof only if necessary and signal for help."
        ],
        "Recover (After a Flood)": [
            "**Return Home Safely**: Return home only when authorities say it is safe.",
            "**Avoid Hazards**: Be aware of areas where floodwaters have receded and watch out for debris. Floodwaters often erode roads and walkways. Do not attempt to drive through areas that are still flooded.",
            "**Clean Up Safely**: Wear protective clothing, including rubber gloves and boots. Be cautious when cleaning up; mold can be a serious health hazard.",
            "**Electrical Safety**: Do not touch electrical equipment if it is wet or if you are standing in water. If it is safe to do so, turn off the electricity at the main breaker or fuse box to prevent electric shock."
        ]
    }

def get_shelter_info():
    """
    Returns a dictionary of shelter resources and higher ground advice.
    """
    return {
        "Find a Shelter": [
            "**Red Cross Shelter Locator**: Visit [redcross.org/shelter](https://www.redcross.org/get-help/disaster-relief-and-recovery-services/find-open-shelters.html) to find open shelters near you.",
            "**FEMA Mobile App**: Download the FEMA App to find open shelters and disaster recovery centers.",
            "**Text for Shelter**: Text **SHELTER** and your **Zip Code** to **43362** (4FEMA) to find the nearest shelter (standard message rates apply)."
        ],
        "Higher Ground Advice": [
            "**Identify Higher Ground**: Look for hills, multi-story buildings, or designated evacuation points in your community that are above the flood level.",
            "**Move Immediately**: If you are in a low-lying area and flash flooding is possible, move to higher ground immediately. Do not wait for an official warning.",
            "**Avoid Attics**: Do not climb into a closed attic to avoid rising floodwater, as you may become trapped. Go to the roof only if necessary and signal for help."
        ]
    }
