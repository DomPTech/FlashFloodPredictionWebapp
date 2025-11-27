import streamlit as st
import pandas as pd
import torch
import joblib
from datetime import datetime
from data_fetcher import fetch_streamflow_data, fetch_sites_by_bbox
from predict import predict_flash_flood
from model import FlashFloodClassifier

# Page configuration
st.set_page_config(
    page_title="Flash Flood Prediction",
    page_icon="🌊",
    layout="wide"
)

# Title and description
st.title("Flash Flood Prediction App")
st.markdown("""
This application predicts the probability of a flash flood at a specific USGS site based on historical streamflow data.
Select your location and a date to get started.
""")

# Load resources (cached)
@st.cache_resource
def load_resources():
    # Load Scaler
    try:
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        st.error("scaler.pkl not found. Please ensure the model is trained.")
        return None, None

    # Load Model
    # We need to know the input dimension. Based on training, it's 6 features.
    input_dim = 6
    model = FlashFloodClassifier(input_dim)
    try:
        model.load_state_dict(torch.load("flash_flood_model.pth"))
        model.eval()
    except FileNotFoundError:
        st.error("flash_flood_model.pth not found. Please ensure the model is trained.")
        return None, None
    
    return model, scaler

model, scaler = load_resources()

if model is None or scaler is None:
    st.stop()

# Sidebar for inputs
st.sidebar.header("Configuration")

# State Selection
# List of US states (abbreviated)
states = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]
selected_state = st.sidebar.selectbox("Select State", states, index=states.index('TN'))

# --- Find My Location Feature ---
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Find Nearby Sites")

# Check for query params (location data)
query_params = st.query_params
user_lat = query_params.get("lat")
user_lon = query_params.get("lon")

if st.sidebar.button("Find My Location"):
    # JavaScript to get location and reload page with params
    js = """
    <script>
    async function getLocationAndRedirect() {
    // Check permission status first (optional, helpful for debugging)
    try {
        if (navigator.permissions) {
        const p = await navigator.permissions.query({ name: 'geolocation' });
        console.log('geolocation permission state:', p.state);
        // You can listen for changes:
        // p.onchange = () => console.log('perm changed', p.state);
        }
    } catch (e) {
        console.warn('Permissions API not available', e);
    }

    navigator.geolocation.getCurrentPosition(
        (position) => {
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;
        const url = new URL(window.location.href);
        url.searchParams.set('lat', lat);
        url.searchParams.set('lon', lon);
        window.location.href = url.toString();
        },
        (error) => {
        // Log entire error object for debugging
        console.error('geolocation error object:', error);

        // Map numeric code to human-readable
        const codeMap = {
            1: 'PERMISSION_DENIED',
            2: 'POSITION_UNAVAILABLE',
            3: 'TIMEOUT'
        };
        const codeName = codeMap[error.code] || 'UNKNOWN_ERROR';

        // Some browsers leave error.message empty — show code and raw object
        const msg = error.message || '(no message provided by browser)';
        alert(`Geolocation error (${codeName} / code ${error.code}): ${msg}`);

        // Helpful fallback: show something in the page or console
        // document.body.insertAdjacentHTML('beforeend',
        //   `<p style="color:red">Could not get location: ${codeName}</p>`);
        },
        {
        enableHighAccuracy: false,
        timeout: 10000,      // 10s timeout
        maximumAge: 0
        }
    );
    }

    // Call after load (not strictly required but often helpful)
    if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', getLocationAndRedirect);
    } else {
    getLocationAndRedirect();
    }
    </script>
    """
    st.components.v1.html(js, height=0)

nearby_sites = []
if user_lat and user_lon:
    try:
        lat = float(user_lat)
        lon = float(user_lon)
        st.sidebar.success(f"Location found: {lat:.4f}, {lon:.4f}")
        
        # Define a bounding box (approx +/- 0.5 degrees, roughly 35 miles)
        bbox_margin = 0.5
        with st.spinner("Scanning for nearby sites..."):
            nearby_sites = fetch_sites_by_bbox(
                lon - bbox_margin, 
                lat - bbox_margin, 
                lon + bbox_margin, 
                lat + bbox_margin
            )
            
        if nearby_sites:
            # Calculate distance and sort
            for site in nearby_sites:
                site['dist'] = ((site['lat'] - lat)**2 + (site['lon'] - lon)**2)**0.5
            
            nearby_sites.sort(key=lambda x: x['dist'])
            nearby_sites = nearby_sites[:5] # Top 5
            st.sidebar.info(f"Found {len(nearby_sites)} nearby sites.")
        else:
            st.sidebar.warning("No active sites found nearby.")
            
    except ValueError:
        st.sidebar.error("Invalid coordinates received.")

# --------------------------------

# Fetch Sites for the selected state
@st.cache_data
def get_sites_for_state(state_code):
    try:
        data = fetch_streamflow_data(state_code)
        sites = []
        if 'value' in data and 'timeSeries' in data['value']:
            for series in data['value']['timeSeries']:
                source_info = series.get('sourceInfo', {})
                site_name = source_info.get('siteName', 'Unknown Site')
                site_code = source_info.get('siteCode', [{}])[0].get('value')
                geo_loc = source_info.get('geoLocation', {}).get('geogLocation', {})
                lat = geo_loc.get('latitude')
                lon = geo_loc.get('longitude')
                
                if site_code and lat and lon:
                    sites.append({
                        'name': site_name,
                        'code': site_code,
                        'lat': lat,
                        'lon': lon
                    })
        return sites
    except Exception as e:
        st.error(f"Error fetching sites: {e}")
        return []

with st.spinner(f"Fetching sites for {selected_state}..."):
    sites = get_sites_for_state(selected_state)

if not sites:
    st.warning(f"No active streamflow sites found for {selected_state}.")
else:
    # Site Selection
    # If we found nearby sites, prioritize them in the list or let user select
    if nearby_sites:
        st.info("Showing nearby sites based on your location.")
        site_options = {f"{s['name']} ({s['code']}) - {s['dist']:.2f} deg away": s for s in nearby_sites}
    else:
        site_options = {f"{s['name']} ({s['code']})": s for s in sites}
    
    if not site_options:
         st.warning("No sites available to select.")
         st.stop()

    selected_site_label = st.sidebar.selectbox("Select Site", list(site_options.keys()))
    selected_site_data = site_options[selected_site_label]

    # Date Selection
    prediction_date = st.sidebar.date_input("Prediction Date", datetime.now())

    # Main Content Area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📍 {selected_site_data['name']}")
        
        # Map
        map_data = pd.DataFrame({
            'lat': [selected_site_data['lat']],
            'lon': [selected_site_data['lon']]
        })
        st.map(map_data)

    with col2:
        st.subheader("Prediction")
        
        if st.button("Predict Flood Probability", type="primary"):
            with st.spinner("Calculating probability..."):
                try:
                    # Format date for the predict function
                    date_str = prediction_date.strftime("%Y-%m-%d")
                    
                    prob = predict_flash_flood(
                        model, 
                        scaler, 
                        selected_site_data['code'], 
                        prediction_date=date_str
                    )
                    
                    if prob is not None:
                        st.metric(label="Flood Probability", value=f"{prob:.2%}")
                        
                        if prob < 0.3:
                            st.success("Low Risk")
                        elif prob < 0.7:
                            st.warning("Moderate Risk")
                        else:
                            st.error("High Risk")
                    else:
                        st.error("Could not generate prediction. Insufficient data.")
                        
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

    # Debug info (optional, can be removed)
    with st.expander("Debug Information"):
        st.write("Selected Site Data:", selected_site_data)
        st.write("Prediction Date:", prediction_date)
