import streamlit as st
import pandas as pd
import torch
import joblib
from datetime import datetime
from data_fetcher import fetch_streamflow_data, fetch_sites_by_bbox
from predict import predict_flash_flood
from model import FlashFloodClassifier
from chatbot import HuggingFaceChatbot
from safety_data import fetch_nws_alerts, get_red_cross_safety_tips
import os

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

# API Token
api_token = st.sidebar.text_input("HuggingFace API Token", type="password", help="Required for AI Assistant. Get one for free at huggingface.co/settings/tokens")
if not api_token:
    api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

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

    # --- Chatbot Tools ---
    def predict_for_chatbot(site_code=None, lat=None, lon=None, site_name=None, query=None):
        """
        Callback function for the chatbot to get flood probability.
        Supports site_code, lat/lon, or site_name/query search.
        """
        try:
            target_site_code = site_code
            
            # If lat/lon provided, find nearest site
            if lat is not None and lon is not None:
                # Reuse the bbox logic from earlier, but maybe just a small box
                bbox_margin = 0.1
                nearby = fetch_sites_by_bbox(lon - bbox_margin, lat - bbox_margin, lon + bbox_margin, lat + bbox_margin)
                if nearby:
                    # Find closest
                    nearby.sort(key=lambda x: ((x['lat'] - lat)**2 + (x['lon'] - lon)**2)**0.5)
                    target_site_code = nearby[0]['code']
                else:
                    return "No nearby USGS sites found for those coordinates."
            
            # If site_name or query provided, try to find a match in the current state's sites
            elif site_name or query:
                search_term = (site_name or query).lower()
                # Use the 'sites' list which is loaded for the selected state
                # 'sites' is available in the local scope because this function is defined inside the script
                # where 'sites' is defined.
                
                if not sites:
                    return "No sites available to search. Please select a state first."
                
                # Simple substring match first
                matches = [s for s in sites if search_term in s['name'].lower()]
                
                if not matches:
                    # Try finding by code if query is numeric
                    if search_term.isdigit():
                         matches = [s for s in sites if search_term in s['code']]
                
                if matches:
                    # Pick the first one or the shortest name match (heuristic)
                    # Let's pick the one with the name that is closest in length to the query, 
                    # assuming exact matches are better.
                    matches.sort(key=lambda x: len(x['name']))
                    best_match = matches[0]
                    target_site_code = best_match['code']
                    # Inform the user which site was picked
                    # We can't easily print to chat here, but we can include it in the return string
                    site_info_str = f"Found site: {best_match['name']} ({best_match['code']}). "
                else:
                    return f"Could not find any sites matching '{search_term}' in {selected_state}."
            
            if not target_site_code:
                # If no site specified, try to use the currently selected site in the UI
                if 'selected_site_data' in locals():
                     target_site_code = selected_site_data['code']
                else:
                    return "Please specify a site code, name, or location."

            # Perform prediction
            # Use today's date for "current" prediction
            date_str = datetime.now().strftime("%Y-%m-%d")
            prob = predict_flash_flood(model, scaler, target_site_code, prediction_date=date_str)
            
            prefix = site_info_str if 'site_info_str' in locals() else ""
            
            if prob is not None:
                risk_level = "Low" if prob < 0.3 else "Moderate" if prob < 0.7 else "High"
                return f"{prefix}The flood probability for site {target_site_code} is {prob:.1%} ({risk_level} Risk)."
            else:
                return f"{prefix}Could not generate prediction (insufficient data)."
                
        except Exception as e:
            return f"Error calculating prediction: {str(e)}"

    chatbot_tools = {
        "get_flood_probability": predict_for_chatbot
    }

    # Initialize Chatbot
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Re-initialize if token changes
    if api_token and st.session_state.get("last_token") != api_token:
        st.session_state.chatbot = HuggingFaceChatbot(api_token=api_token, tools=chatbot_tools)
        st.session_state.last_token = api_token
    elif "chatbot" not in st.session_state and api_token:
         st.session_state.chatbot = HuggingFaceChatbot(api_token=api_token, tools=chatbot_tools)
    
    # Ensure tools are updated if chatbot exists (in case of code reload)
    if "chatbot" in st.session_state and st.session_state.chatbot:
        st.session_state.chatbot.tools = chatbot_tools

    # Main Content Area
    tab1, tab2, tab3 = st.tabs(["Flood Prediction", "AI Assistant", "Safety Info"])

    with tab1:
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

    with tab2:
        st.subheader("AI Flood Assistant")
        
        if not api_token:
            st.warning("Please enter a HuggingFace API Token in the sidebar to use the AI Assistant.")
            st.markdown("[Get a free token here](https://huggingface.co/settings/tokens)")
        else:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Ask about floods, safety, or this app..."):
                # Add user message to history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate response
                with st.chat_message("assistant"):
                    if st.session_state.chatbot:
                        with st.spinner("Thinking..."):
                            response = st.session_state.chatbot.get_response(prompt, st.session_state.messages[:-1])
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("Chatbot not initialized. Please check your token.")

    with tab3:
        # --- NWS Alerts ---
        st.markdown("### Active NWS Alerts")
        
        # Determine location for alerts
        alert_lat, alert_lon = None, None
        if user_lat and user_lon:
            alert_lat, alert_lon = float(user_lat), float(user_lon)
            location_desc = "your location"
        else:
            location_desc = f"{selected_state}"
            
        with st.spinner(f"Fetching active alerts for {location_desc}..."):
            alerts = fetch_nws_alerts(state_code=selected_state, lat=alert_lat, lon=alert_lon)
            
        if alerts:
            st.warning(f"Found {len(alerts)} active alert(s) for {location_desc}.")
            for alert in alerts:
                with st.expander(f"**{alert['event']}** - {alert['severity']} Severity"):
                    st.markdown(f"**Headline:** {alert['headline']}")
                    st.markdown(f"**Area:** {alert['areaDesc']}")
                    st.markdown(f"**Description:**\n{alert['description']}")
                    if alert['instruction']:
                        st.info(f"**Instruction:**\n{alert['instruction']}")
                    st.caption(f"Effective: {alert['effective']} | Expires: {alert['expires']}")
        else:
            st.success(f"No active NWS alerts found for {location_desc} at this time.")
            
        st.markdown("---")
        
        # --- Red Cross Safety Tips ---
        st.markdown("### American Red Cross Flood Safety Tips")
        
        tips = get_red_cross_safety_tips()
        
        for category, items in tips.items():
            with st.expander(f"**{category}**", expanded=True):
                for item in items:
                    st.markdown(f"- {item}")
        
        st.caption("Source: American Red Cross Flood Safety Guidelines")

    # Debug info (optional, can be removed)
    with st.expander("Debug Information"):
        st.write("Selected Site Data:", selected_site_data)
        st.write("Prediction Date:", prediction_date)
