# FLASH: An AI Chatbot for Real-Time Flash Flood Risk Detection and Information Dissemination

A comprehensive Streamlit-based web application for predicting flash flood probabilities using real-time streamflow data from USGS monitoring sites. The app combines machine learning, geospatial analysis, and AI-powered assistance to provide users with actionable flood risk information and safety guidance. This was created as a high school project for AGU25.

## Overview

This application helps communities and individuals assess flash flood risks by:
- Predicting flood probability at USGS streamflow monitoring sites across the United States
- Providing real-time weather alerts from the National Weather Service
- Offering AI-assisted flood risk analysis and safety guidance
- Displaying historical flood news for specific geographic areas
- Showing location-based safety information including shelter locations

## Technologies & APIs

### Machine Learning & Data Processing
- **PyTorch** - Deep learning framework for the flood prediction model
  - Custom neural network classifier with 3 fully-connected layers (64→32→1 neurons)
  - Binary classification for flood risk assessment
- **scikit-learn** - Feature scaling and data preprocessing
  - StandardScaler for normalizing streamflow metrics
- **pandas** - Time series data manipulation and feature engineering
- **NumPy** - Numerical computations

### Web Framework
- **Streamlit** - Interactive web application framework
  - Multi-tab interface (Dashboard, AI Assistant, Safety Info)
  - Real-time data visualization
  - Session state management for chat history
- **streamlit-folium** - Interactive map integration

### Geospatial & Mapping
- **Folium** - Interactive Leaflet maps
  - Draw tools for geographic area selection
  - Custom markers and popups for USGS sites
- **geopy** - Reverse geocoding via Nominatim (OpenStreetMap)
  - Converts coordinates to human-readable location names

### External APIs & Data Sources

#### USGS Water Services API
- **Endpoint**: `https://waterservices.usgs.gov/nwis/`
- **Purpose**: Real-time and historical streamflow data
- **Parameters**: 
  - State-based site queries
  - Bounding box spatial queries
  - Daily values and instantaneous values
- **Data**: Streamflow measurements (parameter code: 00060) from active monitoring sites

#### HuggingFace Inference API
- **Model**: DeepSeek-R1-0528
- **Purpose**: AI-powered chatbot with function calling capabilities
- **Features**:
  - Natural language flood risk queries
  - Tool calling for real-time predictions
  - Context-aware safety recommendations
- **Integration**: OpenAI-compatible API via HuggingFace Router

#### National Weather Service API
- **Endpoint**: `https://api.weather.gov/alerts`
- **Purpose**: Real-time weather alerts and warnings
- **Features**:
  - State-based and coordinate-based alert queries
  - Flood watches, warnings, and advisories
  - Detailed alert metadata (severity, urgency, instructions)

#### Google News RSS
- **Purpose**: Historical flash flood news aggregation
- **Features**:
  - Geographic and temporal filtering (2015-present)
  - Multi-source news aggregation
  - Deduplication and ranking

#### Nominatim (OpenStreetMap)
- **Purpose**: Reverse geocoding for location identification
- **Features**:
  - Converts lat/lon to city/county/state
  - Used for news search and location context

### AI & NLP
- **OpenAI Python SDK** - Client for HuggingFace API interaction
  - Function/tool calling protocol
  - Streaming and chat completions
- **HuggingFace Hub** - Model hosting and inference

### Additional Libraries
- **feedparser** - RSS feed parsing for Google News
- **joblib** - Model serialization (scaler and weights)
- **requests** - HTTP client for API calls

## Features

### 1. Interactive Dashboard
- Select locations by state or use GPS-based "Find My Location"
- View USGS monitoring sites on an interactive map
- Get flood probability predictions with risk categorization:
  - **Low Risk**: < 30%
  - **Moderate Risk**: 30-70%
  - **High Risk**: > 70%

### 2. Historical News Analysis
- Draw custom search areas on the map to query historical flood events
- Automatic location identification using reverse geocoding
- News from the past 10 years across multiple sources
- Deduplication and relevance ranking

### 3. AI Assistant
- Natural language queries about flood risks
- Function calling for:
  - Real-time flood probability predictions
  - Location-based news retrieval
- Context-aware responses using conversation history
- Automatic filtering of model reasoning artifacts

### 4. Safety Information
- **National Weather Service Alerts**: Real-time warnings for selected areas
- **American Red Cross Guidelines**: Curated safety tips for before, during, and after floods
- **Shelter Information**: Guidance for finding higher ground and safe shelters

## Getting Started

### Prerequisites
- Python 3.8+
- HuggingFace API token (free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd FlashFloodChatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure model files exist:
   - `flash_flood_model.pth` - Trained PyTorch model
   - `scaler.pkl` - Fitted StandardScaler

### Running the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

### Configuration

- **HuggingFace API Token**: Enter in the sidebar or set as environment variable:
  ```bash
  export HUGGINGFACEHUB_API_TOKEN=your_token_here
  ```

## Architecture

### Model Architecture
The flood prediction model is a simple feedforward neural network:
```
Input (6 features) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid) → Probability
```

**Input Features** (engineered from streamflow data):
1. Current streamflow (CFS)
2. 7-day average
3. 7-day standard deviation
4. 30-day average
5. 30-day standard deviation
6. Rate of change

### Application Structure
```
app.py                    # Main Streamlit application
├── model.py              # PyTorch model definition
├── data_fetcher.py       # USGS API client
├── predict.py            # Prediction logic
├── chatbot.py            # HuggingFace AI integration
├── news_collector.py     # Google News RSS scraper
├── safety_data.py        # NWS alerts and safety tips
└── feature_engineering.py # Feature extraction from timeseries
```

## Data Flow

1. **Location Selection** → USGS API query → Active sites
2. **Site Selection** → Historical data fetch → Feature engineering
3. **Prediction** → Scaled features → Model inference → Probability
4. **Safety Checks** → NWS API → Active alerts for region
5. **AI Queries** → HuggingFace API → Function calls → Aggregated response

## Acknowledgments

- **USGS** for comprehensive streamflow data
- **National Weather Service** for real-time alerts
- **American Red Cross** for safety guidelines
- **HuggingFace** for AI inference infrastructure
- **OpenStreetMap/Nominatim** for geocoding services

---

**Disclaimer**: This application provides estimates based on historical data and should not be the sole source for emergency decision-making. Always follow official guidance from local authorities and the National Weather Service.
