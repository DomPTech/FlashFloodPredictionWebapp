import feedparser
import urllib.parse
import requests
import urllib3

# Disable SSL warnings for this module
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_location_name(lat, lon):
    """
    Reverse geocode coordinates to get a meaningful location name (City, County, State).
    Uses direct API call to Nominatim to bypass SSL certificate issues on some environments.
    """
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        'lat': lat,
        'lon': lon,
        'format': 'json',
        'addressdetails': 1
    }
    headers = {
        'User-Agent': 'flash_flood_chatbot_v1'
    }
    
    try:
        # verify=False is used here to resolve persistent SSL certificate errors on the user's machine
        response = requests.get(url, params=params, headers=headers, verify=False, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            address = data.get('address', {})
            
            city = address.get('city') or address.get('town') or address.get('village')
            county = address.get('county')
            state = address.get('state')
            
            parts = []
            if city: parts.append(city)
            if county: parts.append(county)
            if state: parts.append(state)
            
            return ", ".join(parts) if parts else "Unknown Location"
        else:
            print(f"Geocoding failed with status {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None

def fetch_flood_news(location_query):
    """
    Fetch flash flood news for a specific location using Google News RSS.
    Tries multiple query variations to maximize results.
    """
    if not location_query or location_query == "Unknown Location":
        return []

    # Generate query variations
    queries = []
    
    # 1. Broad query: flash flood {location} after:2015 (No quotes to allow flexible matching)
    queries.append(f'flash flood {location_query} after:2015-01-01')
    
    # 2. If location has commas (e.g. "Nashville, Davidson County, Tennessee"), try "City State"
    if "," in location_query:
        parts = [p.strip() for p in location_query.split(",")]
        if len(parts) >= 2:
            # Try "City State"
            city_state = f"{parts[0]} {parts[-1]}"
            queries.append(f'flash flood {city_state} after:2015-01-01')
            
    # 3. Just the location without date filter if others fail (recent news)
    queries.append(f'flash flood {location_query}')

    all_news = []
    seen_links = set()
    
    for q in queries:
        try:
            encoded_query = urllib.parse.quote(q)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            
            print(f"Fetching news for query: {q}")
            
            # Use requests to fetch content first to handle SSL and Headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
            }
            response = requests.get(rss_url, headers=headers, verify=False, timeout=10)
            
            if response.status_code != 200:
                print(f"Failed to fetch news feed. Status: {response.status_code}")
                continue
                
            feed = feedparser.parse(response.content)
            
            for entry in feed.entries:
                if entry.link not in seen_links:
                    all_news.append({
                        'title': entry.title,
                        'link': entry.link,
                        'published': entry.published,
                        'summary': entry.summary,
                        'source': entry.source.title if hasattr(entry, 'source') else 'Google News'
                    })
                    seen_links.add(entry.link)
            
            # If we found a good number of results, we can stop trying broader/other queries
            if len(all_news) >= 5:
                break
                
        except Exception as e:
            print(f"Error fetching news for query '{q}': {e}")
            continue
        
    return all_news

if __name__ == "__main__":
    # Test
    loc = "Nashville, TN"
    print(f"Fetching news for {loc}...")
    news = fetch_flood_news(loc)
    for item in news[:3]:
        print(f"- {item['title']} ({item['published']})")
