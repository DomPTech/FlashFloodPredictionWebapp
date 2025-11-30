import feedparser
import urllib.parse
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests

def test_news_fetching(location_query):
    print(f"Testing for location: {location_query}")
    
    # Test 1: Simple query
    query = f'flash flood {location_query}'
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    print(f"\n--- Test 1: Simple Query (via requests) ---")
    print(f"URL: {rss_url}")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
    try:
        response = requests.get(rss_url, headers=headers, verify=False, timeout=10)
        print(f"Response Status: {response.status_code}")
        
        feed = feedparser.parse(response.content)
        print(f"Entries found: {len(feed.entries)}")
        if feed.entries:
            print(f"First entry: {feed.entries[0].title}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_news_fetching("Nashville, TN")
    test_news_fetching("Nashville Tennessee")
