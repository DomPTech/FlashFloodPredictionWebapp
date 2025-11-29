from safety_data import fetch_nws_alerts, get_red_cross_safety_tips
import json

def test_nws_alerts():
    print("Testing NWS Alerts for TN...")
    alerts_tn = fetch_nws_alerts(state_code='TN')
    print(f"Found {len(alerts_tn)} alerts for TN.")
    if len(alerts_tn) > 0:
        print("Sample Alert:", json.dumps(alerts_tn[0], indent=2))
    
    # Test with lat/lon (e.g., Nashville)
    print("\nTesting NWS Alerts for Lat/Lon (36.1627, -86.7816)...")
    alerts_loc = fetch_nws_alerts(lat=36.1627, lon=-86.7816)
    print(f"Found {len(alerts_loc)} alerts for location.")

def test_red_cross_tips():
    print("\nTesting Red Cross Tips...")
    tips = get_red_cross_safety_tips()
    for category, items in tips.items():
        print(f"{category}: {len(items)} items")
        assert len(items) > 0

if __name__ == "__main__":
    test_nws_alerts()
    test_red_cross_tips()
