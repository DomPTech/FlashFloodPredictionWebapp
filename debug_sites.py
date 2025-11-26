from data_fetcher import fetch_streamflow_data
import json

def test_state(state_code):
    print(f"Testing state: {state_code}")
    try:
        data = fetch_streamflow_data(state_code)
        
        if 'value' in data and 'timeSeries' in data['value']:
            print(f"  Found 'value' and 'timeSeries'. Count: {len(data['value']['timeSeries'])}")
            for i, series in enumerate(data['value']['timeSeries']):
                source_info = series.get('sourceInfo', {})
                
                if i == 0:
                    print(f"    First Item SourceInfo:")
                    print(json.dumps(source_info, indent=2))
                    break
        else:
            print("  'value' or 'timeSeries' missing in response.")

    except Exception as e:
        print(f"  Error: {e}")

test_state('TN')
