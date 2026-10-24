#!/usr/bin/env python3
"""
Download flight data from OpenSky Network using REST API.

This script uses OpenSky's REST API to download flight trajectory data.
Credentials are loaded from credentials.json file.

Requirements:
    pip install requests pandas

Usage:
    python download_flight_rest.py
"""

import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import time


def load_credentials():
    """Load OpenSky credentials from credentials.json file."""
    cred_file = Path(__file__).parent / "credentials.json"
    
    with open(cred_file) as f:
        creds = json.load(f)
    
    client_id = creds['clientId']
    client_secret = creds['clientSecret']
    
    return client_id, client_secret


def get_flights_by_route(client_id, client_secret, airport1, airport2, begin, end):
    """
    Get flights between two airports using OpenSky REST API.
    
    Parameters
    ----------
    client_id : str
        OpenSky client ID
    client_secret : str
        OpenSky client secret
    airport1 : str
        ICAO code of first airport
    airport2 : str
        ICAO code of second airport
    begin : int
        Unix timestamp for start time
    end : int
        Unix timestamp for end time
    
    Returns
    -------
    list
        List of flight dictionaries
    """
    url = f"https://opensky-network.org/api/flights/arrival"
    
    params = {
        'airport': airport2,
        'begin': begin,
        'end': end
    }
    
    print(f"Querying flights arriving at {airport2}")
    print(f"Time range: {datetime.fromtimestamp(begin)} to {datetime.fromtimestamp(end)}")
    
    response = requests.get(
        url, 
        params=params,
        auth=(client_id, client_secret)
    )
    
    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}")
        print(response.text)
        return []
    
    flights = response.json()
    
    # Filter for flights from airport1
    filtered = [f for f in flights if f.get('estDepartureAirport') == airport1]
    
    print(f"Found {len(filtered)} flights from {airport1} to {airport2}")
    
    return filtered


def get_flight_track(client_id, client_secret, icao24, time_start):
    """
    Get track (trajectory) for a specific flight using OpenSky REST API.
    
    Parameters
    ----------
    client_id : str
        OpenSky client ID
    client_secret : str
        OpenSky client secret
    icao24 : str
        ICAO24 aircraft address
    time_start : int
        Unix timestamp when flight began
    
    Returns
    -------
    pandas.DataFrame
        Flight trajectory data
    """
    url = f"https://opensky-network.org/api/tracks/all"
    
    params = {
        'icao24': icao24,
        'time': time_start
    }
    
    print(f"\nDownloading track for aircraft {icao24}")
    print(f"Time: {datetime.fromtimestamp(time_start)}")
    
    response = requests.get(
        url,
        params=params,
        auth=(client_id, client_secret)
    )
    
    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}")
        print(response.text)
        return None
    
    data = response.json()
    
    if not data or 'path' not in data:
        print("No track data found")
        return None
    
    # Convert path to DataFrame
    path = data['path']
    df = pd.DataFrame(path, columns=[
        'time', 'latitude', 'longitude', 'baro_altitude',
        'true_track', 'on_ground'
    ])
    
    # Add callsign and icao24
    df['callsign'] = data.get('callsign', '')
    df['icao24'] = data.get('icao24', '')
    
    print(f"Downloaded {len(df)} positions")
    
    return df


def download_flight_data(departure_icao, arrival_icao, date_str, output_file):
    """
    Download flight trajectory data for a route.
    
    Parameters
    ----------
    departure_icao : str
        Departure airport ICAO code (e.g., 'KBOS')
    arrival_icao : str
        Arrival airport ICAO code (e.g., 'EIDW')
    date_str : str
        Date in format 'YYYY-MM-DD'
    output_file : str
        Output CSV filename
    """
    # Load credentials
    client_id, client_secret = load_credentials()
    print(f"Using OpenSky client ID: {client_id}")
    
    # Convert date to Unix timestamps (full day)
    date = datetime.strptime(date_str, '%Y-%m-%d')
    begin = int(date.timestamp())
    end = begin + 86400  # +24 hours
    
    # Get flights for this route
    flights = get_flights_by_route(
        client_id, client_secret,
        departure_icao, arrival_icao,
        begin, end
    )
    
    if not flights:
        print(f"\nNo flights found for {departure_icao} → {arrival_icao} on {date_str}")
        return False
    
    # Use the first flight
    flight = flights[0]
    icao24 = flight['icao24']
    callsign = flight.get('callsign', 'Unknown')
    first_seen = flight['firstSeen']
    
    print(f"\nSelected flight:")
    print(f"  Callsign: {callsign}")
    print(f"  Aircraft: {icao24}")
    print(f"  Departure: {flight.get('estDepartureAirport', 'Unknown')}")
    print(f"  Arrival: {flight.get('estArrivalAirport', 'Unknown')}")
    
    # Rate limiting - wait a second between API calls
    time.sleep(1)
    
    # Get track data
    df = get_flight_track(client_id, client_secret, icao24, first_seen)
    
    if df is None or df.empty:
        print("Failed to download track data")
        return False
    
    # Save to CSV
    output_path = Path(output_file)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Success! Saved {len(df)} positions to {output_path}")
    print(f"Columns: {', '.join(df.columns)}")
    
    return True


def main():
    """Download example flight data."""
    
    # Configuration - use a recent date for better data availability
    config = {
        'departure_icao': 'KBOS',  # Boston Logan
        'arrival_icao': 'EIDW',    # Dublin
        'date_str': '2025-01-12',  # Yesterday
        'output_file': 'flight_bos_dub.csv'
    }
    
    # Alternative configurations (uncomment to use):
    
    # New York to London (high frequency route)
    # config = {
    #     'departure_icao': 'KJFK',  # JFK New York
    #     'arrival_icao': 'EGLL',    # Heathrow London
    #     'date_str': '2025-01-12',
    #     'output_file': 'flight_jfk_lhr.csv'
    # }
    
    # Los Angeles to Tokyo (trans-Pacific)
    # config = {
    #     'departure_icao': 'KLAX',  # LAX Los Angeles
    #     'arrival_icao': 'RJTT',    # Haneda Tokyo
    #     'date_str': '2025-01-12',
    #     'output_file': 'flight_lax_hnd.csv'
    # }
    
    print("=" * 60)
    print("OpenSky Flight Data Downloader (REST API)")
    print("=" * 60)
    
    try:
        success = download_flight_data(**config)
        
        if success:
            print("\n" + "=" * 60)
            print("Download complete! You can now use this data in track.md")
            print("=" * 60)
        else:
            print("\nDownload failed. Try a different date or route.")
            
    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

