#!/usr/bin/env python3
"""
Simple flight data download using traffic library without Trino.

Try to get flight data using methods that don't require Trino database access.
"""

import os
from pathlib import Path
import pandas as pd

# Set credentials before import
# Set OPENSKY_USERNAME and OPENSKY_PASSWORD in your environment

from traffic.data import opensky
from datetime import datetime, timedelta


def download_recent_flight():
    """Try to download a recent flight using simpler API methods."""
    
    print("Attempting to download recent flight data...")
    print("This uses simpler API calls that might not require Trino access.")
    print()
    
    # Try to get data for a specific aircraft over a recent time period
    # Using a shorter, more recent time window
    
    # Get data from last few hours (more likely to work with basic access)
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=6)
    
    print(f"Querying OpenSky for flights from {start_time} to {end_time}")
    print("Looking for transatlantic flights...")
    print()
    
    # Try to get flights in a bounding box (North Atlantic)
    # This is a simpler query that might work
    try:
        # Bounding box for North Atlantic (rough coordinates)
        flights = opensky.history(
            start=start_time,
            stop=end_time,
            bounds=(-80, 30, 0, 60)  # (west, south, east, north)
        )
        
        if flights:
            print(f"Found {len(flights)} flight(s)")
            flight = flights[0]
            print(f"Selected: {flight.callsign}")
            
            df = flight.data
            df.to_csv('recent_flight.csv', index=False)
            print(f"Saved {len(df)} positions to recent_flight.csv")
            return True
        else:
            print("No flights found in bounding box")
            return False
            
    except Exception as e:
        print(f"Bounding box query failed: {e}")
        print()
        print("Unfortunately, your OpenSky account doesn't have access to")
        print("historical data through the Trino database.")
        print()
        print("Options:")
        print("1. Use pre-downloaded sample data")
        print("2. Apply for research access at opensky-network.org")
        print("3. Use synthetic flight data for demonstration")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Simple OpenSky Flight Data Downloader")
    print("=" * 60)
    print()
    
    download_recent_flight()

