#!/usr/bin/env python3
"""
Download flight data from OpenSky Network.

This script downloads flight trajectory data for analysis in the track.md notebook.
It uses the traffic library to query OpenSky for flights between specific airports.

Requirements:
    pip install traffic

OpenSky Account:
    Create a free account at https://opensky-network.org/
    Fill in your credentials in the script below.

Usage:
    python download_flight_data.py
"""

from pathlib import Path
import pandas as pd
import os

# Set OpenSky credentials as environment variables before importing traffic
# The traffic library reads these at initialization time
# Set OPENSKY_USERNAME and OPENSKY_PASSWORD in your environment

# Import traffic library
try:
    from traffic.data import opensky, airports
except ImportError:
    print("Error: traffic library not found")
    print("Install with: pip install traffic")
    exit(1)


def download_flight(start_date, end_date, departure, arrival, output_file):
    """
    Download flight data from OpenSky and save to CSV.
    
    Parameters
    ----------
    start_date : str
        Start datetime in format "YYYY-MM-DD HH:MM"
    end_date : str
        End datetime in format "YYYY-MM-DD HH:MM"
    departure : str
        Departure airport ICAO code (e.g., "KBOS" for Boston)
    arrival : str
        Arrival airport ICAO code (e.g., "EIDW" for Dublin)
    output_file : str or Path
        Output CSV filename
    """
    print(f"Querying flights from {departure} to {arrival}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Get airport objects
    departure_airport = airports[departure]
    arrival_airport = airports[arrival]
    
    print(f"Departure: {departure_airport.name}")
    print(f"Arrival: {arrival_airport.name}")
    
    # Query flights
    print("\nQuerying OpenSky... (this may take a minute)")
    flights = opensky.history(
        start=start_date,
        stop=end_date,
        departure_airport=departure_airport,
        arrival_airport=arrival_airport,
    )
    
    if flights is None or len(flights) == 0:
        print("No flights found for this route/date")
        return False
    
    print(f"Found {len(flights)} flight(s)")
    
    # Use the first flight
    flight = flights[0]
    print(f"\nSelected flight: {flight.callsign}")
    print(f"Aircraft: {flight.typecode}")
    
    # Extract dataframe
    df = flight.data
    
    # Save to CSV
    output_path = Path(output_file)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} data points to {output_path}")
    print(f"Columns: {', '.join(df.columns)}")
    
    return True


def main():
    """Download example flight data."""
    
    # Credentials are set via environment variables at the top of the script
    print(f"Using OpenSky credentials from environment")
    print()
    
    # ============================================================
    # CONFIGURE FLIGHT QUERY
    # ============================================================
    
    # Example: Boston (KBOS) to Dublin (EIDW) - using recent date
    config = {
        "start_date": "2025-01-12 00:00",
        "end_date": "2025-01-13 00:00",
        "departure": "KBOS",  # Boston Logan
        "arrival": "EIDW",    # Dublin
        "output_file": "flight_bos_dub.csv"
    }
    
    # Alternative examples (uncomment to use):
    
    # New York (JFK) to London (EGLL)
    # config = {
    #     "start_date": "2024-06-01 00:00",
    #     "end_date": "2024-06-02 00:00",
    #     "departure": "KJFK",  # JFK New York
    #     "arrival": "EGLL",    # Heathrow London
    #     "output_file": "flight_jfk_lhr.csv"
    # }
    
    # Los Angeles (LAX) to Tokyo (RJTT)
    # config = {
    #     "start_date": "2024-06-01 00:00",
    #     "end_date": "2024-06-02 00:00",
    #     "departure": "KLAX",  # LAX Los Angeles
    #     "arrival": "RJTT",    # Haneda Tokyo
    #     "output_file": "flight_lax_hnd.csv"
    # }
    
    # Download the flight
    success = download_flight(**config)
    
    if success:
        print("\nSuccess! Flight data downloaded.")
        print("You can now use this data in your notebook.")
    else:
        print("\nFailed to download flight data.")
        print("Try a different date or route.")


if __name__ == "__main__":
    main()

