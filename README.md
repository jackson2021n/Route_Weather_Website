Live Demo: https://route-weather-website.onrender.com/

# Route Weather Planner

A web app that estimates the weather you will encounter along a driving route.

Enter:
- Start location
- Destination
- Departure date and time

The app finds towns along the route, estimates ETAs, and pulls hourly forecasts for each stop.

## Tech Stack
- Python, Flask
- OSRM (routing)
- OpenStreetMap Nominatim (geocoding)
- National Weather Service API (hourly forecasts)
- Pandas

## Run locally (Windows PowerShell)
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
Open:
http://127.0.0.1:5000

Notes

Routes can take time because the app rate limits requests to public APIs.

