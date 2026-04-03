Live Demo: https://route-weather-website.onrender.com/

# Route Weather Planner

A web app that estimates the weather you will encounter along a driving route.

Enter:
- Start location
- Destination
- Departure date and time

The app finds towns along the route, estimates ETAs, and pulls hourly forecasts for each stop.

# How It Works
user inputs start, destination, departure time

route API returns path and duration

program samples points and estimates ETAs

weather API returns hourly forecasts for each point

app renders a table

## Tech Stack
- Python, Flask
- OSRM (routing)
- OpenStreetMap Nominatim (geocoding)
- National Weather Service API (hourly forecasts)
- Pandas

Notes

Ensure location is written excatly how prompted (City, STATE ABBREVIATION IN CAPS) *Don't forget the comma

Routes can take time because the app rate limits requests to public APIs.

Free version of Render may take time to "wake up" the site

Stay in the site while it is loading to avoid errors

