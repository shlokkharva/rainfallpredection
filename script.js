
// scripts.js

// Initialize map
var map = L.map('map').setView([20.5937, 78.9629], 5); // Coordinates for India

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

// List of Indian states and their coordinates
const states = [
    { name: 'Delhi', coords: [28.6139, 77.2090] },
    { name: 'Mumbai', coords: [19.0760, 72.8777] },
    { name: 'Kolkata', coords: [22.5726, 88.3639] },
    { name: 'Chennai', coords: [13.0827, 80.2707] },
    { name: 'Bengaluru', coords: [12.9716, 77.5946] },
    { name: 'Hyderabad', coords: [17.3850, 78.4867] },
    { name: 'Ahmedabad', coords: [23.0225, 72.5714] },
    { name: 'Pune', coords: [18.5204, 73.8567] },
    { name: 'Jaipur', coords: [26.9124, 75.7873] },
    { name: 'Chandigarh', coords: [30.7333, 76.7794] },
    { name: 'Lucknow', coords: [26.8467, 80.9462] },
    // Add more states with their coordinates
];

async function fetchPrecipitationData() {
    try {
        const apiKey = 'j6RrAyeZiuVc0Ofe78V6dF86lc0nC74N'; // Replace with your Tomorrow.io API key
        const url = `https://api.tomorrow.io/v4/timelines?location=India&fields=precipitation&units=metric&apikey=${apiKey}`;
        const response = await fetch(url);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching precipitation data:', error);
        return {};
    }
}

async function plotPrecipitation() {
    const data = await fetchPrecipitationData();

    // Check if the data structure has the expected fields
    if (!data || !data.timelines || !data.timelines[0] || !data.timelines[0].intervals) {
        console.error('Unexpected data format:', data);
        return;
    }

    // Example function to find the closest precipitation data point
    function getPrecipitationForCoords(lat, lon) {
        const closestPoint = data.timelines[0].intervals.reduce((closest, interval) => {
            const dist = Math.sqrt(Math.pow(interval.latitude - lat, 2) + Math.pow(interval.longitude - lon, 2));
            return dist < closest.distance ? { value: interval.precipitation, distance: dist } : closest;
        }, { value: 0, distance: Infinity });

        return closestPoint.value;
    }

    states.forEach(state => {
        const precipitation = getPrecipitationForCoords(state.coords[0], state.coords[1]);

        L.circleMarker(state.coords, {
            radius: 10,
            color: precipitation > 0 ? 'red' : 'red',
            fillOpacity: 0.6
        }).addTo(map)
        .bindPopup(`${state.name}: ${precipitation.toFixed(2)} mm`)
        .openPopup();
    });
}

// Plot precipitation on the map
plotPrecipitation();