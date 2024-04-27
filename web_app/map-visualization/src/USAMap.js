import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import * as d3 from 'd3';

const stateFipsMapping = {
  '01': 'Alabama',
  '02': 'Alaska',
  '04': 'Arizona',
  '05': 'Arkansas',
  '06': 'California',
  '08': 'Colorado',
  '09': 'Connecticut',
  '10': 'Delaware',
  '11': 'District of Columbia',
  '12': 'Florida',
  '13': 'Georgia',
  '15': 'Hawaii',
  '16': 'Idaho',
  '17': 'Illinois',
  '18': 'Indiana',
  '19': 'Iowa',
  '20': 'Kansas',
  '21': 'Kentucky',
  '22': 'Louisiana',
  '23': 'Maine',
  '24': 'Maryland',
  '25': 'Massachusetts',
  '26': 'Michigan',
  '27': 'Minnesota',
  '28': 'Mississippi',
  '29': 'Missouri',
  '30': 'Montana',
  '31': 'Nebraska',
  '32': 'Nevada',
  '33': 'New Hampshire',
  '34': 'New Jersey',
  '35': 'New Mexico',
  '36': 'New York',
  '37': 'North Carolina',
  '38': 'North Dakota',
  '39': 'Ohio',
  '40': 'Oklahoma',
  '41': 'Oregon',
  '42': 'Pennsylvania',
  '44': 'Rhode Island',
  '45': 'South Carolina',
  '46': 'South Dakota',
  '47': 'Tennessee',
  '48': 'Texas',
  '49': 'Utah',
  '50': 'Vermont',
  '51': 'Virginia',
  '53': 'Washington',
  '54': 'West Virginia',
  '55': 'Wisconsin',
  '56': 'Wyoming',
  // Include mappings for Maine and Nebraska congressional districts
  'ME-01': 'Maine CD-1',
  'ME-02': 'Maine CD-2',
  'NE-01': 'Nebraska CD-1',
  'NE-02': 'Nebraska CD-2',
  'NE-03': 'Nebraska CD-3',
};

const USAMap = ({ winProbabilitiesCsv }) => {
  const [mapData, setMapData] = useState(null);
  const [winProbabilities, setWinProbabilities] = useState({});

  useEffect(() => {
    // Fetch the GeoJSON data and set it to state
    const fetchMapData = async () => {
      try {
        const response = await fetch('cb_2018_us_cd116_500k.geojson'); // Corrected path to the GeoJSON file
        const json = await response.json();
        setMapData(json);
      } catch (error) {
        console.error('Error fetching map data:', error);
      }
    };

    fetchMapData();
  }, []);

  useEffect(() => {
    // Fetch the latest win probabilities and set it to state
    const fetchWinProbabilities = async () => {
      try {
        const response = await fetch('/biden_win_probabilities.csv'); // Corrected path to the CSV file
        const text = await response.text();
        const probabilities = parseCsv(text);
        setWinProbabilities(probabilities);
        console.log('Win Probabilities:', probabilities); // Log to check fetched win probabilities
      } catch (error) {
        console.error('Error fetching win probabilities:', error);
      }
    };

    fetchWinProbabilities();
  }, [winProbabilitiesCsv]);

  const parseCsv = (csvText) => {
    // Parse CSV text and return an object with win probabilities
    const data = d3.csvParse(csvText);
    const probabilities = {};
    // Assuming the last row contains the most recent probabilities
    const lastRow = data[data.length - 1];
    Object.keys(lastRow).forEach(state => {
      if (state !== 'Date') { // Exclude the 'Date' column
        probabilities[state] = +lastRow[state];
      }
    });
    return probabilities;
  };

  const colorMapBasedOnProbability = (feature) => {
    // Determine the color of the state or district based on Biden's win probability
    const stateFips = feature.properties.STATEFP;
    const stateName = stateFipsMapping[stateFips];
    // Check if the stateFips is one of the undefined values or if stateName is not found in winProbabilities
    if (['60', '66', '69', '72', '78'].includes(stateFips) || !winProbabilities.hasOwnProperty(stateName)) {
      // Return a default style for undefined 'STATEFP' values or missing data
      return {
        fillColor: 'grey',
        weight: 2,
        opacity: 1,
        color: 'white',
        fillOpacity: 0.7
      };
    }
    // If stateName is found, proceed with color calculation
    const probability = winProbabilities[stateName];
    const blueIntensity = probability * 255;
    const redIntensity = (1 - probability) * 255;
    const color = `rgb(${redIntensity}, 0, ${blueIntensity})`;
    return {
      fillColor: color,
      weight: 2,
      opacity: 1,
      color: 'white',
      fillOpacity: 0.7
    };
  };

  const onEachFeature = (feature, layer) => {
    // Define the function for interactive layer behavior here
    // This can include binding popups, styling, etc.
    const stateFips = feature.properties.STATEFP;
    const stateName = stateFipsMapping[stateFips];
    if (!stateName || !winProbabilities.hasOwnProperty(stateName)) {
      layer.bindPopup(`Data not available for: ${stateFips} - ${stateName}`);
    } else {
      const probability = winProbabilities[stateName];
      layer.bindPopup(`${stateName}: ${probability * 100}% chance`);
    }
  };

  return (
    <MapContainer center={[37.8, -96]} zoom={4} style={{ height: '100vh', width: '100%' }}>
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      {mapData && <GeoJSON data={mapData} onEachFeature={onEachFeature} style={colorMapBasedOnProbability} />}
    </MapContainer>
  );
};

export default USAMap;
