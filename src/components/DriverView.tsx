import React, { useState, useEffect, useRef } from 'react';
import { useTranslation } from '@/contexts/TranslationContext';
import { ArrowLeft } from 'lucide-react';

const regions = [
  'North West Delhi',
  'North East Delhi',
  'South West Delhi',
  'South East Delhi'
];

const intersections = [
  'Sai Baba Chowk',
  'Madhuban Chowk',
  'Rithala Metro Junction',
  'Deepali Chowk',
  'Mukarba Chowk',
  'Punjabi Bagh Crossing'
];

// Approximate coordinates for Delhi intersections (for map display)
const intersectionCoordinates: Record<string, { lat: number; lng: number }> = {
  'Sai Baba Chowk': { lat: 28.7041, lng: 77.1025 },
  'Madhuban Chowk': { lat: 28.7141, lng: 77.1125 },
  'Rithala Metro Junction': { lat: 28.7241, lng: 77.1225 },
  'Deepali Chowk': { lat: 28.6941, lng: 77.0925 },
  'Mukarba Chowk': { lat: 28.7041, lng: 77.1025 },
  'Punjabi Bagh Crossing': { lat: 28.6841, lng: 77.0825 },
};

interface DriverViewProps {
  onBack?: () => void;
}

interface IntersectionData {
  Region_Name: string;
  Intersection_ID: string;
  Max_Vehicle_Count: number;
  Max_Green_Time: number;
  [key: string]: any;
}

export const DriverView: React.FC<DriverViewProps> = ({ onBack }) => {
  const { t } = useTranslation();
  const [region, setRegion] = useState(regions[0]);
  const [intersectionName, setIntersectionName] = useState(intersections[0]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [latest, setLatest] = useState<any | null>(null);
  const [allIntersections, setAllIntersections] = useState<IntersectionData[]>([]);
  const [mapLoading, setMapLoading] = useState(true);
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<google.maps.Map | null>(null);
  const markersRef = useRef<google.maps.Marker[]>([]);

  // Fetch all intersections data for map
  useEffect(() => {
    const fetchAllIntersections = async () => {
      try {
        const resp = await fetch('http://localhost:5000/api/db/all-intersections');
        if (resp.ok) {
          const data = await resp.json();
          if (data.ok && data.intersections) {
            setAllIntersections(data.intersections);
          }
        }
      } catch (err) {
        console.error('Failed to fetch all intersections:', err);
      }
    };
    fetchAllIntersections();
    // Refresh every 30 seconds
    const interval = setInterval(fetchAllIntersections, 30000);
    return () => clearInterval(interval);
  }, []);

  // Initialize Google Map
  useEffect(() => {
    if (!mapRef.current) {
      return;
    }

    let checkInterval: NodeJS.Timeout | null = null;
    let timeoutId: NodeJS.Timeout | null = null;

    // Wait for Google Maps to load
    const initMap = () => {
      if (typeof google === 'undefined' || !google.maps || !mapRef.current) {
        // Retry after a short delay if Google Maps isn't loaded yet
        setTimeout(initMap, 100);
        return;
      }

      try {
        // Initialize map centered on Delhi
        const map = new google.maps.Map(mapRef.current, {
          center: { lat: 28.7041, lng: 77.1025 },
          zoom: 12,
          mapTypeId: google.maps.MapTypeId.ROADMAP,
        });

        mapInstanceRef.current = map;
        setMapLoading(false);
      } catch (err) {
        console.error('Error initializing map:', err);
        setMapLoading(false);
      }
    };

    // Start initialization
    if (typeof google !== 'undefined' && google.maps) {
      initMap();
    } else {
      // Wait for script to load
      checkInterval = setInterval(() => {
        if (typeof google !== 'undefined' && google.maps) {
          if (checkInterval) clearInterval(checkInterval);
          initMap();
        }
      }, 100);

      // Timeout after 10 seconds
      timeoutId = setTimeout(() => {
        if (checkInterval) clearInterval(checkInterval);
        setMapLoading(false);
        console.error('Google Maps failed to load');
      }, 10000);
    }

    return () => {
      if (checkInterval) clearInterval(checkInterval);
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, []);

  // Update markers when intersections data changes
  useEffect(() => {
    if (!mapInstanceRef.current || allIntersections.length === 0) return;

    // Clear existing markers
    markersRef.current.forEach(marker => marker.setMap(null));
    markersRef.current = [];

    // Create markers for each intersection
    allIntersections.forEach((intersection) => {
      const coords = intersectionCoordinates[intersection.Intersection_ID];
      if (!coords) return;

      const mg = parseInt(String(intersection.Max_Green_Time || '0'), 10) || 0;
      const vc = parseInt(String(intersection.Max_Vehicle_Count || '0'), 10) || 0;

      // Density-based coloring derived from current vehicle count
      // Thresholds:
      // <20 Low (green), 20-35 Moderate (orange), 35-49 High (red), >=50 Very High (maroon)
      let color = '#4CAF50'; // Low
      if (vc >= 50) {
        color = '#800000'; // Very High (maroon)
      } else if (vc >= 35) {
        color = '#F44336'; // High (red)
      } else if (vc >= 20) {
        color = '#FF9800'; // Moderate (orange)
      }

      const marker = new google.maps.Marker({
        position: coords,
        map: mapInstanceRef.current,
        title: `${intersection.Intersection_ID} - ${intersection.Region_Name}`,
        icon: {
          path: google.maps.SymbolPath.CIRCLE,
          scale: 8,
          fillColor: color,
          fillOpacity: 0.8,
          strokeColor: '#FFFFFF',
          strokeWeight: 2,
        },
        // Remove numeric count label on marker; color now represents density
      });

      // Add info window
      const infoWindow = new google.maps.InfoWindow({
        content: `
          <div style="padding: 8px;">
            <h3 style="margin: 0 0 8px 0; font-size: 16px;">${intersection.Intersection_ID}</h3>
            <p style="margin: 4px 0;"><strong>Region:</strong> ${intersection.Region_Name}</p>
            <p style="margin: 4px 0;"><strong>Vehicles:</strong> ${vc}</p>
            <p style="margin: 4px 0;"><strong>Green Time:</strong> ${mg}s</p>
            <p style="margin: 4px 0; color: ${color};"><strong>Density:</strong> ${
              vc >= 50 ? 'Very High' : vc >= 35 ? 'High' : vc >= 20 ? 'Moderate' : 'Low'
            }</p>
          </div>
        `,
      });

      // Add click listener to marker
      if ((marker as any).addListener) {
        (marker as any).addListener('click', () => {
          infoWindow.open(mapInstanceRef.current, marker);
        });
      }

      markersRef.current.push(marker);
    });
  }, [allIntersections]);

  const handleConfirm = async () => {
    if (!region || !intersectionName) {
      setError(t('Please select both Region and Intersection Name.'));
      return;
    }
    setError('');
    setLoading(true);
    try {
      // fetch latest stored row
      const params = new URLSearchParams();
      params.append('region_name', region);
      params.append('intersection_id', intersectionName);
      const resp = await fetch(`http://localhost:5000/api/db/latest?${params.toString()}`);
      const text = await resp.text();
      let data: any = null;
      try {
        data = JSON.parse(text);
      } catch (err) {
        data = { error: text };
      }
      if (!resp.ok) {
        setError(data.error || t('Failed to fetch latest data'));
        setLatest(null);
      } else {
        setLatest(data.row || null);
      }
    } catch (err: any) {
      setError(err.message || String(err));
      setLatest(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-bg">
      <div className="container mx-auto px-6 py-8">
        {onBack && (
          <button
            onClick={onBack}
            className="mb-4 flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>{t('Back')}</span>
          </button>
        )}
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Map Section */}
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-card border border-border">
              <h2 className="text-xl font-semibold mb-4">{t('Live Traffic Map')}</h2>
              <div className="relative w-full h-[500px] rounded-lg border border-border overflow-hidden bg-gray-900">
                <div 
                  ref={mapRef} 
                  className="w-full h-full"
                  style={{ minHeight: '500px', width: '100%' }}
                />
                {/* Density Legend (side overlay) */}
                <div className="absolute top-4 right-4 z-10 bg-card/90 backdrop-blur rounded-md border border-border p-3 text-xs text-foreground shadow-card">
                  <div className="font-semibold mb-2">Traffic Density</div>
                  <div className="flex flex-col gap-2">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#4CAF50' }}></div>
                      <span>Low (&lt; 20)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#FF9800' }}></div>
                      <span>Moderate (20 - 35)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#F44336' }}></div>
                      <span>High (35 - 49)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#800000' }}></div>
                      <span>Very High (50+)</span>
                    </div>
                  </div>
                </div>
                {mapLoading && (
                  <div className="absolute inset-0 flex items-center justify-center bg-card/80 text-muted-foreground z-10">
                    {t('Loading map...')}
                  </div>
                )}
                {!mapLoading && !mapInstanceRef.current && (
                  <div className="absolute inset-0 flex items-center justify-center bg-card text-muted-foreground">
                    <div className="text-center">
                      <p>Map failed to load. Please check your internet connection.</p>
                      <p className="text-sm mt-2">Intersection data will still be available below.</p>
                    </div>
                  </div>
                )}
              </div>
              {allIntersections.length > 0 && (
                <div className="mt-2 text-sm text-muted-foreground">
                  Showing {allIntersections.length} intersection{allIntersections.length !== 1 ? 's' : ''} on map
                </div>
              )}
            </div>
          </div>

          {/* Details Section */}
          <div className="space-y-6">
            <div className="p-6 rounded-xl bg-card border border-border">
              <h2 className="text-xl font-semibold mb-4">{t('Check Specific Intersection')}</h2>
              <div className="space-y-3">
                <select
                  aria-label={t('Select Region')}
                  value={region}
                  onChange={(e) => setRegion(e.target.value)}
                  className="w-full p-3 rounded-xl border border-gray-300 bg-transparent text-white"
                >
                  {regions.map((r) => (
                    <option key={r} value={r} className="bg-card">
                      {r}
                    </option>
                  ))}
                </select>
                <select
                  aria-label={t('Select Intersection')}
                  value={intersectionName}
                  onChange={(e) => setIntersectionName(e.target.value)}
                  className="w-full p-3 rounded-xl border border-gray-300 bg-transparent text-white"
                >
                  {intersections.map((intersection) => (
                    <option key={intersection} value={intersection} className="bg-card">
                      {intersection}
                    </option>
                  ))}
                </select>
                {error && <div className="text-red-400">{error}</div>}
                <button 
                  onClick={handleConfirm} 
                  className="w-full bg-primary text-white px-4 py-2 rounded hover:opacity-90 transition-opacity"
                >
                  {t('Confirm Intersection')}
                </button>
              </div>
            </div>

            <div>
              {loading && <div className="text-sm text-muted-foreground">{t('Loading...')}</div>}
              {latest && (
                <div className="p-4 rounded-xl bg-card border border-border">
                  <h3 className="font-semibold mb-4">{t('Traffic Status for')} {latest.Region_Name} / {latest.Intersection_ID}</h3>
                  {(() => {
                      const mg = parseInt(String(latest.Max_Green_Time || '0'), 10) || 0;
                      const mv = parseInt(String(latest.Max_Vehicle_Count || '0'), 10) || 0;
                      if (mg > 55) {
                        return (
                          <div className="mt-3 p-3 bg-red-500/10 border border-red-300 rounded">
                            <div className="font-semibold text-red-600">{t('Traffic density: Very High')}</div>
                            <div>{t('Estimated vehicles at intersection: up to')} {mv} {t('vehicles.')}</div>
                            <div>{t('Expect significant delays. Consider alternate routes or avoid the area if possible.')}</div>
                            <div className="text-sm text-muted-foreground">{t('Emergency and priority vehicles may be given extended green time.')}</div>
                          </div>
                        );
                      }
                      if (mg < 30) {
                        return (
                          <div className="mt-3 p-3 bg-green-500/10 border border-green-300 rounded">
                            <div className="font-semibold text-green-700">{t('Traffic density: Low')}</div>
                            <div>{t('Estimated vehicles at intersection: up to')} {mv} {t('vehicles.')}</div>
                            <div>{t('Traffic is flowing smoothly. Minimal delays expected.')}</div>
                            <div className="text-sm text-muted-foreground">{t('You should be able to cross quickly. Drive safely and follow signals.')}</div>
                          </div>
                        );
                      }
                      return (
                        <div className="mt-3 p-3 bg-yellow-500/10 border border-yellow-300 rounded">
                          <div className="font-semibold text-yellow-800">{t('Traffic density: Moderate')}</div>
                          <div>{t('Estimated vehicles at intersection: up to')} {mv} {t('vehicles.')}</div>
                          <div>{t('Some delays possible. Exercise caution and follow traffic directions.')}</div>
                          <div className="text-sm text-muted-foreground">{t('Peak adjustments may be active; expect variable signal timings.')}</div>
                        </div>
                      );
                    })()}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};