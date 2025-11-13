declare namespace google {
  namespace maps {
    class Map {
      constructor(element: HTMLElement, options?: MapOptions);
      setCenter(latlng: LatLng | LatLngLiteral): void;
      setZoom(zoom: number): void;
    }

    class Marker {
      constructor(options?: MarkerOptions);
      setMap(map: Map | null): void;
      setPosition(latlng: LatLng | LatLngLiteral): void;
      setIcon(icon: string | Icon | Symbol): void;
      addListener(eventName: string, handler: Function): void;
    }

    interface MapOptions {
      center?: LatLng | LatLngLiteral;
      zoom?: number;
      mapTypeId?: MapTypeId;
    }

    interface MarkerOptions {
      position?: LatLng | LatLngLiteral;
      map?: Map | null;
      title?: string;
      icon?: string | Icon | Symbol;
      label?: string | MarkerLabel;
    }

    interface LatLng {
      lat(): number;
      lng(): number;
    }

    interface LatLngLiteral {
      lat: number;
      lng: number;
    }

    interface Icon {
      url: string;
      scaledSize?: Size;
      anchor?: Point;
    }

    interface Symbol {
      path: SymbolPath | string;
      scale?: number;
      fillColor?: string;
      fillOpacity?: number;
      strokeColor?: string;
      strokeOpacity?: number;
      strokeWeight?: number;
      rotation?: number;
      anchor?: Point;
    }

    class InfoWindow {
      constructor(options?: InfoWindowOptions);
      open(map?: Map | null, anchor?: Marker): void;
      close(): void;
      setContent(content: string | HTMLElement): void;
    }

    interface InfoWindowOptions {
      content?: string | HTMLElement;
      position?: LatLng | LatLngLiteral;
      map?: Map | null;
    }

    namespace SymbolPath {
      const CIRCLE: number;
      const BACKWARD_CLOSED_ARROW: number;
      const BACKWARD_OPEN_ARROW: number;
      const FORWARD_CLOSED_ARROW: number;
      const FORWARD_OPEN_ARROW: number;
    }

    interface MarkerLabel {
      text: string;
      color?: string;
      fontSize?: string;
      fontWeight?: string;
    }

    interface Size {
      width: number;
      height: number;
    }

    interface Point {
      x: number;
      y: number;
    }

    enum MapTypeId {
      ROADMAP = 'roadmap',
      SATELLITE = 'satellite',
      HYBRID = 'hybrid',
      TERRAIN = 'terrain',
    }

    type SymbolPath = number;
  }
}

