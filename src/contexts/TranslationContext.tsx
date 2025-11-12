import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface TranslationContextType {
  translations: Record<string, string>;
  selectedState: string;
  setSelectedState: (state: string) => void;
  isLoading: boolean;
  t: (key: string, fallback?: string) => string;
}

const TranslationContext = createContext<TranslationContextType | undefined>(undefined);

// Default English translations
const DEFAULT_TRANSLATIONS: Record<string, string> = {
  'Select State': 'Select State',
  'Choose a state or UT': 'Choose a state or UT',
  'AI Traffic Management System': 'AI Traffic Management System',
  'Select your role to continue': 'Select your role to continue',
  'Driver': 'Driver',
  'Check traffic conditions at intersections': 'Check traffic conditions at intersections',
  'Traffic Authority': 'Traffic Authority',
  'Manage and analyze traffic patterns': 'Manage and analyze traffic patterns',
  'Driver — Check Intersection': 'Driver — Check Intersection',
  'Select Region': 'Select Region',
  'Select Intersection': 'Select Intersection',
  'Please select both Region and Intersection Name.': 'Please select both Region and Intersection Name.',
  'Confirm Intersection': 'Confirm Intersection',
  'Loading...': 'Loading...',
  'Traffic Status for': 'Traffic Status for',
  'Traffic density: Very High': 'Traffic density: Very High',
  'Estimated vehicles at intersection: up to': 'Estimated vehicles at intersection: up to',
  'vehicles.': 'vehicles.',
  'Expect significant delays. Consider alternate routes or avoid the area if possible.': 'Expect significant delays. Consider alternate routes or avoid the area if possible.',
  'Emergency and priority vehicles may be given extended green time.': 'Emergency and priority vehicles may be given extended green time.',
  'Traffic density: Low': 'Traffic density: Low',
  'Traffic is flowing smoothly. Minimal delays expected.': 'Traffic is flowing smoothly. Minimal delays expected.',
  'You should be able to cross quickly. Drive safely and follow signals.': 'You should be able to cross quickly. Drive safely and follow signals.',
  'Traffic density: Moderate': 'Traffic density: Moderate',
  'Some delays possible. Exercise caution and follow traffic directions.': 'Some delays possible. Exercise caution and follow traffic directions.',
  'Peak adjustments may be active; expect variable signal timings.': 'Peak adjustments may be active; expect variable signal timings.',
  'AI Traffic Management': 'AI Traffic Management',
  'Real-time traffic optimization system': 'Real-time traffic optimization system',
  'System Online': 'System Online',
  'Enter Region': 'Enter Region',
  'Failed to fetch latest data': 'Failed to fetch latest data',
  'Recent Activity': 'Recent Activity',
  'vehicles detected': 'vehicles detected',
  'signal': 'signal',
  'min ago': 'min ago',
};

interface TranslationProviderProps {
  children: ReactNode;
}

export const TranslationProvider: React.FC<TranslationProviderProps> = ({ children }) => {
  const [translations, setTranslations] = useState<Record<string, string>>(DEFAULT_TRANSLATIONS);
  const [selectedState, setSelectedStateInternal] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);

  // Load selected state from localStorage on mount
  useEffect(() => {
    const savedState = localStorage.getItem('selectedState');
    if (savedState) {
      setSelectedStateInternal(savedState);
    }
  }, []);

  // Fetch translations when state changes
  useEffect(() => {
    if (!selectedState) {
      setTranslations(DEFAULT_TRANSLATIONS);
      return;
    }

    const fetchTranslations = async () => {
      setIsLoading(true);
      try {
        // First, check what language the state uses
        const langResponse = await fetch(`http://localhost:5000/api/language/state?state=${encodeURIComponent(selectedState)}`);
        if (!langResponse.ok) {
          setTranslations(DEFAULT_TRANSLATIONS);
          setIsLoading(false);
          return;
        }
        
        const langData = await langResponse.json();
        const targetLanguage = langData.language || 'Hindi';
        
        // If language is English, no translation needed
        if (targetLanguage === 'English') {
          setTranslations(DEFAULT_TRANSLATIONS);
          setIsLoading(false);
          return;
        }

        // Get all translation keys
        const keys = Object.keys(DEFAULT_TRANSLATIONS);
        const texts = keys.map(key => DEFAULT_TRANSLATIONS[key]);

        // Fetch translations from backend
        const response = await fetch('http://localhost:5000/api/translate/batch', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            texts: texts,
            state: selectedState,
          }),
        });

        if (response.ok) {
          const data = await response.json();
          const translatedTexts = data.translated || [];

          // Create translations object
          const newTranslations: Record<string, string> = {};
          keys.forEach((key, index) => {
            newTranslations[key] = translatedTexts[index] || DEFAULT_TRANSLATIONS[key];
          });

          setTranslations(newTranslations);
        } else {
          console.error('Failed to fetch translations');
          setTranslations(DEFAULT_TRANSLATIONS);
        }
      } catch (error) {
        console.error('Error fetching translations:', error);
        setTranslations(DEFAULT_TRANSLATIONS);
      } finally {
        setIsLoading(false);
      }
    };

    fetchTranslations();
  }, [selectedState]);

  const setSelectedState = (state: string) => {
    setSelectedStateInternal(state);
    localStorage.setItem('selectedState', state);
  };

  const t = (key: string, fallback?: string): string => {
    return translations[key] || fallback || key;
  };

  return (
    <TranslationContext.Provider
      value={{
        translations,
        selectedState,
        setSelectedState,
        isLoading,
        t,
      }}
    >
      {children}
    </TranslationContext.Provider>
  );
};

export const useTranslation = (): TranslationContextType => {
  const context = useContext(TranslationContext);
  if (!context) {
    throw new Error('useTranslation must be used within a TranslationProvider');
  }
  return context;
};

