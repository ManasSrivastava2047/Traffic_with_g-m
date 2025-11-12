import React, { useState } from 'react';
import { TrafficDashboard } from '@/components/TrafficDashboard';
import { DriverView } from '@/components/DriverView';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { useTranslation } from '@/contexts/TranslationContext';

const INDIAN_STATES_AND_UTS = [
  'Andhra Pradesh',
  'Arunachal Pradesh',
  'Assam',
  'Bihar',
  'Chhattisgarh',
  'Goa',
  'Gujarat',
  'Haryana',
  'Himachal Pradesh',
  'Jharkhand',
  'Karnataka',
  'Kerala',
  'Madhya Pradesh',
  'Maharashtra',
  'Manipur',
  'Meghalaya',
  'Mizoram',
  'Nagaland',
  'Odisha',
  'Punjab',
  'Rajasthan',
  'Sikkim',
  'Tamil Nadu',
  'Telangana',
  'Tripura',
  'Uttar Pradesh',
  'Uttarakhand',
  'West Bengal',
  'Andaman and Nicobar Islands',
  'Chandigarh',
  'Dadra and Nagar Haveli and Daman and Diu',
  'Delhi',
  'Jammu and Kashmir',
  'Ladakh',
  'Lakshadweep',
  'Puducherry',
];

const Index = () => {
  const [role, setRole] = useState<'selector' | 'authority' | 'driver'>('selector');
  const { selectedState, setSelectedState, t } = useTranslation();

  if (role === 'authority') return <TrafficDashboard />;
  if (role === 'driver') return <DriverView />;

  return (
    <div className="min-h-screen bg-gradient-bg relative">
      {/* State Dropdown - Top Left */}
      <div className="absolute top-6 left-6 z-10">
        <div className="flex flex-col gap-2">
          <Label htmlFor="state-select" className="text-sm font-medium text-foreground">
            {t('Select State')}
          </Label>
          <Select value={selectedState} onValueChange={setSelectedState}>
            <SelectTrigger id="state-select" className="w-[250px] bg-card border-border">
              <SelectValue placeholder={t('Choose a state or UT')} />
            </SelectTrigger>
            <SelectContent className="max-h-[300px]">
              {INDIAN_STATES_AND_UTS.map((state) => (
                <SelectItem key={state} value={state}>
                  {state}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Main Content - Centered */}
      <div className="min-h-screen flex items-center justify-center">
        <div className="space-y-8 text-center">
          <h1 className="text-3xl font-bold bg-gradient-primary bg-clip-text text-transparent">
            {t('AI Traffic Management System')}
          </h1>
          <p className="text-muted-foreground">{t('Select your role to continue')}</p>
          <div className="flex gap-8 justify-center">
            <button 
              onClick={() => setRole('driver')}
              className="w-64 h-64 bg-gradient-card border-2 border-border rounded-2xl p-6 group hover:shadow-glow transition-all duration-300 hover:scale-105"
            >
              <div className="h-full flex flex-col items-center justify-center space-y-4">
                <div className="w-20 h-20 bg-primary/10 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform">
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-10 h-10 text-primary" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                </div>
                <h2 className="text-xl font-semibold">{t('Driver')}</h2>
                <p className="text-sm text-muted-foreground">{t('Check traffic conditions at intersections')}</p>
              </div>
            </button>

            <button 
              onClick={() => setRole('authority')}
              className="w-64 h-64 bg-gradient-card border-2 border-border rounded-2xl p-6 group hover:shadow-glow transition-all duration-300 hover:scale-105"
            >
              <div className="h-full flex flex-col items-center justify-center space-y-4">
                <div className="w-20 h-20 bg-primary/10 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform">
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-10 h-10 text-primary" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                  </svg>
                </div>
                <h2 className="text-xl font-semibold">{t('Traffic Authority')}</h2>
                <p className="text-sm text-muted-foreground">{t('Manage and analyze traffic patterns')}</p>
              </div>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
