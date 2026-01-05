import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.groklexa.app',
  appName: 'Groklexa',
  
  // For development: load from your Flask server
  // Change this to your server's IP address
  server: {
    url: 'https://YOUR_SERVER_IP:5001',
    cleartext: false,  // We use HTTPS
    allowNavigation: ['*']
  },
  
  // For production: use bundled web assets
  // Uncomment this and comment out 'server' above:
  // webDir: 'www',
  
  android: {
    // Allow self-signed certificates in development
    allowMixedContent: true,
  },
  
  plugins: {
    // Microphone permissions
    Permissions: {
      permissions: ['microphone']
    }
  }
};

export default config;
