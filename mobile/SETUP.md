# Groklexa Mobile App Setup

This is a Capacitor-based Android/iOS app that connects to your Groklexa server.

## Prerequisites

- Node.js 18+ and npm
- Android Studio (for Android builds)
- Xcode (for iOS builds, macOS only)
- Your Groklexa server running and accessible over HTTPS

## Quick Start (Android)

### 1. Install dependencies

```bash
cd mobile
npm install
```

### 2. Configure server URL

Edit `capacitor.config.ts` and set your server IP:

```typescript
server: {
    url: 'https://192.168.1.XXX:5001',  // Your server's IP
    ...
}
```

### 3. Initialize Capacitor and add Android

```bash
npx cap add android
```

### 4. Open in Android Studio

```bash
npx cap open android
```

### 5. Build and run

In Android Studio:
1. Wait for Gradle sync to complete
2. Connect your phone via USB (enable USB debugging)
3. Click the green "Run" button

## Trusting the Self-Signed Certificate

Since your server uses a self-signed certificate:

1. On your phone, open Chrome and navigate to `https://YOUR_SERVER_IP:5001`
2. Accept the security warning and proceed
3. The certificate is now trusted for the Capacitor app

Alternatively, for development, you can configure Android to trust all certificates (not recommended for production).

## Troubleshooting

### "net::ERR_CERT_AUTHORITY_INVALID"
The app can't verify your self-signed certificate. Visit the URL in Chrome first to accept it.

### "net::ERR_CONNECTION_REFUSED"
- Make sure Flask is running: `uv run python web_app.py`
- Make sure your phone is on the same network as the server
- Check firewall allows port 5001

### Microphone not working
The app needs microphone permissions. Check Android Settings > Apps > Groklexa > Permissions.

## Development Workflow

After making changes to the web app:

```bash
# No sync needed - the app loads directly from your server!
# Just refresh in the app or restart it
```

## Building for Production

For a standalone APK that doesn't need a server connection, you would:

1. Bundle the web assets into `www/`
2. Change `capacitor.config.ts` to use `webDir: 'www'` instead of `server.url`
3. Implement local inference (future work)

For now, the server-connected mode is recommended.
