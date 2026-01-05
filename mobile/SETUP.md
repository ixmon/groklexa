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

### 4. Apply Android overrides (required for mic + SSL)

Copy the override files to enable microphone and self-signed certificate support:

```bash
# Copy MainActivity.java (handles permissions + SSL bypass)
cp android-overrides/app/src/main/java/com/groklexa/app/MainActivity.java \
   android/app/src/main/java/com/groklexa/app/MainActivity.java

# Copy network security config
mkdir -p android/app/src/main/res/xml
cp android-overrides/app/src/main/res/xml/network_security_config.xml \
   android/app/src/main/res/xml/network_security_config.xml
```

Then edit `android/app/src/main/AndroidManifest.xml`:

1. Add permissions before the `<application>` tag:
```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS" />
<uses-permission android:name="android.permission.INTERNET" />
```

2. Add network security config to the `<application>` tag:
```xml
<application
    android:networkSecurityConfig="@xml/network_security_config"
    ... other attributes ...>
```

### 5. Open in Android Studio

```bash
npx cap open android
```

### 6. Build and run

In Android Studio:
1. Wait for Gradle sync to complete
2. Connect your phone via USB (enable USB debugging)
3. Click the green "Run" button

## Self-Signed Certificates

The `MainActivity.java` override includes SSL bypass for development. This allows the app to connect to servers with self-signed certificates without additional steps.

**⚠️ Warning:** The SSL bypass should only be used for development. For production, use a proper CA-signed certificate.

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
