package com.groklexa.app;

import android.Manifest;
import android.content.pm.PackageManager;
import android.net.http.SslError;
import android.os.Bundle;
import android.webkit.PermissionRequest;
import android.webkit.SslErrorHandler;
import android.webkit.WebChromeClient;
import android.webkit.WebView;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.getcapacitor.BridgeActivity;

/**
 * MainActivity for Groklexa Android app.
 * 
 * This class handles:
 * - SSL bypass for self-signed development certificates
 * - Runtime microphone permission requests
 * - WebView permission grants for getUserMedia
 */
public class MainActivity extends BridgeActivity {
    private static final int PERMISSION_REQUEST_CODE = 1;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        // Request microphone permission at runtime (required for Android 6+)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) 
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, 
                new String[]{Manifest.permission.RECORD_AUDIO}, 
                PERMISSION_REQUEST_CODE);
        }
        
        // Allow self-signed certificates in development
        // WARNING: This bypasses SSL validation - only use for development!
        bridge.getWebView().setWebViewClient(new android.webkit.WebViewClient() {
            @Override
            public void onReceivedSslError(WebView view, SslErrorHandler handler, SslError error) {
                handler.proceed();
            }
        });
        
        // Grant microphone permission to WebView when requested by JavaScript
        bridge.getWebView().setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(PermissionRequest request) {
                request.grant(request.getResources());
            }
        });
    }
}
