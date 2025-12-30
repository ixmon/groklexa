#!/bin/bash
# Generate a self-signed SSL certificate for local development

DOMAIN="spark.home.arpa"
CERT_DIR="certs"

mkdir -p "$CERT_DIR"

# Generate private key
openssl genrsa -out "$CERT_DIR/key.pem" 2048

# Generate certificate signing request and self-signed certificate
openssl req -new -x509 -key "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" -days 365 \
    -subj "/C=US/ST=State/L=City/O=Development/CN=$DOMAIN" \
    -addext "subjectAltName=DNS:$DOMAIN,DNS:*.home.arpa,DNS:localhost,IP:127.0.0.1"

echo "Certificate generated successfully!"
echo "Files created:"
echo "  - $CERT_DIR/key.pem (private key)"
echo "  - $CERT_DIR/cert.pem (certificate)"
echo ""
echo "You can now start the server with HTTPS support."
echo "Note: Your browser will show a security warning - click 'Advanced' and 'Proceed' to accept the self-signed certificate."

