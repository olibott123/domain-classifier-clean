#!/bin/bash

# Create RSA key from base64 environment variable
if [ ! -z "$SNOWFLAKE_KEY_BASE64" ]; then
  echo "$SNOWFLAKE_KEY_BASE64" | base64 -d > /workspace/rsa_key.der
  chmod 600 /workspace/rsa_key.der
  echo "Created Snowflake key file from environment variable"
else
  echo "SNOWFLAKE_KEY_BASE64 environment variable not set. Snowflake integration will be disabled."
fi

# Start the application with increased timeout
exec gunicorn --timeout 300 --bind 0.0.0.0:${PORT:-8080} api_service:app
