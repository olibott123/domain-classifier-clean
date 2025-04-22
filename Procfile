web: mkdir -p /workspace && echo "$SNOWFLAKE_KEY_BASE64" | base64 -d > /workspace/rsa_key.der && chmod 600 /workspace/rsa_key.der && pip install -e . && gunicorn --config gunicorn_config.py main:app
