# Gunicorn configuration file
import os
# Set timeout to 5 minutes (300 seconds)
# This allows long-running requests to complete
timeout = 300
# Number of worker processes
workers = int(os.environ.get('GUNICORN_WORKERS', 2))
# Pre-load application code before forking
preload_app = True
# Maximum requests a worker will process before restarting
max_requests = 1000
max_requests_jitter = 50
# Log level
loglevel = 'info'
# Enable access logging
accesslog = '-'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
# Bind to all interfaces
bind = '0.0.0.0:' + os.environ.get('PORT', '8080')
# Change this line to point to your new application entry point
wsgi_app = 'main:app'
