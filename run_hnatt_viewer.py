import os, subprocess

os.environ['FLASK_APP'] = "app/app.py"
os.environ['FLASK_DEBUG'] = "1"
subprocess.call(['flask', 'run'])