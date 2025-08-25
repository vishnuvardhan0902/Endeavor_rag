from api import app

# This module is a small shim so existing run commands like
#   uvicorn main:app --reload
# continue to work even though the app live in `api.py`.
