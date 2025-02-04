from fasthtml.common import *
from monsterui.all import *
from pathlib import Path
from starlette.responses import FileResponse

# Import our modules
from routes import setup_routes

# Create your app with the theme
app = FastHTML(hdrs=Theme.blue.headers())

# Setup routes
setup_routes(app)

if __name__ == "__main__":
    serve() 