from fasthtml.common import *
from monsterui.all import *
from pathlib import Path
from starlette.responses import FileResponse
from typing import List

from components import *
from llm_utils import get_suggested_queries
from faiss_utils import find_similar_images_faiss, find_similar_to_images_faiss, find_nearby_similar_images, FaissImageIndex  
from hybrid_image_search import find_similar_images_hybrid

faiss_cache = "faiss_cache"

class AppState:
    def __init__(self):
        self.queries = []

state = AppState()

def setup_routes(app):
    @app.get("/images/{path:path}")
    async def serve_image(path: str):
        from urllib.parse import unquote
        # Decode the URL-encoded path
        decoded_path = unquote(path)
        # Handle Windows paths correctly
        system_path = Path(decoded_path)
        if not system_path.exists():
            return Card(P(f"Image not found: {decoded_path}", cls=TextT.error))
        return FileResponse(str(system_path))

    @app.get("/")
    def home():
        return Titled("Context-Aware Image Search Engine: Find by Style and Content", Container(
            H1("Search using plain English descriptions, find visually similar images, and combine location data with visual similarity", cls=TextPresets.bold_lg),
            Section(
                DescriptionForm(),
                Div(id="suggestions"),
                AddQueryForm(),
                QueryListContainer(state.queries),
                SearchForm(),
                Div(id="results"),
                ImageModal(),
                cls="space-y-6"
            ),
            cls=ContainerT.xl
        )
    )

    @app.post("/suggest")
    async def suggest(description: str):
        try:
            suggested_queries = await get_suggested_queries(description)
            return SuggestionsList(suggested_queries)
        except Exception as e:
            return Card(
                P(f"Error getting suggestions: {str(e)}", cls=TextT.error),
                id="suggestions"
            )

    @app.post("/add_query")
    def add_query(query: str, weight: float):
        state.queries.append((query, weight))
        return QueryListContainer(state.queries)

    @app.put("/update_weight/{idx}")
    async def update_weight(idx: int, request: Request):
        try:
            form = await request.form()
            value = float(form.get('value', 1.0))
            print(f"Updating weight at index {idx} to {value}")
            state.queries[idx] = (state.queries[idx][0], value)
            return QueryListContainer(state.queries)
        except Exception as e:
            print(f"Error updating weight: {e}")
            return Card(
                P(f"Error updating weight: {str(e)}", cls=TextT.error),
                id="query-list"
            )


    @app.delete("/delete_query/{idx}")
    def delete_query(idx: int):
        state.queries.pop(idx)
        return QueryListContainer(state.queries)

    @app.get("/images/{path:path}")
    async def serve_image(path: str):
        return FileResponse(path)
        
    @app.post("/search_faiss")
    async def search_faiss(
        directory: str = "", 
        use_location: Optional[str] = None
    ):
        try:
            if not state.queries:
                return Card(
                    P("Please add at least one search query first", cls=TextT.warning),
                    id="results"
                )
            
            # Convert checkbox state to boolean - simply check if it exists
            use_location_bool = use_location is not None
            print(f"Location search enabled: {use_location_bool}")

            results = await find_similar_images_faiss(
                directory,
                state.queries,
                batch_size=256,
                min_score=15.0,
                cache_dir=faiss_cache,
                use_location=use_location_bool
            )
            
            return ResultsDisplay(
                results, 
                directory,
                f"Found {len(results)} matches"
            )
            
        except Exception as e:
            return Card(
                P(f"Error: {str(e)}", cls=TextT.error),
                id="results"
            )
        
    @app.post("/similar-multiple-faiss")
    async def find_similar_multiple_faiss(request: Request):
        try:
            form = await request.form()
            directory = form.get('directory')
            selected_refs = form.getlist('selected_refs')
            
            if not selected_refs:
                return Card(
                    P("Please select at least one image", cls=TextT.warning),
                    id="results",
                    hx_swap_oob="true"
                )
            
            weight = 1.0 / len(selected_refs)
            references = [(ref, weight) for ref in selected_refs]
            
            results = find_similar_to_images_faiss(
                references,
                directory,
                batch_size=256,
                min_score=20.0,
                cache_dir=faiss_cache
            )
            
            return Div(
                ResultsDisplay(
                    results,
                    directory,
                    f"Found {len(results)} images similar to {len(selected_refs)} selected images"
                ),
                id="results",
                hx_swap_oob="true"
            )
            
        except Exception as e:
            return Card(
                P(f"Error: {str(e)}", cls=TextT.error),
                id="results",
                hx_swap_oob="true"
            )
        
    @app.post("/clear_faiss_index")
    def clear_faiss_index():
        try:
            index = FaissImageIndex(cache_dir=faiss_cache)
            index.clear_index()
            return Card(
                P("Faiss index cleared successfully", cls=TextT.success),
                id="results"
            )
        except Exception as e:
            return Card(
                P(f"Error clearing index: {str(e)}", cls=TextT.error),
                id="results"
            )
    
    @app.post("/nearby-similar")
    async def find_nearby_similar(request: Request):
        try:
            params = request.query_params
            path = params.get('path')
            
            if not path:
                return Card(
                    P("Missing path parameter", cls=TextT.error),
                    id="results"
                )
                
            path = path.replace('/', os.path.sep)
            
            print(f"Searching for images near: {path}")
            results = find_nearby_similar_images(
                path,
                max_distance_km=10.0,
                min_similarity=30.0
            )
            
            if not results:
                return Card(
                    P("No nearby similar images found", cls=TextT.warning),
                    id="results"
                )
            
            return ResultsDisplay(
                results,
                os.path.dirname(path),
                f"Found {len(results)} nearby similar images"
            )
            
        except Exception as e:
            return Card(
                P(f"Error: {str(e)}", cls=TextT.error),
                id="results"
            )
        
    @app.post("/search_hybrid")
    async def search_hybrid(
        directory: str = "", 
        use_location: Optional[str] = None
    ):
        try:
            if not state.queries:
                return Card(
                    P("Please add at least one search query first", cls=TextT.warning),
                    id="results"
                )
            
            # if not directory:
            #     return Card(
            #         P("Please select an image directory first", cls=TextT.warning),
            #         id="results"
            #     )
            
            # Convert checkbox state to boolean - simply check if it exists
            use_location_bool = use_location is not None
            print(f"Location search enabled: {use_location_bool}")
            
            # Create cache directories if they don't exist
            cache_dir = Path("search_cache")
            clip_cache = cache_dir / "clip"
            sd_cache = cache_dir / "sd"
            clip_cache.mkdir(parents=True, exist_ok=True)
            sd_cache.mkdir(parents=True, exist_ok=True)
            
            results = await find_similar_images_hybrid(
                directory,
                state.queries,
                batch_size=32,
                min_score=15.0,
                cache_dir=str(cache_dir),
                clip_weight=0.6,
                sd_weight=0.4,
                use_location=use_location_bool
            )
            
            return ResultsDisplay(
                results, 
                directory,
                f"Found {len(results)} matches (Hybrid CLIP+SD)"
            )
            
        except Exception as e:
            print(f"Error in hybrid search route: {str(e)}")
            return Card(
                P(f"Error: {str(e)}", cls=TextT.error),
                id="results"
            )