from fasthtml.common import *
from monsterui.all import *
from image_utils import get_gps_data

def QueryItem(query: str, weight: float, index: int):
    return Card(
        DivFullySpaced(
            DivLAligned(
                P(query, cls=TextPresets.bold_sm),
                Div(
                    Label("Weight:"),
                    Input(
                        type="range",
                        value=str(weight),
                        min="0",
                        max="2",
                        step="0.1",
                        name="value",
                        hx_put=f"/update_weight/{index}",
                        hx_trigger="change",
                        hx_target="#query-list",
                        hx_include="this",
                        oninput="this.nextElementSibling.textContent = this.value"
                    ),
                    Span(f"{weight:.1f}", cls="weight-display"),
                    cls="query-controls"
                )
            ),
            Button(
                "×",
                hx_delete=f"/delete_query/{index}",
                hx_target="#query-list",
                cls=ButtonT.danger
            )
        ),
        cls=CardT.hover
    )

def SearchForm(directory: str = ""):
    return Form(
        DivFullySpaced(
            # Left side with directory input
            LabelInput("Search Directory:", 
                      id="directory",
                      name="directory",
                      value=directory,
                      required=True),
            # Right side with controls
            DivLAligned(
                # Location toggle
                DivLAligned(
                    FormLabel("Use Location", fr="use_location"),
                    Switch(id="use_location", 
                          name="use_location",
                          value="true",
                          checked=True), 
                    cls="mr-4"
                ),
                # Clear index button
                Button(
                    DivLAligned(
                        Div("Clear Faiss Index", id="clear-index-text"),
                        Loading(htmx_indicator=True, type=LoadingT.spinner, cls="htmx-indicator")
                    ),
                    cls=ButtonT.ghost,
                    hx_post="/clear_faiss_index",
                    hx_target="#results"
                )
            )
        ),
        DivLAligned(
            Button(
                DivLAligned(
                    Div("Search with CLIP", id="search-faiss-text"),
                    Loading(htmx_indicator=True, type=LoadingT.spinner, cls="htmx-indicator")
                ),
                cls=ButtonT.primary,
                hx_post="/search_faiss",
                hx_target="#results"
            ),
            Button(
                DivLAligned(
                    Div("Hybrid Search (CLIP+SD)", id="search-hybrid-text"),
                    Loading(htmx_indicator=True, type=LoadingT.spinner, cls="htmx-indicator")
                ),
                cls=ButtonT.secondary,
                hx_post="/search_hybrid",
                hx_target="#results"
            )
        )
    )

def DescriptionForm():
    return Form(
        LabelTextArea(
            "Detailed Description:",
            id="description",
            placeholder="Describe the kind of images you're looking for in detail...",
            required=True
        ),
        Button(
            DivLAligned(
                Div("Get Query Suggestions", id="suggest-text"),
                Loading(htmx_indicator=True, 
                       type=LoadingT.dots, 
                       cls="htmx-indicator")
            ),
            cls=ButtonT.primary
        ),
        hx_post="/suggest",
        hx_target="#suggestions"
    )


def SuggestedQuery(query: str, weight: float):
    return Card(
        DivFullySpaced(
            DivLAligned(
                P(f"{query} (weight: {weight})", cls=TextPresets.bold_sm)
            ),
            Button(
                "Add This Query",
                hx_post="/add_query",
                hx_vals=f'{{"query": "{query}", "weight": {weight}}}',
                hx_target="#query-list",
                cls=ButtonT.secondary
            )
        ),
        cls=CardT.hover
    )

def SuggestionsList(suggestions):
    return Div(
        H3("Suggested Queries:", cls=TextPresets.bold_lg),
        *[SuggestedQuery(query, weight) for query, weight in suggestions],
        id="suggestions"
    )

def ImageModal():
    return Modal(
        ModalHeader(H3("Image Preview")),
        ModalBody(
            Img(
                id="modal-preview-img", 
                style="width: 100%; height: 90vh; object-fit: contain;"
            ),
            cls="p-0"
        ),
        id="image-modal",
        size="full"  # MonsterUI's full-size dialog option
    )

def ResultItem(result, directory: str):
    path = str(Path(result['path'])).replace('\\', '/')
    img_url = f"/images/{path}"
    
    # Check if image has GPS data
    has_gps = get_gps_data(result['path']) is not None
    
    # Get various scores
    visual_score = result.get('visual_score', result.get('mean_score', 0))
    distance_km = result.get('distance_km')
    distance_score = result.get('distance_score', 0)
    combined_score = result.get('combined_score', visual_score)
    location_scores = {k: v for k, v in result['individual_scores'].items() if k.startswith('location_')}

    nearby_url = f"/nearby-similar?path={result['path']}&directory={directory}"
    nearby_url = nearby_url.replace('\\', '/')  # Ensure URL-friendly path

    return Card(
        Div(
            LabelCheckboxX(
                label="Find Similar",
                name="selected_refs",
                value=str(result['path']),
                oninput="""
                    const hasChecked = document.querySelectorAll('input[name=selected_refs]:checked').length > 0;                    
                    document.getElementById('multi-similar-faiss-btn').disabled = !hasChecked;
                """
            ),
            Img(
                src=img_url,
                uk_toggle="target: #image-modal",
                onclick=f"document.getElementById('modal-preview-img').src = '{img_url}'"
            ),
            # Score section
            Div(
                H4(f"Combined Score: {combined_score:.1f}%"),
                P(f"Visual Similarity: {visual_score:.1f}%", cls=TextPresets.bold_sm),
                *[Div(
                    P(f"Location: {loc.replace('location_', '')}", cls=TextPresets.bold_sm),
                    P(f"Match: {score['score']:.1f}% ({score['distance']:.1f}km away)", 
                      cls=TextPresets.muted_sm)
                ) for loc, score in location_scores.items()],
                *([] if distance_km is None else [
                    P(f"Distance: {distance_km:.1f}km", cls=TextPresets.bold_sm),
                    P(f"Location Score: {distance_score:.1f}%", cls=TextPresets.muted_sm)
                ]),
                cls="space-y-2"
            ),
            # Path and GPS info
            P(f"Path: {result['path']}", cls=TextPresets.muted_sm),
            # Add Find Nearby Similar button if image has GPS data
            *([Button(
                DivLAligned(
                    Div("Find Nearby Similar", id=f"nearby-text-{path}"),
                    Loading(htmx_indicator=True, type=LoadingT.spinner, cls="htmx-indicator")
                ),
                cls=ButtonT.secondary,
                hx_post=nearby_url,
                hx_target="#results"
            )] if has_gps else [P("No GPS data", cls=TextT.muted)]),
        ),
        cls=CardT.hover
    )

def ResultsDisplay(results, directory: str, title: str):
    return Div(
        H2(title),
        Form(
            Grid(
                *[ResultItem(result, directory) for result in results[:100]],
                cols_sm=1, cols_md=2, cols_lg=3, cols_xl=4, gap=4
            ),
            Input(type="hidden", name="directory", value=directory),
            Div(
                DivLAligned(
                    Button(
                        DivLAligned(
                            Div("Find Similar", id="multi-similar-faiss-text"),
                            Loading(htmx_indicator=True, type=LoadingT.spinner, cls="htmx-indicator")
                        ),
                        type="submit",
                        disabled=True,
                        id="multi-similar-faiss-btn",
                        cls=(ButtonT.primary, "search-button"),
                        hx_post="/similar-multiple-faiss"
                    )
                ),
                cls="multi-select-controls"
            ),
            id="results"
        )
    )


def QueryListContainer(queries):
    if not queries:
        return Card(
            P("No queries added yet", cls=TextPresets.muted_sm),
            id="query-list"
        )
    
    return Div(
        *[QueryItem(query, weight, idx) 
          for idx, (query, weight) in enumerate(queries)],
        id="query-list",
        cls="space-y-4"
    )

def AddQueryForm():
    return Form(
        LabelInput(
            "Search Query:",
            id="query",
            placeholder="Enter search term",
            required=True
        ),
        DivLAligned(
            Label("Initial Weight:"),
            Input(
                type="range",
                name="weight",
                value="1.0",
                min="0",
                max="2",
                step="0.1",
                oninput="this.nextElementSibling.textContent = this.value"
            ),
            Span("1.0", cls="weight-display")
        ),
        Button(
            "Add Query",
            type="submit",
            cls=ButtonT.primary
        ),
        hx_post="/add_query",
        hx_target="#query-list",
        id="add-query-form"
    )