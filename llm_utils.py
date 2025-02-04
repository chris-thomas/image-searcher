import httpx
import json

async def get_suggested_queries(description: str):
    """Call LLM API to get suggested queries and weights"""
    CLIP_PATTERNS = """
    Common LAION-2B/CLIP caption patterns:
    - "photograph of {subject}"
    - "{style} photo of {subject}"
    - "high resolution {subject}"
    - "professional photograph of {subject}"
    - "detailed image of {subject}"
    """
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1234/v1/chat/completions",
            json={
                "messages": [{
                    "role": "user",
                    "content": f"""Given this description: {description}

Using these patterns: {CLIP_PATTERNS}

Suggest 4-6 search queries that match CLIP caption styles.
For each query, assign a weight (0-2) based on importance.

Return ONLY a JSON array of [query, weight] pairs."""
                }],
                "temperature": 0.7,
                "max_tokens": 500
            }
        )
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        if content.startswith('```json'): content = content[7:-3]
        return json.loads(content)

async def get_location_from_query(query: str) -> dict:
    """Ask LLM to identify location and provide coordinates if present"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:  # 10 second timeout
            response = await client.post(
                "http://localhost:1234/v1/chat/completions",
                json={
                    "messages": [{
                        "role": "user",
                        "content": f"""Given this search query: {query}

If this contains the name of a country, city, town or village, return a JSON object with:
- location_name: The identified location
- latitude: Approximate latitude
- longitude: Approximate longitude
- confidence: How confident you are this is correct (0-100%)
- has_location: true

Do not guess the location based on a generic term like Botantical Gardens or Cyberpunk city.

Only return a location if the query contains the name of a country, city, town or village, otherwise return {{"has_location": false}}

If no location is mentioned return {{"has_location": false}}

If you are not 100% confident, return {{"has_location": false}}

Return ONLY the JSON object."""
                    }],
                    "temperature": 0.1,
                    "max_tokens": 500
                }
            )
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            if content.startswith('```json'): content = content[7:-3]
            data = json.loads(content)
            
            # Ensure has_location is set based on presence of coordinates
            if 'latitude' in data and 'longitude' in data:
                data['has_location'] = True
            return data
    except Exception as e:
        print(f"Warning: Location lookup failed - {str(e)}")
        return {"has_location": False}  # Return safe default