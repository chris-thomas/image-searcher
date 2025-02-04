# Building a Context and Style Aware Image Search Engine: Combining CLIP, Stable Diffusion, and FAISS

This application demonstrates how to build a sophisticated image search engine that understands natural language, visual content, artistic styles, and geographical context. Users can search their image collections using plain English descriptions, find visually and stylistically similar images, and even combine location data with visual similarity. This runs completely locally without the need for cloud resources, APIs or models.

This is a rapidly build prototype as a demonstration of what is possible with rapid prototyping and iterative refinement using AI dialogue engineering tools. Note it was rapidly created to demonstrate this type of search engine and features. It is intended to be a project to be built upon.

What makes this search engine special is its dual-model approach to understanding context and style. For example, you can search for "sunset over mountains in van gogh style" and it will find relevant images that match both the content and artistic style, even if they weren't explicitly tagged with those terms. You can then adjust the search by adding more terms like "with cypress trees" or "in Switzerland" and adjust how much weight to give to each criterion and model.

Using FAISS allows the searching of huge quantities of images almost instantly, once the images have been indexed. My experiments have been on dataset collection of 10,000 images, although this would scale orders of magnitude higher.

This powerful combination of technologies creates a search engine that understands images both semantically and aesthetically. The integration of CLIP and Stable Diffusion components allows for unprecedented control over content and style-based searches, while the location awareness and multi-query capabilities provide a complete solution for sophisticated image search.

The prototype demonstrates how multiple AI models can work together to create intuitive and powerful tools for managing and exploring image collections. 

## Core Technologies

### Hybrid Neural Search (CLIP + Stable Diffusion)

My search engine leverages two powerful AI models working in tandem. We've used the `ViT-B/16` variant of CLIP and components from Stable Diffusion v1.4, offering an excellent balance between performance and resource usage.

#### CLIP Features:
* **Architecture**: Vision Transformer (ViT) with a 16x16 patch size
* **Input Resolution**: 224x224 pixels
* **Embedding Dimension**: 512
* **Model Size**: ~150M parameters
* **Performance**: 68.3% zero-shot ImageNet accuracy
* **Strengths**: General object and scene recognition

#### Stable Diffusion Components:
* **VAE**: For efficient image encoding
* **CLIP Text Encoder**: Specialized in artistic style understanding
* **Embedding Dimension**: 512 (matched to CLIP)
* **Strengths**: Artistic style and aesthetic comprehension

### FAISS (Facebook AI Similarity Search)

This models give us powerful embeddings, but searching through them efficiently requires sophisticated indexing. This implementation uses dual FAISS indices:

**Dual IndexFlatIP**: Simple but effective exact search indices
  - One for CLIP embeddings (512 dimensions)
  - One for SD embeddings (512 dimensions)
  - Optimized for cosine similarity search
  - Good performance for collections up to ~100k images
  - Full indices loaded into memory for fast searches

**Storage Features**:
  - Separate persistent storage for each model
  - Simple JSON metadata storage
  - Fast load and save operations

**Performance Features**:
  - Batch processing for efficient index updates
  - Exact similarity search (no approximation)
  - Support for incremental updates

Memory-mapped file support is available for handling large collections if needed.

### FastHTML and MonsterUI

The user interface is built using FastHTML, a Python framework for server-rendered applications. This is enhanced with MonsterUI components combining the simplicity of FastHTML with the power of TailwindCSS. This combination provides:

* Real-time search results without page refreshes
* Responsive grid layout that adapts to screen size
* Smooth loading states and transitions
* Modal image previews
* Interactive weight adjustments for both models
* Style-specific UI controls for artistic searches

The UI is specifically designed to handle the dual-model approach:

* Visual indicators for which model found each match
* Preview thumbnails that highlight artistic elements
* Intuitive sliders for balancing model influence

## Key Features

### AI-Powered Query Suggestions

One of the most powerful features is the ability to get intelligent query suggestions using a Large Language Model. The system understands both CLIP's and SD's strengths and can suggest effective search terms optimized for each model.

#### 1. Smart Query Generation

* Users can input a general description of what they're looking for
* The LLM converts this into multiple optimized search terms for both models
* Each term comes with a suggested weight and model assignment

For example, if a user inputs "I want to find impressionist paintings of people at the beach", the LLM might suggest:

* "People playing in the sand" (CLIP weight: 0.8)
* "Impressionist painting style" (SD weight: 0.9)
* "Beach activities" (CLIP weight: 0.7)
* "Loose brushwork texture" (SD weight: 0.6)
* "Sunny beach day" (CLIP weight: 0.4)

#### 2. CLIP-Aware and Style-Aware Formatting

* Suggestions are formatted to match each model's strengths
* CLIP terms focus on concrete objects and scenes
* SD terms focus on artistic styles and techniques
* Balances specific and general descriptions
* Optimizes term combinations for hybrid search

#### 3. Interactive Refinement

* Users can modify suggested weights for both models
* Add or remove suggested terms
* Combine suggestions with their own queries
* Get new suggestions based on initial results
* Fine-tune style vs. content balance

### Multi-Model Search

The hybrid search system allows users to leverage both models' strengths:

#### 1. Content Search (CLIP)
* Object recognition
* Scene understanding
* Spatial relationships
* Action recognition
* Color and composition

#### 2. Style Search (SD)
* Artistic techniques
* Visual styles
* Aesthetic qualities
* Texture recognition
* Artistic periods

#### 3. Location Awareness
* Extracts GPS coordinates from EXIF data
* Handles location-based queries
* Combines with both CLIP and SD results
* Geographic clustering of similar styles

Note this is using an LLM for the location lookup, only as its a rapidly built prototype. It might be more optimal to use an LLM to extract the place name and then pass the cleaned location to a GeoLocation API.

For example, you could build a complex query:

* "Impressionist style" (SD weight: 0.8)
* "Garden scene" (CLIP weight: 0.7)
* "In France" (Location weight: 0.5)
* "With flowers" (CLIP weight: 0.6)
* "Monet-like brushstrokes" (SD weight: 0.4)

### Similar Image Finding

The "Find Similar" feature now considers both visual content and artistic style:

#### 1. Dual Similarity Analysis
* Uses CLIP for content similarity
* Uses SD for style similarity
* Adjustable balance between the two
* Optional style-only or content-only modes

#### 2. Location Integration
* Optional geographic radius filtering
* Finds stylistically similar images from nearby locations
* Useful for finding different artistic interpretations of the same location

#### 3. Combined Scoring
* Weighted combination of CLIP and SD scores
* Location proximity factoring
* Style coherence boosting
* Customizable similarity thresholds

## References

* [FAISS article](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/){:target="_blank"}
* [FAISS repo](https://github.com/facebookresearch/faiss){:target="_blank"}
* [Stable Diffusion on Hugging Face](https://huggingface.co/spaces/stabilityai/stable-diffusion){:target="_blank"}
* [CLIP](https://openai.com/index/clip/){:target="_blank"}

P.S. Want to explore more AI insights together? Follow along with my latest work and discoveries here:

<!-- Subscribe to receive AI technical insights, news, and best practices: -->

[Subscribe to Updates](https://chris-thomas.kit.com/33b5bb9175){:target="_blank" .md-button}

[Connect with me on LinkedIn](https://uk.linkedin.com/in/christhomasuk){:target="_blank" .md-button}  

[Follow me on X (Twitter)](https://twitter.com/intent/follow?screen_name=chris_thomas_uk){:target="_blank" .md-button }
