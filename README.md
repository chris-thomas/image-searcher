# Building a Context and Style Aware Image Search Engine: Combining CLIP, Stable Diffusion, and FAISS

This application demonstrates how to build a sophisticated image search engine that understands natural language, visual content, artistic styles, and geographical context. Users can search their image collections using plain English descriptions, find visually and stylistically similar images, and even combine location data with visual similarity. This runs completely locally without the need for cloud resources, APIs or models.

This is a rapidly build prototype as a demonstration of what is possible with rapid prototyping and iterative refinement using AI dialogue engineering tools. Note it was rapidly created to demonstrate this type of search engine and features. It is intended to be a project to be built upon.

What makes this search engine special is its dual-model approach to understanding context and style. For example, you can search for "sunset over mountains in van gogh style" and it will find relevant images that match both the content and artistic style, even if they weren't explicitly tagged with those terms. You can then adjust the search by adding more terms like "with cypress trees" or "in Switzerland" and adjust how much weight to give to each criterion and model.

Using FAISS allows the searching of huge quantities of images almost instantly, once the images have been indexed. My experiments have been on dataset collection of 10,000 images, although this would scale orders of magnitude higher.

This powerful combination of technologies creates a search engine that understands images both semantically and aesthetically. The integration of CLIP and Stable Diffusion components allows for unprecedented control over content and style-based searches, while the location awareness and multi-query capabilities provide a complete solution for sophisticated image search.

The prototype demonstrates how multiple AI models can work together to create intuitive and powerful tools for managing and exploring image collections. 

### Hybrid Neural Search (CLIP + Stable Diffusion)

This search engine leverages two powerful AI models working in tandem. It uses the `ViT-B/16` variant of CLIP and components from Stable Diffusion v1.4, offering an excellent balance between performance and resource usage.

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

When you search for "sunset over mountains in van gogh style":

1. CLIP converts the content aspects ("sunset over mountains") into a high-dimensional vector
2. SD's text encoder processes the style aspects ("van gogh style")
3. Both image and text are projected into their respective 512-dimensional embedding spaces
4. The system combines scores from both models with configurable weights
5. Results reflect both content relevance and style matching

This combination was chosen because:

* It is fast enough for real-time search
* Small enough to run on consumer GPUs
* CLIP excels at understanding concrete concepts
* Stable Diffusion components excel at artistic style comprehension
* Efficient memory usage when processing batches

The real magic happens when combining multiple searches - we can mathematically combine these vectors with different weights to find images that match multiple criteria simultaneously.

### FAISS (Facebook AI Similarity Search)

These models provide powerful embeddings, although searching through them efficiently requires sophisticated indexing. This implementation uses dual FAISS indices:

**Dual IndexFlatIP**: Simple but effective exact search indices

  - One for CLIP embeddings (512 dimensions)
  - One for Stable Diffusion embeddings (512 dimensions)
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
* "Impressionist painting style" (Stable Diffusion weight: 0.9)
* "Beach activities" (CLIP weight: 0.7)
* "Loose brushwork texture" (Stable Diffusion weight: 0.6)
* "Sunny beach day" (CLIP weight: 0.4)

#### 2. CLIP-Aware and Style-Aware Formatting

* Suggestions are formatted to match each model's strengths
* CLIP terms focus on concrete objects and scenes
* Stable Diffusion terms focus on artistic styles and techniques
* Balances specific and general descriptions
* Optimizes term combinations for hybrid search

#### 3. Interactive Refinement

* Users can modify suggested weights
* Add or remove suggested terms
* Combine suggestions with their own queries

### Multi-Model Search

The hybrid search system allows users to leverage both models' strengths:

#### 1. Content Search (CLIP)
* Object recognition
* Scene understanding
* Spatial relationships
* Action recognition
* Color and composition

#### 2. Style Search (Stable Diffusion)
* Artistic techniques
* Visual styles
* Aesthetic qualities
* Texture recognition
* Artistic periods

#### 3. Location Awareness
* Extracts GPS coordinates from EXIF data
* Handles location-based queries
* Combines with both CLIP and Stable Diffusion results
* Geographic clustering of similar styles

For example, you could build a complex query:

* "Impressionist style" (Stable Diffusion weight: 0.8)
* "Garden scene" (CLIP weight: 0.7)
* "In France" (Location weight: 0.5)
* "With flowers" (CLIP weight: 0.6)
* "Monet-like brushstrokes" (Stable Diffusion weight: 0.4)

### Similar Image Finding

The "Find Similar" feature now considers both visual content and artistic style:

#### 1. Dual Similarity Analysis
* Uses CLIP for content similarity
* Uses Stable Diffusion for style similarity
* Optional content-only mode

#### 2. Location Integration
* Optional geographic distance filtering
* Finds stylistically similar images from nearby locations
* Useful for finding different artistic interpretations of the same location

#### 3. Combined Scoring
* Weighted combination of CLIP and Stable Diffusion scores
* Location proximity factoring
* Customizable similarity thresholds

## References

* [FastHTML](https://www.fastht.ml/)
* [MonsterUI](https://github.com/AnswerDotAI/MonsterUI)
* [FAISS article](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
* [FAISS repo](https://github.com/facebookresearch/faiss)
* [Stable Diffusion on Hugging Face](https://huggingface.co/spaces/stabilityai/stable-diffusion)
* [CLIP](https://openai.com/index/clip/)

P.S. Want to explore more AI insights together? Follow along with my latest work and discoveries here:

<!-- Subscribe to receive AI technical insights, news, and best practices: -->

[Subscribe to Updates](https://chris-thomas.kit.com/33b5bb9175)

[Connect with me on LinkedIn](https://uk.linkedin.com/in/christhomasuk) 

[Follow me on X (Twitter)](https://twitter.com/intent/follow?screen_name=chris_thomas_uk)
