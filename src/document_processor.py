"""
Document Loaders
Goal: Transform messy, noisy source into a clean, structured, and context-rich dataset 
    suitable for the next stages of the RAG pipeline: filtering, cleaning, and enrichment.
    
"""

from typing import List, Dict, Any
import os
from pathlib import Path
from tqdm import tqdm

# ============================================================
# 1. CODE REPOSITORY LOADER
# ============================================================

class GitHubRepoLoader:
    """
    Load and parse GitHub repository
    
    Challenges:
    - Ignore binary files, dependencies
    - Parse multiple languages
    - Extract docstrings and comments
    - Maintain file structure context
    """
    
    # Configuration 
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        
        # Files to ignore
        self.ignore_patterns = {
            '.git', 'node_modules', '__pycache__', '.venv',
            'venv', 'dist', 'build', '.next', '.cache'
        }
        
        # File extensions to process
        self.code_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java',
            '.cpp', '.c', '.go', '.rs', '.rb', '.php',
            '.md', '.txt', '.yaml', '.yml', '.json'
        }
    
    # Filter 
    def should_process(self, file_path: Path) -> bool:
        """Determine if file should be processed"""
        # Check ignore patterns
        for ignore in self.ignore_patterns:
            if ignore in file_path.parts:
                return False
        
        # Check extension
        if file_path.suffix not in self.code_extensions:
            return False
        
        # Check file size (skip huge files)
        if file_path.stat().st_size > 1_000_000:  # 1MB
            return False
        
        return True
    
    # Enrichment
    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract useful metadata from file"""
        relative_path = file_path.relative_to(self.repo_path)
        
        return {
            'source': 'github_repo',
            'file_path': str(relative_path),
            'file_name': file_path.name,
            'file_type': file_path.suffix,
            'directory': str(relative_path.parent),
            'language': self._detect_language(file_path)
        }
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from extension"""
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.md': 'markdown',
        }
        return ext_to_lang.get(file_path.suffix, 'text')
    
    def load_file(self, file_path: Path) -> Dict[str, Any]:
        """Load single file with metadata"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            return {
                'content': content,
                'metadata': self.extract_metadata(file_path)
            }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    # Orchestrator
    def load_repo(self) -> List[Dict[str, Any]]:
        """Load entire repository"""
        documents = []
        
        print(f"Loading repository: {self.repo_path}")
        
        file_iterator = self.repo_path.rglob('*') # rglob is a generator, so it yields files one by one.
        
        try:
            for file_path in tqdm(file_iterator, desc="Processing GitHub Files"):  
                if file_path.is_file() and self.should_process(file_path):
                    doc = self.load_file(file_path)
                    if doc:
                        documents.append(doc)
        except Exception as e:
            print(f"🛑 Critical error during repository traversal: {e}")
        
        print(f"✅ Loaded {len(documents)} files")
        return documents


# ============================================================
# 2. WEB CONTENT LOADER (Wikis, Docs)
# ============================================================

class WebContentLoader:
    """
    Load content from URLs (documentation, wikis)
    
    Challenges:
    - Ignore invisible code, styling, boilerplate & navigation
    - Maintain web page structure context
    
    Uses: requests + BeautifulSoup
    """
    
    def __init__(self):
        import requests
        from bs4 import BeautifulSoup
        self.requests = requests  # Lazy Importing
        self.BeautifulSoup = BeautifulSoup
    
    def load_url(self, url: str) -> Dict[str, Any]:
        """Load single URL"""
        try:
            response = self.requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = self.BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style tags
            for tag in soup(['script', 'style', 'nav', 'footer']):  # heuristic
                tag.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n', strip=True) # Preserving paragraph breaks; valuable structural information for our downstream chunker.
            
            # Extract title
            title = soup.find('title')
            title_text = title.text if title else url
            
            return {
                'content': text,
                'metadata': {
                    'source': 'web',
                    'url': url,
                    'title': title_text
                }
            }
        
        except Exception as e:
            print(f"Error loading {url}: {e}")
            return None
    
    def load_multiple(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Load multiple URLs"""
        documents = []
        for url in tqdm(urls, desc="Fetching URLs"):
            doc = self.load_url(url)
            if doc:
                documents.append(doc)
        return documents


# ============================================================
# 3. VIDEO TRANSCRIPT LOADER (YouTube)
# ============================================================

#Todo
class VideoTranscriptLoader:
    """
    Extract transcripts from YouTube videos
    
    Uses: youtube-transcript-api
    """
    
    def __init__(self):
        from youtube_transcript_api import YouTubeTranscriptApi
        self.api = YouTubeTranscriptApi
    
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        import re
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'^([0-9A-Za-z_-]{11})$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract video ID from {url}")
    
    def load_transcript(self, url: str) -> Dict[str, Any]:
        """Load transcript from YouTube video"""
        try:
            video_id = self.extract_video_id(url)
            
            # Get transcript
            transcript_list = self.api.get_transcript(video_id)
            
            # Combine all text
            text = ' '.join([entry['text'] for entry in transcript_list])
            
            return {
                'content': text,
                'metadata': {
                    'source': 'youtube',
                    'video_id': video_id,
                    'url': url
                }
            }
        
        except Exception as e:
            print(f"Error loading transcript from {url}: {e}")
            return None


# ============================================================
# 4. UNIFIED DOCUMENT PROCESSOR
# ============================================================


class DocumentProcessor: # Facade Design Pattern 
    """
    Unified interface for all document types
    """
    
    def __init__(self):    # has-a relationship (is composed of)
        self.repo_loader = GitHubRepoLoader # Class object; defer instantiation
        self.web_loader = WebContentLoader()
        # self.video_loader = VideoTranscriptLoader()
    
    def process_github_repo(self, repo_path: str) -> List[Dict]:
        """Process GitHub repository"""
        loader = self.repo_loader(repo_path)
        return loader.load_repo()
    
    def process_urls(self, urls: List[str]) -> List[Dict]:
        """Process documentation URLs"""
        return self.web_loader.load_multiple(urls)
    
    def process_videos(self, video_urls: List[str]) -> List[Dict]:
        """Process YouTube videos"""
        documents = []
        for url in video_urls:
            doc = self.video_loader.load_transcript(url)
            if doc:
                documents.append(doc)
        return documents
    
    def process_all(self, config: Dict) -> List[Dict]: # for building data-drive system, ingested via configurtion file
        """Process all sources from config"""
        all_documents = []
        
        # GitHub repos
        if 'github_repos' in config:
            print("\n==============================================")
            print("🚀 Starting: GitHub Repository Ingestion")
            print("==============================================")
            
            for repo_path in config['github_repos']:
                try:
                    docs = self.process_github_repo(repo_path)
                    all_documents.extend(docs)
                except Exception as e:
                    print(f"❌ Failed to process GitHub repo at {repo_path}. Error: {e}")        
        # URLs
        if 'urls' in config:
            print("\n==============================================")
            print("🌐 Starting: Web Content (URLs) Fetch")
            print("==============================================")
            
            try:
                docs = self.process_urls(config['urls'])
                all_documents.extend(docs)
            except Exception as e:
                print(f"❌ Failed to process the list of URLs. Error: {e}")        
        
        # --- 3. Videos (Future Implementation) ---
        if 'videos' in config:
            print("\n==============================================")
            print("🎬 Starting: Video Transcript Loading")
            print("==============================================")
            
            try:
                # When implemented, this will trigger a tqdm bar
                # docs = self.process_videos(config['videos']) 
                # all_documents.extend(docs)
                print("🚧 Video Transcript Loader is pending implementation.")
            except Exception as e:
                print(f"❌ Failed to process the list of videos. Error: {e}")
        
        print(f"\n✅ Total documents loaded: {len(all_documents)}")
        return all_documents


# ============================================================
# TESTING
# ============================================================

def test_loaders():
    """Test all loaders"""
    
    # Test configuration
    config = {
        'github_repos': ['./test_repo'], 
        'urls': [
            'https://docs.python.org/3/tutorial/index.html'
        ]
    }
    
    processor = DocumentProcessor()
    documents = processor.process_all(config)
    
    # Analyze results
    print("\nDocument Analysis:")
    sources = {}
    for doc in documents:
        source = doc['metadata']['source']
        sources[source] = sources.get(source, 0) + 1
    
    for source, count in sources.items():
        print(f"  {source}: {count} documents")

if __name__ == "__main__":
    test_loaders()
   
   
# ============================================================
# TODO: Future Enhancements and Scalability Planning
# ============================================================

# --- 1. CORE SCALABILITY & PERFORMANCE (High Priority) ---

# TODO: OPTIMIZATION 1.1: STREAMING (MEMORY SAFETY)
"""
What: Redesign `load_repo` (and similar methods) as a **generator** that yields one document at a time, instead of reading all contents into a single in-memory list.
Why: Prevents **Out-of-Memory (OOM) errors** when processing extremely large repositories (e.g., millions of files). Allows **streaming** processing with a constant, low memory footprint, making ingestion memory-safe.
When: Implement before integrating large organizational or open-source repositories (e.g., Linux kernel).
"""

# TODO: OPTIMIZATION 1.2: PARALLELISM (INGESTION THROUGHPUT)
"""
What: Introduce **parallel processing** to the file reading and content fetching stages.
      - Use `ThreadPoolExecutor` for **I/O-bound** parallelism (e.g., reading local files, fetching web content).
      - Use `ProcessPoolExecutor` for **CPU-bound** parallelism (e.g., complex content cleaning or symbol extraction).
Why: Utilizes all available CPU cores, **drastically reducing total ingestion time** for large datasets (100k+ files or URLs) and maximizing throughput.
When: Implement when optimizing ingestion throughput or scaling to enterprise-level datasets.
"""

# TODO: OPTIMIZATION 1.3: INCREMENTAL PROCESSING (STATE MANAGEMENT)
"""
What: Implement a **persistence layer** (e.g., SQLite, cache file) to store the **last-processed timestamp** for every source item (file path, URL). Loaders will only re-process an item if its current modification time is newer than the stored timestamp.
Why: Enables **incremental updates**, transforming the system from stateless to stateful. This prevents expensive, time-consuming reprocessing of unchanged data, saving significant resources (API quotas, CPU time).
When: **High Priority.** Implement before the system ingests a large volume of data (thousands of items) or when processing time becomes a bottleneck.
"""

# --- 2. RESILIENCE & CODE QUALITY ---

# TODO: QUALITY 2.1: ROBUST ERROR ISOLATION
"""
What: In the `process_all` method:
      1. Wrap independent source processing blocks (GitHub, URLs, Videos) in separate **`try...except`** blocks.
      2. Collect specific failure messages in an `errors` list.
      3. Change the method return signature to `(successful_documents, errors)`.
Why: Implements **fault tolerance and resilience**. A non-critical failure in one source (e.g., invalid GitHub path) will not cause a **cascading failure** that halts the entire pipeline. Provides the caller with a clear success/failure report.
When: In v1 improvement (as this is fundamental for reliability).
"""

# TODO: QUALITY 2.2: REFACTOR FOR DEPENDENCY INJECTION (DI)
"""
What: Modify the `DocumentProcessor`'s `__init__` method to **accept loader instances as arguments** (e.g., `__init__(self, repo_loader, web_loader)`), instead of creating them internally.
Why: **Decouples** the `DocumentProcessor` from specific loader implementations, allowing for flexible swapping of implementations (e.g., using `MockLoaders` for testing) and improving testability.
When: Once unit tests or alternative loader implementations (e.g., mock, async) are introduced.
"""

# --- 3. DATA QUALITY & EXTENSIBILITY ---

# TODO: DATA QUALITY 3.1: INTELLIGENT HTML CLEANING
"""
What: Enhance `WebContentLoader`'s noise removal.
      - Expand cleaning to use precise **CSS selectors** (e.g., `.sidebar`, `#menu`) to eliminate page junk.
      - **OR** switch to a dedicated library (e.g., `trafilatura`, `goose3`) for intelligent main content extraction.
Why: Improves **resilience against diverse HTML structures** and ensures only high-quality, relevant text is extracted, which is crucial for downstream retrieval accuracy.
When: Implement before scaling ingestion across heterogeneous documentation sources.
"""

# TODO: EXTENSIBILITY 3.2: SYMBOL EXTRACTION (PYTHON)
"""
What: For Python files (`.py`), parse the source using the built-in `ast` module to extract metadata for symbols (classes, functions, methods, docstrings, etc.).
How: Use `ast.parse(text)` and a `NodeVisitor`. Attach this structured metadata to the file-level document OR create separate "symbol documents" for embedding.
Why: Enables **higher-precision retrieval, filtering, and ranking** by allowing search at the specific symbol level, crucial for technical documentation and codebase indexing.
When: Implement before indexing large Python repositories or enabling symbol-aware search functionality.
"""

# ============================================================
# METRICS FOR EVALUATION
# ============================================================

# METRIC: NOISE REDUCTION RATIO
# Formula: (Number of files/sections ignored by filters) / (Total files/sections encountered in source)
# Goal: Track the effectiveness of filters/cleaning in reducing irrelevant content.

# METRIC: DOWNSTREAM RETRIEVAL HIT RATE
# Measure: Process a "golden dataset" using the pipeline, index the results, and test the resulting vector store against a set of known queries.
# Goal: Directly measure the business value/accuracy of the ingested data for RAG (Retrieval-Augmented Generation) or search.

# METRIC: THROUGHPUT
# Formula: Total documents processed / Time taken (e.g., documents per second)
# Goal: Track the raw speed of the ingestion pipeline. Directly tied to Parallelism (1.2) and Streaming (1.1) fixes.

# METRIC: ERROR RATE
# Formula: Number of failed sources / Total sources attempted
# Goal: Track the stability of the system. Directly tied to Error Isolation (2.1) fix.
