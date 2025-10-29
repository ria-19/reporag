from pathlib import Path
from src.document_processor import GitHubRepoLoader, WebContentLoader 

# --- TEST 1: GitHubRepoLoader ---
def test_github():
    # 1. Setup 
    repo_path = './test_repo' 
    if not Path(repo_path).exists():
        print("ERROR: Test repo not found. Please create a small 'test_repo' folder.")
        return

    loader = GitHubRepoLoader(repo_path)
    documents = loader.load_repo()
    
    # 2. Assertions
    print("\n--- GitHub Unit Test ---")
    assert len(documents) > 0, "GitHub loader returned 0 documents."
    
    # 3. Inspection
    print(f"Loaded {len(documents)} GitHub documents.")
    print(f"Sample content start: {documents[0]['content'][:50]}...")
    print(f"Sample file path: {documents[0]['metadata'].get('file_path')}")
    assert documents[0]['metadata'].get('source') == 'github_repo', "Source tag is incorrect."

# --- TEST 2: WebContentLoader ---
def test_web():
    loader = WebContentLoader()
    urls = ['https://docs.python.org/3/library/csv.html'] 
    documents = loader.load_multiple(urls)
    
    # 2. Assertions
    print("\n--- Web Unit Test ---")
    assert len(documents) == len(urls), "Web loader failed to load all URLs."
    
    # 3. Inspection
    print(f"Loaded {len(documents)} Web documents.")
    print(f"Sample content is clean: {documents[0]['content'][:50]}...")
    print(f"Sample URL: {documents[0]['metadata'].get('url')}")
    assert documents[0]['metadata'].get('source') == 'web', "Source tag is incorrect."


if __name__ == "__main__":
    test_github()
    test_web()
    print("\n✅ All independent loader tests completed.")