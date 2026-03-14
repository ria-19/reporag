# src/ingestion/loaders.py
"""
LoaderPort implementations.

Loaders are the write path's interface to the world.
They yield RawFile objects — the indexer never knows
where the files came from (local disk, GitHub, zip).

WHY streaming (yield) not batch (return list)?
A repo can have 10,000 files. Loading all into memory
before indexing starts = unnecessary RAM spike.
Streaming: parse file N while file N-1 is being embedded.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

from src.core.models import RawFile, Language
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

# Files and directories to skip
IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", ".mypy_cache", ".pytest_cache", "eggs",
    "test", "tests", "test_*",
}
SUPPORTED_EXTENSIONS = {
    ".py": Language.PYTHON,
    ".js": Language.JAVASCRIPT,
    ".ts": Language.TYPESCRIPT,
}
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024   # 1MB — skip generated/minified files


class LocalRepoLoader:
    """
    Walks a local directory, yields RawFile per source file.
    Used for: local repos, already-cloned paths, tests.
    """

    def stream_files(self, repo_path: str, repo_name: str) -> Iterator[RawFile]:
        """
        Walk repo_path recursively.
        Skip: ignored dirs, unsupported extensions, files > 1MB.
        Yield: one RawFile per valid source file.
        """
        root = Path(repo_path)
        files_yielded = 0
        files_skipped = 0

        for dirpath, dirnames, filenames in os.walk(root):
            # Prune ignored dirs IN PLACE
            # WHY in place? os.walk uses dirnames to decide
            # which subdirs to recurse into. Modifying in place
            # prevents os.walk from descending into them.
            dirnames[:] = [
                d for d in dirnames
                if d not in IGNORE_DIRS and not d.startswith(".")
            ]

            for filename in filenames:
                full_path = Path(dirpath) / filename
                rel_path = os.path.relpath(full_path, repo_path)

                if not full_path.is_file(): # symlinks/special files skip
                    continue
                
                ext = full_path.suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    files_skipped += 1
                    continue

                size = os.path.getsize(full_path)
                if size > MAX_FILE_SIZE_BYTES or size == 0:
                    files_skipped += 1
                    continue
                
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    
                    yield RawFile(
                        repo_name=repo_name,
                        path=rel_path,
                        content=content,
                        language=SUPPORTED_EXTENSIONS[ext],
                        size_bytes=size
                    )
                    files_yielded += 1

                except Exception as e:
                    logger.warning(f"Skipping {rel_path}: {e}")


        logger.info("LocalRepoLoader: %d files yielded, %d skipped", files_yielded, files_skipped)


class GitHubRepoLoader:
    """
    Clones a GitHub repo to a temp dir, then delegates to LocalRepoLoader.

    WHY temp dir? We don't want to manage clone lifecycle.
    tempfile.TemporaryDirectory() cleans up on __exit__.
    Repo is deleted after indexing — we keep only the chunks.
    """

    def stream_files(self, github_url: str, repo_name: str) -> Iterator[RawFile]:
        """
        Clone repo, stream files, cleanup on done.
        """
        with tempfile.TemporaryDirectory() as tmpdir:   
            logger.info(f"Cloning {github_url} into temporary directory: {tmpdir}")
            env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

            try:
                subprocess.run(
                    [
                        "git", "clone", 
                        "--depth=1", 
                        "--single-branch",
                        "-c", "advice.detachedHead=false",
                        github_url, 
                        tmpdir
                    ],
                    check=True,
                    # capture_output=True,
                    text=True,
                    env=env,
                    timeout=60 # 1-minute timeout protection
                )
            except subprocess.TimeoutExpired:
                logger.error(f"Clone timed out for {github_url}")
                raise ValueError("Repository is too large or connection is too slow.")
            except subprocess.CalledProcessError as e:
                # We strip the error to avoid leaking system paths in the API response
                err_msg = e.stderr.splitlines()[-1] if e.stderr else "Unknown error"
                logger.error(f"Git clone failed: {err_msg}")
                raise ValueError(f"Git error: {err_msg}")

            cloned_files = list(Path(tmpdir).iterdir())
            if not cloned_files:
                raise ValueError(f"Clone succeeded but directory is empty: {github_url}")

            logger.info("Cloning %s → %s", github_url, tmpdir)
            yield from LocalRepoLoader().stream_files(tmpdir, repo_name)
            
            
        
        

