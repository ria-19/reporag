"""
Pass 1: RawFile → List[CodeChunk] + List[RawEdge]

This is the most complex file in the write path.
It does two things:
  1. Parse code into logical units (functions, classes, module-level blocks)
  2. Extract raw dependency names (calls_out_raw, imports_raw)

It does NOT:
  - Resolve symbol names to chunk_ids (that's graph.py, Pass 2)
  - Embed chunks (that's embedding/embedder.py)
  - Write to storage (that's indexer.py)

Strategy pattern: parser.py picks the right strategy per language.
Adding a new language = one new strategy file + one line in REGISTRY.
This file never changes when you add a language.

Tree-sitter version: >= 0.22
Grammars: tree-sitter-python, tree-sitter-javascript (pip install)
WHY new API: binary wheels, no runtime compilation, no C toolchain.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Generator

# tree-sitter >= 0.22 imports
# Each grammar is a separate pip package that ships pre-compiled
from tree_sitter import Language, Parser, Node, Query, QueryCursor
import tree_sitter_python
import tree_sitter_javascript

from src.core.models import (
    RawFile, CodeChunk, RawEdge,
    ChunkType, EdgeType, Language as Lang,
    make_chunk_id,
)
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════
# LANGUAGE OBJECTS
# Built once at module load, reused for every parse.
# WHY module-level? Parser construction is expensive.
# Building it per-file would add ~50ms per file.
# ════════════════════════════════════════════════════════

# tree-sitter >= 0.22: Language() takes a single pointer
# from the grammar's .language() function
PY_LANGUAGE = Language(tree_sitter_python.language())
JS_LANGUAGE = Language(tree_sitter_javascript.language())


# ════════════════════════════════════════════════════════
# STRATEGY INTERFACE
# Every language strategy must implement this contract.
# Parser.parse() calls this interface — never the concrete class.
# ════════════════════════════════════════════════════════

class LanguageStrategy(ABC):
    """
    Abstract base for language-specific parsing.

    Contract:
      parse() receives a RawFile and repo_name.
      Returns (chunks, raw_edges) — Pass 1 output only.
      No resolution, no embedding, no storage.
    """

    @abstractmethod
    def parse(
        self,
        raw_file: RawFile,
        repo_name: str,
    ) -> tuple[list[CodeChunk], list[RawEdge]]:
        """
        Parse one file into chunks and unresolved edges.

        Args:
            raw_file:  the file to parse
            repo_name: used in chunk_id construction

        Returns:
            chunks:    one per function/class/module-level block
            raw_edges: unresolved calls and imports
        """
        ...


# ════════════════════════════════════════════════════════
# PYTHON STRATEGY
# Uses tree-sitter for reliable AST parsing.
# Handles: functions, methods, classes, module-level code.
# ════════════════════════════════════════════════════════

PYTHON_BUILTINS = {
    "len", "print", "range", "isinstance", "type", "str", "int",
    "list", "dict", "set", "tuple", "bool", "float", "enumerate",
    "zip", "map", "filter", "sorted", "reversed", "open", "super",
    "hasattr", "getattr", "setattr", "any", "all", "max", "min",
}

class PythonStrategy(LanguageStrategy):
    """
    Python parser using tree-sitter.

    Node types we care about (from tree-sitter-python grammar):
      function_definition  → functions and methods
      class_definition     → classes
      decorated_definition → @decorator + function/class

    WHY tree-sitter over Python's stdlib ast module?
      stdlib ast: Python-only, no JS, no Java.
      tree-sitter: same API for every language.
      We want one mental model across all strategies.
    """

    def __init__(self):
        self._parser = Parser(PY_LANGUAGE)

    def parse(
        self,
        raw_file: RawFile,
        repo_name: str,
    ) -> tuple[list[CodeChunk], list[RawEdge]]:

        # tree-sitter works on bytes, not strings
        # WHY bytes? tree-sitter was designed for editors
        # that work with byte buffers, not Python strings.
        source_bytes = raw_file.content.encode("utf-8")
        tree = self._parser.parse(source_bytes)

        chunks: list[CodeChunk] = []
        raw_edges: list[RawEdge] = []

        # Walk the AST, extract top-level nodes we care about

        self._walk(
            node=tree.root_node,
            source_bytes=source_bytes,
            raw_file=raw_file,
            repo_name=repo_name,
            chunks=chunks,
            raw_edges=raw_edges,
            parent_class=None,   # top level: no parent class yet
        )

        # Extract import edges for this file
        import_edges = self._extract_imports(
            tree.root_node, source_bytes, raw_file, repo_name
        )
        raw_edges.extend(import_edges)

        return chunks, raw_edges

    def _walk(
        self,
        node: Node,
        source_bytes: bytes,
        raw_file: RawFile,
        repo_name: str,
        chunks: list[CodeChunk],
        raw_edges: list[RawEdge],
        parent_class: str | None,
    ) -> None:
        """
        Recursive AST walk.

        parent_class tracks whether we are inside a class body.
        WHY recursive? The AST is a tree. Classes contain methods.
        Methods contain nested functions (rare but valid Python).
        We want all of them.

        WHY pass parent_class down?
        When we find a function_definition node, we need to know
        if we're inside a class_definition to set chunk_type correctly:
          parent_class=None  → ChunkType.FUNCTION
          parent_class="Foo" → ChunkType.FUNCTION (method lives inside Foo)
        """
        for child in node.children:

            if child.type in ("function_definition", "async_function_definition"):
                chunk, edges = self._handle_function(
                    child, source_bytes, raw_file, repo_name, parent_class
                )
                if chunk:
                    chunks.append(chunk)
                    raw_edges.extend(edges)

            elif child.type == "class_definition":
                class_chunk, class_edges = self._handle_class(
                    child, source_bytes, raw_file, repo_name
                )
                if class_chunk:
                    chunks.append(class_chunk)
                    raw_edges.extend(class_edges)

                # Recurse into class body to find methods
                # WHY recurse here and not in _handle_class?
                # _handle_class creates the class header chunk.
                # Method chunks are separate — different chunk_ids.
                # Keeping them separate means we can retrieve a method
                # without retrieving the whole class.
                class_name = self._get_node_name(child, source_bytes)
                body = child.child_by_field_name("body")
                if body:
                    self._walk(
                        body, source_bytes, raw_file, repo_name,
                        chunks, raw_edges,
                        parent_class=class_name,   # ← methods will see this
                    )

            elif child.type == "decorated_definition":
                # @decorator wraps a function or class
                # The actual function/class is the last child
                inner = child.children[-1]
                if inner.type in ("function_definition", "async_function_definition"):
                    chunk, edges = self._handle_function(
                        inner, source_bytes, raw_file, repo_name, parent_class,
                        decorator_node=child,
                    )
                    if chunk:
                        chunks.append(chunk)
                        raw_edges.extend(edges)
                elif inner.type == "class_definition":
                    class_chunk, class_edges = self._handle_class(
                        inner, source_bytes, raw_file, repo_name
                    )
                    if class_chunk:
                        chunks.append(class_chunk)
                        raw_edges.extend(class_edges)
                    class_name = self._get_node_name(inner, source_bytes)
                    body = inner.child_by_field_name("body")
                    if body:
                        self._walk(
                            body, source_bytes, raw_file, repo_name,
                            chunks, raw_edges,
                            parent_class=class_name,
                        )

            # TODO: Module-level code (assignments, expressions not inside defs)
            # Skipping these for now — add MODULE chunk type later if needed
            # collect module-level assignments as MODULE chunks

    def _handle_function(
        self,
        node: Node,
        source_bytes: bytes,
        raw_file: RawFile,
        repo_name: str,
        parent_class: str | None,
        decorator_node: Node | None = None,
    ) -> tuple[CodeChunk | None, list[RawEdge]]:
        """
        Build a CodeChunk for one function/method node.
        """
        func_name = self._get_node_name(node, source_bytes)

        params_node = node.child_by_field_name("parameters")
        signature_str = self._get_node_text(params_node, source_bytes) if params_node else "()"

        start_node = decorator_node if decorator_node else node
        start_line = start_node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        full_text = self._get_node_text(start_node, source_bytes)

        body_node = node.child_by_field_name("body")
        docstring = self._extract_docstring(body_node, source_bytes)
        calls_out_raw = self._extract_calls(body_node, source_bytes)

        chunk = CodeChunk(
            repo_name=repo_name,
            chunk_id = make_chunk_id(repo_name, raw_file.path, parent_class, func_name, signature_str),
            symbol_name=func_name,
            chunk_type=ChunkType.FUNCTION,
            language=raw_file.language,   
            file_path=raw_file.path,
            text=full_text,
            docstring=docstring,
            start_line=start_line,
            end_line=end_line,
            calls_out_raw=calls_out_raw,    
            parent_class=parent_class,      
        )

        edges = [
            RawEdge(source_id=chunk.chunk_id, target_name=call, edge_type=EdgeType.CALLS)
            for call in calls_out_raw
        ]  

        return chunk, edges

    def _handle_class(
        self,
        node: Node,
        source_bytes: bytes,
        raw_file: RawFile,
        repo_name: str,
    ) -> tuple[CodeChunk | None, list[RawEdge]]:
        """
        Build a CodeChunk for the class HEADER only.
        Methods are handled separately in _walk via recursion.

        Class header includes:
          - class name
          - base classes (for inheritance context)
          - class-level docstring
          - class-level variable assignments (not method bodies)

        """
        class_name = self._get_node_name(node, source_bytes)  
        start_line = node.start_point[0] + 1

        body_node = node.child_by_field_name("body")
        docstring = self._extract_docstring(body_node, source_bytes)

        if body_node:
            header_text = source_bytes[node.start_byte:body_node.start_byte].decode("utf-8").strip()
            end_line = body_node.start_point[0] + 1
        else:
            header_text = self._get_node_text(node, source_bytes)
            end_line = node.end_point[0] + 1


        chunk = CodeChunk(
            repo_name=repo_name,
            chunk_id=make_chunk_id(repo_name, raw_file.path, None, class_name, ""),
            symbol_name=class_name,
            chunk_type=ChunkType.CLASS,
            language=raw_file.language,
            file_path=raw_file.path,
            text=header_text,
            docstring=docstring,
            start_line=start_line,
            end_line=end_line,
            parent_class=None  # nested classes not handled — future work
        )

        edges = [] # Note extracting base classes as of now

        return chunk, edges
    
    def _extract_imports(
        self,
        root: Node,
        source_bytes: bytes,
        raw_file: RawFile,
        repo_name: str,
    ) -> list[RawEdge]:
        """
        Extract import statements as RawEdge(IMPORTS).

        Node types:
          import_statement      → import os, import numpy as np
          import_from_statement → from src.auth import verify

        """
        query_str = """
        (import_statement) @import
        (import_from_statement) @import.from
        """

        query = PY_LANGUAGE.query(query_str)
        cursor = QueryCursor(query)
        matches = cursor.matches(root)

        import_edges = []
        
        # Instead of dangling MODULE chunk_id (becz we havent implement module chunk yet!), using file path as source
        # Simple, always exists, unambiguous
        source_id = f"{repo_name}::{raw_file.path}::imports"

        # This is not a real chunk_id — it's a file-level identifier
        # for import edges only. Kuzu stores it, we never retrieve it
        # as a chunk. Clean, no dangling references.

        for match in matches:
            for _, nodes in match[1].items():
                for node in nodes:
                    import_text = self._get_node_text(node, source_bytes).strip("\"'")

                    import_edges.append(
                        RawEdge(
                            source_id=source_id, 
                            target_name=import_text,
                            edge_type=EdgeType.IMPORTS
                        )
                    )

        return import_edges

    def _get_node_name(self, node: Node, source_bytes: bytes) -> str:
        """Get the name field of a named node as a string."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return "unknown"
        return source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")

    def _get_node_text(self, node: Node, source_bytes: bytes) -> str:
        """Get full source text of any node."""
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def _extract_docstring(self, body_node: Node, source_bytes: bytes) -> str | None:
        """
        Python docstring = first statement in body if it's a string literal.

        In tree-sitter: look for expression_statement containing a string
        as the first child of the body block.
        """
        if not body_node or body_node.type != "block":
            return None
        
        if len(body_node.children) == 0:
            return None
        
        first_stmt = body_node.children[0]

        if first_stmt.type == "expression_statement":
            if first_stmt.children and first_stmt.children[0].type == "string":
                return self._get_node_text(first_stmt.children[0], source_bytes)

        return None

    def _extract_calls(self, body_node: Node, source_bytes: bytes) -> list[str]:
        """
        Find all function calls inside a body node.
        Returns raw symbol names: ["fetch_user", "hash_password", "os.path.join"]

        WHY raw names and not chunk_ids?
        At parse time we don't know which file fetch_user lives in.
        graph.py resolves names → chunk_ids in Pass 2.

        Only attribute and identifier calls
        """
        
        calls = set()

        def walk_for_calls(n: Node):
            if n.type == "call":
                func_node = n.child_by_field_name("function")

                if func_node:
                    if func_node.type in ("attribute", "identifier"):
                        raw_name = self._get_node_text(func_node, source_bytes)
                        base_name = raw_name.split(".")[0]   # "os.path.join" → "os"
                        
                        if base_name not in PYTHON_BUILTINS:
                            calls.add(base_name)
            
            for child in n.children:
                walk_for_calls(child)

        if body_node:
            walk_for_calls(body_node)

        return list(calls)


# ════════════════════════════════════════════════════════
# JAVASCRIPT/TYPESCRIPT STRATEGY
# Same interface, tree-sitter-javascript grammar
# ════════════════════════════════════════════════════════

JAVASCRIPT_BUILTINS = {
    # 1. Core Global Objects (Namespaces & Constructors)
    "console", "Math", "JSON", "Object", "Array", "String", "Number", 
    "Boolean", "Date", "RegExp", "Map", "Set", "WeakMap", "WeakSet", 
    "Promise", "Error", "Symbol", "Proxy", "Reflect", "Intl",

    # 2. Core Global Functions
    "parseInt", "parseFloat", "isNaN", "isFinite", "decodeURI", 
    "decodeURIComponent", "encodeURI", "encodeURIComponent", "eval", 

    # 3. Timers
    "setTimeout", "setInterval", "clearTimeout", "clearInterval",

    # 4. Common Browser / Web APIs
    "window", "document", "fetch", "alert", "prompt", "confirm", 
    "navigator", "location", "localStorage", "sessionStorage",

    # 5. Common Node.js Globals
    "process", "Buffer", "require", "module", "exports", "global",

    # 6. Keywords that act like functions
    "super",

    # 7. current object - not a resolable object
    "this", "self"
}

class JavaScriptStrategy(LanguageStrategy):
    """
    JavaScript/TypeScript parser using tree-sitter.

    Node types (from tree-sitter-javascript grammar):
      function_declaration     → function foo() {}
      arrow_function           → const foo = () => {}
      method_definition        → class method
      class_declaration        → class Foo {}

    """

    def __init__(self):
        self._parser = Parser(JS_LANGUAGE)

    def parse(
        self,
        raw_file: RawFile,
        repo_name: str,
    ) -> tuple[list[CodeChunk], list[RawEdge]]:
        
        source_bytes = raw_file.content.encode("utf-8")
        tree = self._parser.parse(source_bytes)

        chunks: list[CodeChunk] = []
        raw_edges: list[RawEdge] = []

        self._walk(
            node=tree.root_node,
            source_bytes=source_bytes,
            raw_file=raw_file,
            repo_name=repo_name,
            chunks=chunks,
            raw_edges=raw_edges,
            parent_class=None,   # top level: no parent class yet
        )

        # Extract import edges for this file (ES6 Imports + CommonJS Requires)
        import_edges = self._extract_imports(
            tree.root_node, source_bytes, raw_file, repo_name
        )
        raw_edges.extend(import_edges)

        return chunks, raw_edges


    def _walk(
        self,
        node: Node,
        source_bytes: bytes,
        raw_file: RawFile,
        repo_name: str,
        chunks: list[CodeChunk],
        raw_edges: list[RawEdge],
        parent_class: str | None,
    ) -> None:

        for child in node.children:

            # JS Function and Class Methods
            if child.type in ("function_declaration", "generator_function_declaration", "method_definition"):
                chunk, edges = self._handle_function(
                    child, source_bytes, raw_file, repo_name, parent_class
                )
                if chunk:
                    chunks.append(chunk)
                    raw_edges.extend(edges)   

            elif child.type == "class_declaration":
                chunk, edges = self._handle_class(
                    child, source_bytes, raw_file, repo_name, parent_class
                )
                if chunk:
                    chunks.append(chunk)
                    raw_edges.extend(edges)  

                class_name = self._get_node_text(child, source_bytes)
                body = child.child_by_field_name("body")
                if body:
                    self._walk(
                        body, source_bytes, raw_file, repo_name,
                        chunks, raw_edges,
                        parent_class=class_name,
                    )      

    def _handle_function(
        self,
        node: Node,
        source_bytes: bytes,
        raw_file: RawFile,
        repo_name: str,
        parent_class: str | None,
    ) -> tuple[CodeChunk | None, list[RawEdge]]:
        
        func_name = self._get_node_name(node, source_bytes)

        params_node = node.child_by_field_name("parameters")
        signature_str = self._get_node_text(params_node, source_bytes) if params_node else "()"

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        full_text = self._get_node_text(node, source_bytes)

        docstring = self._extract_jsdoc(node, source_bytes) # JSDoc is outside the body in JS!

        body_node = node.child_by_field_name("body")
        calls_out_raw = self._extract_calls(body_node, source_bytes)

        chunk = CodeChunk(
            repo_name=repo_name,
            chunk_id = make_chunk_id(repo_name, raw_file.path, parent_class, func_name, signature_str),
            symbol_name=func_name,
            chunk_type=ChunkType.FUNCTION,
            language=raw_file.language,   
            file_path=raw_file.path,
            text=full_text,
            docstring=docstring,
            start_line=start_line,
            end_line=end_line,
            calls_out_raw=calls_out_raw,    
            parent_class=parent_class,      
        )

        edges = [
            RawEdge(source_id=chunk.chunk_id, target_name=call, edge_type=EdgeType.CALLS)
            for call in calls_out_raw
        ]  

        return chunk, edges

    def _handle_class(
        self,
        node: Node,
        source_bytes: bytes,
        raw_file: RawFile,
        repo_name: str,
        parent_class: str | None = None
    ) -> tuple[CodeChunk | None, list[RawEdge]]:

        class_name = self._get_node_name(node, source_bytes)  
        start_line = node.start_point[0] + 1

        body_node = node.child_by_field_name("body")
        docstring = self._extract_jsdoc(node, source_bytes)

        if body_node:
            header_text = source_bytes[node.start_byte:body_node.start_byte].decode("utf-8").strip()
            end_line = body_node.start_point[0] + 1
        else:
            header_text = self._get_node_text(node, source_bytes)
            end_line = node.end_point[0] + 1


        chunk = CodeChunk(
            repo_name=repo_name,
            chunk_id=make_chunk_id(repo_name, raw_file.path, None, class_name, ""),
            symbol_name=class_name,
            chunk_type=ChunkType.CLASS,
            language=raw_file.language,
            file_path=raw_file.path,
            text=header_text,
            docstring=docstring,
            start_line=start_line,
            end_line=end_line
        )

        edges = [] # Note extracting base classes as of now

        return chunk, edges
    
    def _extract_imports(
        self,
        root: Node,
        source_bytes: bytes,
        raw_file: RawFile,
        repo_name: str,
    ) -> list[RawEdge]:
        """
        Catches ES6 `import ... from 'x'` AND CommonJS `require('x')`.
        """

        query_str = """
        ;; ES6 Imports
        (import_statement source: (string) @import.source)
        
        ;; CommonJS Requires (e.g., const fs = require('fs'))
        (call_expression 
            function: (identifier) @func (#eq? @func "require")
            arguments: (arguments (string) @require.source))
        """
        query = JS_LANGUAGE.query(query_str)
        cursor = QueryCursor(query)
        matches = cursor.matches(root)
        
        import_edges = []
        # Instead of dangling MODULE chunk_id (becz we havent implement module chunk yet!), using file path as source
        # Simple, always exists, unambiguous
        source_id = f"{repo_name}::{raw_file.path}::imports" 
        
        for match in matches:
            for capture_name, nodes in match[1].items():
                if capture_name in ("import.source", "require.source"):
                    for node in nodes:
                        raw_target = self._get_node_text(node, source_bytes).strip("\"'")
                        
                        import_edges.append(
                            RawEdge(
                                source_id=source_id, 
                                target_name=raw_target, 
                                edge_type=EdgeType.IMPORTS
                            )
                        )
                        
        return import_edges

    def _get_node_name(self, node: Node, source_bytes: bytes) -> str:
        """Get the name field of a named node as a string."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return "unknown"
        return source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")

    def _get_node_text(self, node: Node, source_bytes: bytes) -> str:
        """Get full source text of any node."""
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def _extract_jsdoc(self, node: Node, source_bytes: bytes) -> str | None:
        """
        In JS/TS, docstrings are comment nodes that appear IMMEDIATELY
        before the function/class declaration.
        """
        prev_sibling = node.prev_sibling
        if prev_sibling and prev_sibling.type == "comment":
            text = self._get_node_text(prev_sibling, source_bytes)
            if text.startswith("/**"): # Ensure it's a JSDoc block, not just an inline comment
                return text
        return None

    def _extract_calls(self, body_node: Node, source_bytes: bytes) -> list[str]:
        """
        Extracts JS calls. 
        JS uses 'member_expression' (e.g., console.log, this.validate).
        """
        calls = set()

        def walk_for_calls(n: Node):
            if n.type == "call_expression":
                func_node = n.child_by_field_name("function")
                if func_node:
                    # In JS: identifier = fetch(), member_expression = this.validate()
                    if func_node.type in ("identifier", "member_expression"):
                        raw_name = self._get_node_text(func_node, source_bytes)
                        base_name = raw_name.split(".")[0]   # "os.path.join" → "os"
                        
                        if base_name not in JAVASCRIPT_BUILTINS:
                            calls.add(base_name)

            for child in n.children:
                walk_for_calls(child)

        if body_node:
            walk_for_calls(body_node)

        return list(calls)


# ════════════════════════════════════════════════════════
# FALLBACK STRATEGY
# For Language.UNKNOWN — char-split with overlap
# ════════════════════════════════════════════════════════

class FallbackStrategy(LanguageStrategy):
    """ doubt2
    Char-split with overlap for unsupported languages.
    Used for: YAML, shell scripts, markdown, config files.

    WHY char-split and not line-split?
    Line counts vary wildly. A minified JS file is 1 line.
    Character count is consistent across file types.

    WHY overlap?
    A concept might straddle a chunk boundary.
    Overlap ensures it appears fully in at least one chunk.

    Chunk size: 150 lines worth ~ 3000 chars
    Overlap:    10% = 300 chars
    """

    CHUNK_SIZE    = 3_000   # characters
    OVERLAP       = 300     # characters (10%)

    def parse(
        self,
        raw_file: RawFile,
        repo_name: str,
    ) -> tuple[list[CodeChunk], list[RawEdge]]:
        """
        Split content into overlapping character chunks.
        No AST, no call extraction — just text chunking.
        Returns no RawEdges (can't extract calls from unknown language).
        """
        content = raw_file.content
        chunks: list[CodeChunk] = []
        start = 0
        idx = 0

        while start < len(content):
            end = min(start + self.CHUNK_SIZE, len(content))
            chunk_text = content[start:end]

            # Count lines for start_line/end_line
            # WHY count lines even in fallback?
            # So the user can click a source reference and land
            # on the right line. Metadata must be consistent.
            lines_before = content[:start].count("\n")
            lines_in_chunk = chunk_text.count("\n")

            chunk_id = make_chunk_id(
                repo=repo_name,
                filepath=raw_file.path,
                classname=None,
                funcname=f"chunk_{idx}",
                signature=str(start),   # position as signature — unique per chunk
            )

            chunks.append(CodeChunk(
                repo_name=repo_name,
                chunk_id=chunk_id,
                symbol_name=f"chunk_{idx}",
                chunk_type=ChunkType.MODULE,
                language=raw_file.language,
                file_path=raw_file.path,
                start_line=lines_before + 1,
                end_line=lines_before + lines_in_chunk + 1,
                text=chunk_text,
                docstring=None,
                calls_out_raw=[],
                imports_raw=[],
            ))

            idx += 1
            # Move forward by CHUNK_SIZE - OVERLAP
            # WHY subtract overlap? So next chunk starts 300 chars
            # before this one ended — ensuring continuity
            start += self.CHUNK_SIZE - self.OVERLAP

        logger.debug(
            "Fallback chunking: %s → %d chunks", raw_file.path, len(chunks)
        )
        return chunks, []   # no edges from fallback


# ════════════════════════════════════════════════════════
# STRATEGY REGISTRY
# Single source of truth for language → strategy mapping.
# Adding a language: one line here, one new strategy class.
# ════════════════════════════════════════════════════════

STRATEGY_REGISTRY: dict[Lang, LanguageStrategy] = {
    Lang.PYTHON:     PythonStrategy(),
    Lang.JAVASCRIPT: JavaScriptStrategy(),
    Lang.TYPESCRIPT: JavaScriptStrategy(),   # same grammar handles TS
    Lang.UNKNOWN:    FallbackStrategy(),
}

# Strategies are stateless — one instance per type is correct.
# WHY stateless? parse() takes all its inputs as arguments.
# No shared mutable state between calls.


# ════════════════════════════════════════════════════════
# PARSER — THE PUBLIC INTERFACE
# This is what indexer.py calls. It knows nothing about
# which strategy runs — that's the point.
# ════════════════════════════════════════════════════════

class CodeParser:
    """
    Public interface for Pass 1.
    """

    def __init__(self):
        self._registry = STRATEGY_REGISTRY

    def parse(
        self,
        raw_file: RawFile,
        repo_name: str,
    ) -> tuple[list[CodeChunk], list[RawEdge]]:
        """
        Parse one file. Picks strategy by language.
        Falls back to FallbackStrategy if language not in registry.
        """
        strategy = self._registry.get(raw_file.language)

        if strategy is None:
            # Language detected but no strategy registered yet
            # (e.g., Java, Rust) — use fallback
            logger.warning(
                "No strategy for language %s in %s — using fallback",
                raw_file.language, raw_file.path
            )
            strategy = self._registry[Lang.UNKNOWN]

        try:
            chunks, raw_edges = strategy.parse(raw_file, repo_name)
            logger.debug(
                "Parsed %s → %d chunks, %d raw edges",
                raw_file.path, len(chunks), len(raw_edges)
            )
            return chunks, raw_edges

        except Exception as e:
            # One file failing must not stop the whole repo indexing
            # Log it, return empty, indexer records the failure
            logger.error(
                "Parse failed for %s: %s", raw_file.path, e
            )
            return [], []

    def stream_parse(
        self,
        files: Generator[RawFile, None, None],
        repo_name: str,
    ) -> Generator[tuple[list[CodeChunk], list[RawEdge]], None, None]:
        """
        Generator wrapper for streaming ingestion.

        WHY a generator here?
        indexer.py streams files from the loader.
        We want to parse as we go — constant memory per file.
        BUT we still need all chunks in ParsedRepo for Pass 2.
        indexer.py accumulates chunks while streaming.

        Usage:
            for chunks, edges in parser.stream_parse(files, repo):
                parsed_repo.chunks.extend(chunks)
                parsed_repo.raw_edges.extend(raw_edges)
        """
        for raw_file in files:
            yield self.parse(raw_file, repo_name)