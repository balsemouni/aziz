"""
CAG Architecture - Solution Recommendation Knowledge Store Module
Maps user problems to commercial solutions with maximum token efficiency
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SolutionEntry:
    """Single problem-solution mapping entry"""
    user_problem: str
    problem_keywords: List[str]
    solution_name: str
    solution_description: str
    key_benefits: List[str]
    pricing_model: str
    implementation_time: str
    target_industries: List[str]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_compact_string(self) -> str:
        """
        Convert to ULTRA-COMPACT format for maximum token efficiency
        Format: PROBLEM:problem|SOLUTION:name|DESC:description|BENEFITS:benefit1,benefit2|PRICE:price|TIME:time
        """
        problem = self.user_problem.strip().replace('\n', ' ').replace('  ', ' ')
        benefits_str = "; ".join(self.key_benefits[:3])  # Top 3 benefits only
        
        return (
            f"PROBLEM:{problem}|"
            f"SOLUTION:{self.solution_name}|"
            f"DESC:{self.solution_description}|"
            f"BENEFITS:{benefits_str}|"
            f"PRICE:{self.pricing_model}|"
            f"TIME:{self.implementation_time}"
        )


class SolutionKnowledgeStore:
    """
    Solution Knowledge Store - Maps user problems to commercial solutions
    
    Efficiently loads problem-solution pairs into context window for chatbot matching
    """
    
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.entries: List[SolutionEntry] = []
        self.knowledge_text: Optional[str] = None
        self.token_count: int = 0
        
    def load_from_sources(self) -> int:
        """
        Load ALL solution entries from JSON
        
        Returns:
            Number of entries loaded
        """
        # FIXED: Handle both attribute and dict-style config access
        if hasattr(self.config, 'solutions_json_path'):
            json_path = self.config.solutions_json_path
        elif hasattr(self.config, 'knowledge_jsonl_path'):
            # Fallback to knowledge_jsonl_path if solutions_json_path doesn't exist
            json_path = self.config.knowledge_jsonl_path
        else:
            raise ValueError("Config must have 'solutions_json_path' or 'knowledge_jsonl_path'")
        
        if not os.path.exists(json_path):
            raise ValueError(f"Solutions file not found: {json_path}")
        
        print(f"\n📂 Loading solutions from: {json_path}")
        
        # Detect file format based on extension
        if json_path.endswith('.jsonl'):
            loaded_count = self._load_from_jsonl(json_path)
        else:
            loaded_count = self._load_from_json(json_path)
        
        print(f"✅ Loaded {loaded_count:,} solution entries")
        
        if loaded_count == 0:
            raise ValueError("No solution data found in file!")
        
        return loaded_count
    
    def _load_from_json(self, path: str) -> int:
        """
        Load ALL solution entries from JSON array
        """
        count = 0
        skipped = 0
        
        print("📖 Reading solutions JSON...")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Handle both array and object with 'solutions' key
            solutions = data if isinstance(data, list) else data.get('solutions', [])
            
            for idx, item in enumerate(solutions):
                try:
                    # Validation
                    required_fields = ['user_problem', 'solution_name', 'solution_description']
                    if not all(item.get(field) for field in required_fields):
                        skipped += 1
                        continue
                    
                    entry = SolutionEntry(
                        user_problem=item['user_problem'].strip(),
                        problem_keywords=item.get('problem_keywords', []),
                        solution_name=item['solution_name'].strip(),
                        solution_description=item['solution_description'].strip(),
                        key_benefits=item.get('key_benefits', []),
                        pricing_model=item.get('pricing_model', 'Contact for pricing'),
                        implementation_time=item.get('implementation_time', 'Varies'),
                        target_industries=item.get('target_industries', []),
                        metadata={
                            'source': 'json',
                            'index': idx
                        }
                    )
                    self.entries.append(entry)
                    count += 1
                    
                    # Progress indicator every 100 entries
                    if count % 100 == 0:
                        print(f"   Loaded {count:,} solutions...")
                    
                except (KeyError, ValueError) as e:
                    skipped += 1
                    if count < 10:  # Only show first 10 errors
                        print(f"⚠️  Skipped entry {idx}: {str(e)}")
                    continue
        
        if skipped > 0:
            print(f"⚠️  Skipped {skipped:,} invalid entries")
        
        return count
    
    def _load_from_jsonl(self, path: str) -> int:
        """
        Load solution entries from JSONL format (one JSON object per line)
        """
        count = 0
        skipped = 0
        
        print("📖 Reading solutions JSONL...")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    item = json.loads(line)
                    
                    # Validation
                    required_fields = ['user_problem', 'solution_name', 'solution_description']
                    if not all(item.get(field) for field in required_fields):
                        skipped += 1
                        continue
                    
                    entry = SolutionEntry(
                        user_problem=item['user_problem'].strip(),
                        problem_keywords=item.get('problem_keywords', []),
                        solution_name=item['solution_name'].strip(),
                        solution_description=item['solution_description'].strip(),
                        key_benefits=item.get('key_benefits', []),
                        pricing_model=item.get('pricing_model', 'Contact for pricing'),
                        implementation_time=item.get('implementation_time', 'Varies'),
                        target_industries=item.get('target_industries', []),
                        metadata={
                            'source': 'jsonl',
                            'line_number': line_num
                        }
                    )
                    self.entries.append(entry)
                    count += 1
                    
                    # Progress indicator
                    if count % 100 == 0:
                        print(f"   Loaded {count:,} solutions...")
                    
                except json.JSONDecodeError as e:
                    skipped += 1
                    if count < 10:
                        print(f"⚠️  Skipped line {line_num}: Invalid JSON")
                    continue
                except (KeyError, ValueError) as e:
                    skipped += 1
                    if count < 10:
                        print(f"⚠️  Skipped line {line_num}: {str(e)}")
                    continue
        
        if skipped > 0:
            print(f"⚠️  Skipped {skipped:,} invalid entries")
        
        return count
    
    def build_knowledge_text(self, max_tokens: Optional[int] = None, use_compact: bool = True) -> str:
        """
        Build solution knowledge text within the token budget.
        
        Args:
            max_tokens: Maximum tokens for knowledge base (from config if None)
            use_compact: Use compact PROBLEM:|SOLUTION: format (recommended)
        """
        if max_tokens is None:
            max_tokens = self.config.max_context_tokens

        # Reserve tokens for system prompt, query, and generation buffer
        reserved = 150 + 200 + 100 + getattr(self.config, 'cache_truncation_buffer', 100)
        available_tokens = max(500, max_tokens - reserved)

        print(f"\n🔨 Building solution knowledge base...")
        print(f"   max_context_tokens = {max_tokens:,}")
        print(f"   Knowledge token budget = {available_tokens:,}")
        print(f"   Total solutions available: {len(self.entries):,}")

        # Deduplication by problem category
        seen_categories = {}
        for entry in self.entries:
            if entry.problem_keywords:
                category = entry.problem_keywords[0].lower()
            else:
                words = entry.user_problem.lower().split()
                category = " ".join(words[:3])
            
            if category not in seen_categories:
                seen_categories[category] = entry

        deduplicated = list(seen_categories.values())
        print(f"   After dedup:             {len(deduplicated):,} unique solution categories")

        # Pack entries shortest-first to maximize count
        sorted_entries = sorted(
            deduplicated,
            key=lambda e: len(e.user_problem) + len(e.solution_description)
        )

        knowledge_parts = []
        current_tokens = 0
        entries_included = 0
        CHARS_PER_TOKEN_EST = 4.0

        max_entries = getattr(self.config, 'max_knowledge_entries', 10000)

        for entry in sorted_entries:
            if use_compact:
                formatted = entry.to_compact_string()
            else:
                formatted = f"Problem: {entry.user_problem}\nSolution: {entry.solution_name}\nDescription: {entry.solution_description}\nBenefits: {', '.join(entry.key_benefits[:3])}\nPricing: {entry.pricing_model}\nTime: {entry.implementation_time}"

            entry_tokens_est = (len(formatted) + 1) / CHARS_PER_TOKEN_EST

            if current_tokens + entry_tokens_est > available_tokens:
                break

            knowledge_parts.append(formatted)
            current_tokens += entry_tokens_est
            entries_included += 1

            if entries_included >= max_entries:
                print(f"   ⚠️  Hit max_knowledge_entries cap ({max_entries:,})")
                break

        self.knowledge_text = "\n\n".join(knowledge_parts)

        # Accurate token count
        self.token_count = len(self.tokenizer.encode(self.knowledge_text))

        coverage_pct = (entries_included / len(self.entries) * 100) if self.entries else 0
        efficiency = (entries_included / self.token_count) if self.token_count > 0 else 0

        print(f"\n📊 SOLUTION KNOWLEDGE BASE STATISTICS:")
        print(f"   {'='*60}")
        print(f"   Total solutions available:   {len(self.entries):,}")
        print(f"   Unique categories:           {len(deduplicated):,}")
        print(f"   Solutions included:          {entries_included:,} ({coverage_pct:.1f}%)")
        print(f"   {'─'*60}")
        print(f"   Tokens used:                 {self.token_count:,}")
        print(f"   Token budget:                {available_tokens:,}")
        print(f"   Token utilization:           {(self.token_count/available_tokens)*100:.1f}%")
        print(f"   {'─'*60}")
        if entries_included > 0:
            print(f"   Avg tokens per solution:     {self.token_count/entries_included:.1f}")
            print(f"   Solutions per 1k tokens:     {efficiency*1000:.1f}")
        print(f"   Format:                      {'Compact' if use_compact else 'Structured'}")
        print(f"   {'='*60}")

        return self.knowledge_text
    
    def get_knowledge_text(self) -> str:
        """Get the built knowledge text"""
        if self.knowledge_text is None:
            raise ValueError("Knowledge text not built. Call build_knowledge_text() first.")
        return self.knowledge_text
    
    def get_token_count(self) -> int:
        """Get the token count of knowledge base"""
        return self.token_count
    
    def get_entry_count(self) -> int:
        """Get total number of loaded entries"""
        return len(self.entries)
    
    def get_coverage_stats(self) -> Dict[str, Any]:
        """Get statistics about solution coverage"""
        if not self.knowledge_text:
            return {'coverage': 0, 'included': 0, 'total': len(self.entries)}
        
        included = len([block for block in self.knowledge_text.split('\n\n') if block.strip()])
        total = len(self.entries)
        coverage = (included / total * 100) if total > 0 else 0
        
        industries = set()
        for entry in self.entries:
            industries.update(entry.target_industries)
        
        return {
            'total_solutions': total,
            'included_solutions': included,
            'excluded_solutions': total - included,
            'coverage_percent': round(coverage, 2),
            'tokens_used': self.token_count,
            'max_tokens': self.config.max_context_tokens,
            'token_efficiency': round(self.token_count / included, 2) if included > 0 else 0,
            'unique_industries': len(industries),
            'industries_list': sorted(industries)
        }
    
    def save_metadata(self, path: Optional[str] = None):
        """Save comprehensive solution knowledge base metadata"""
        if path is None:
            path = getattr(self.config, 'cache_metadata_path', 'cache_metadata.json')
        
        coverage = self.get_coverage_stats()
        
        metadata = {
            'total_solutions': len(self.entries),
            'included_solutions': coverage['included_solutions'],
            'excluded_solutions': coverage['excluded_solutions'],
            'coverage_percent': coverage['coverage_percent'],
            'token_count': self.token_count,
            'max_tokens': self.config.max_context_tokens,
            'token_efficiency': coverage['token_efficiency'],
            'character_count': len(self.knowledge_text) if self.knowledge_text else 0,
            'unique_industries': coverage['unique_industries'],
            'industries': coverage['industries_list']
        }
        
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if getattr(self.config, 'verbose', False):
            print(f"💾 Metadata saved to {path}")
    
    def load_metadata(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Load solution knowledge base metadata"""
        if path is None:
            path = getattr(self.config, 'cache_metadata_path', 'cache_metadata.json')
        
        if not os.path.exists(path):
            return {}
        
        with open(path, 'r') as f:
            return json.load(f)
    
    def preview_entries(self, n: int = 3):
        """Preview first N solutions for debugging"""
        print(f"\n📋 Preview of first {min(n, len(self.entries))} solutions:")
        print("=" * 80)
        
        for i, entry in enumerate(self.entries[:n], 1):
            print(f"\nSolution {i}:")
            print(f"PROBLEM: {entry.user_problem[:100]}...")
            print(f"SOLUTION: {entry.solution_name}")
            print(f"BENEFITS: {', '.join(entry.key_benefits[:2])}...")
            print(f"PRICING: {entry.pricing_model}")
            
            compact = entry.to_compact_string()
            tokens = len(self.tokenizer.encode(compact))
            print(f"Tokens: {tokens}")
        
        print("=" * 80)


# For backward compatibility - alias to original name
KnowledgeStore = SolutionKnowledgeStore