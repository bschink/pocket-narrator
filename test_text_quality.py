#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from pocket_narrator.text_quality import evaluate_text_quality, TextQualityConfig, _Embedder, split_sentences, extract_entities, compute_coherence_entity_overlap, compute_cohesion_semantic
import numpy as np

story = """Once there was a swan who wanted to take a swim. But he had no pond to dive into. He asked his animal friends, one by one, if they could lend him some water, but none of them had any to spare. The swan was feeling very sad until he met a small frog. The frog said he could lend the swan some of his water. The swan was so happy he gave the frog a big hug! The other animals were feeling very envious watching the swan swim. So the frog had an idea and gathered all of his friends around the pond. They took turns sharing the water so everyone could enjoy a swim. Everyone had a great time and the swan thanked the frog for his kindness. From then on, everyone enjoyed taking turns lending and sharing their water."""

print("Evaluating text quality (coherence and cohesion)...\n")

try:
    cfg = TextQualityConfig()
    print(f"Config: use_sentence_transformers={cfg.use_sentence_transformers}")
    
    # Initialize embedder if using sentence transformers
    embedder = None
    if cfg.use_sentence_transformers:
        try:
            print("Loading sentence transformers embedder...")
            embedder = _Embedder(cfg.st_model)
            print(f"Embedder loaded: {cfg.st_model}\n")
        except Exception as e:
            print(f"Warning: Could not load embedder: {e}\n")
    
    # Step-by-step evaluation for debugging
    print("Step 1: Splitting sentences...")
    sents = split_sentences(story, cfg)
    print(f"  Found {len(sents)} sentences:")
    for i, sent in enumerate(sents):
        print(f"    {i+1}. {sent[:60]}...")
    
    print("\nStep 2: Extracting entities...")
    entity_sets = extract_entities(sents, cfg)
    print(f"  Extracted entities from {len(entity_sets)} sentences")
    for i, es in enumerate(entity_sets):
        print(f"    Sentence {i+1}: {es}")
    
    print("\nStep 3: Computing coherence (entity overlap)...")
    coherence_result = compute_coherence_entity_overlap(entity_sets)
    print(f"  Coherence: {coherence_result}")
    
    print("\nStep 4: Computing cohesion (semantic similarity)...")
    if embedder:
        vecs = embedder.encode(sents)
        print(f"  Embeddings shape: {vecs.shape if hasattr(vecs, 'shape') else type(vecs)}")
        if hasattr(vecs, 'shape'):
            print(f"  Vector shape: {vecs[0].shape if len(vecs) > 0 else 'empty'}")
    
    cohesion_result = compute_cohesion_semantic(sents, cfg, embedder=embedder)
    print(f"  Cohesion: {cohesion_result}")
    
    print("\nStep 5: Full evaluation...")
    results = evaluate_text_quality(story, cfg=cfg, embedder=embedder)
    
    print("\n" + "=" * 60)
    print("TEXT QUALITY EVALUATION RESULTS")
    print("=" * 60)
    
    for key, value in sorted(results.items()):
        if isinstance(value, float):
            print(f"{key:.<40} {value:.4f}")
        else:
            print(f"{key:.<40} {value}")
    
    print("\n" + "=" * 60)
    print("KEY METRICS:")
    print("=" * 60)
    coh_val = results.get('coherence', float('nan'))
    cohes_val = results.get('cohesion_mean', float('nan'))
    quality_val = results.get('text_quality', float('nan'))
    
    coh_str = f"{coh_val:.4f}" if not np.isnan(coh_val) else "NaN"
    cohes_str = f"{cohes_val:.4f}" if not np.isnan(cohes_val) else "NaN"
    quality_str = f"{quality_val:.4f}" if not np.isnan(quality_val) else "NaN"
    
    print(f"Coherence (entity overlap):    {coh_str}")
    print(f"Cohesion (semantic similarity): {cohes_str}")
    print(f"Overall Text Quality Score:    {quality_str}")
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()

