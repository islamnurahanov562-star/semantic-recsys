# Error Analysis

- Model: sentence-transformers/all-MiniLM-L6-v2
- Index: faiss-flat
- Items: 8804
- Queries: 50

## Aggregate Metrics

| Metric | Value |
| --- | ---: |
| n_queries | 50 |
| recall@1 | 0.21 |
| hit_rate@1 | 0.24 |
| recall@3 | 0.32 |
| hit_rate@3 | 0.36 |
| recall@10 | 0.4217 |
| hit_rate@10 | 0.48 |
| mrr | 0.3116 |
| avg_latency_ms | 74.452 |

## Lowest-Recall Queries

### A son challenges caste barriers to honor his father

- Recall@10: 0.0000
- First relevant rank: not found
- Relevant titles:
  - Manusangada (Dramas)
- Top-3 recommendations:
  - 1. Kapoor & Sons (Comedies), score=0.4165
  - 2. Life of An Outcast (Dramas), score=0.3971
  - 3. Material (Comedies), score=0.3914

Failure type: other (possible label incompleteness). The query clearly matches the source description. `Life of An Outcast` also describes caste bias and a father helping his son, so the relevance set may omit a plausible match.

### A grieving husband seeks love after a painful betrayal

- Recall@10: 0.0000
- First relevant rank: not found
- Relevant titles:
  - Satu Hari (International TV Shows)
- Top-3 recommendations:
  - 1. The Lost Husband (Dramas), score=0.5132
  - 2. Finding Hubby (Comedies), score=0.4246
  - 3. Time Please (Comedies), score=0.4196

Failure type: model limitation. The source description matches betrayal followed by a new relationship, but the top results match generic title terms such as "husband" and "love".

### Outlaw brothers encounter lawmen and supernatural dangers

- Recall@10: 0.0000
- First relevant rank: not found
- Relevant titles:
  - From Dusk Till Dawn (Classic & Cult TV)
- Top-3 recommendations:
  - 1. Lawless (Dramas), score=0.5700
  - 2. The Brothers Grimm (Action & Adventure), score=0.5503
  - 3. Malevolent (Horror Movies), score=0.5229

Failure type: model limitation. The source description combines bank-robbing brothers, lawmen, and demons; the top results match only subsets of those concepts.

### A suspect uses television footage to challenge a murder charge

- Recall@10: 0.0000
- First relevant rank: not found
- Relevant titles:
  - The Whole Truth (Dramas)
  - Trial 4 (Crime TV Shows)
  - Long Shot (Documentaries)
- Top-3 recommendations:
  - 1. Making a Murderer (Crime TV Shows), score=0.5821
  - 2. The Murder Detectives (British TV Shows), score=0.5711
  - 3. 20 Minutes (Crime TV Shows), score=0.5570

Failure type: model limitation. `Long Shot` directly matches the television-footage clue but is not retrieved. The other two labels concern murder cases without that clue and should be reviewed separately.

### A hotheaded newcomer turns to underground fights for escape

- Recall@10: 0.0000
- First relevant rank: not found
- Relevant titles:
  - Never Back Down (Action & Adventure)
  - Pandigai (Action & Adventure)
- Top-3 recommendations:
  - 1. 6 Underground (Action & Adventure), score=0.5334
  - 2. S.W.A.T.: Under Siege (Action & Adventure), score=0.4398
  - 3. Fallen (Thrillers), score=0.4332

Failure type: model limitation. Both source descriptions explicitly mention joining an underground fight club, while the top result is an unrelated spy film matching the word "Underground".

### Families witness the emotional strain of childhood cancer

- Recall@10: 0.0000
- First relevant rank: not found
- Relevant titles:
  - A Lion in the House (TV Shows)
  - Weed the People (Documentaries)
- Top-3 recommendations:
  - 1. Diamonds in the Sky (Dramas), score=0.4817
  - 2. Dad (Dramas), score=0.4426
  - 3. If Anything Happens I Love You (Dramas), score=0.3935

Failure type: other (label incompleteness). `Diamonds in the Sky` in the top results describes families affected by cancer and appears relevant, but is absent from the labels. `Weed the People` is narrower and should also be reviewed.

### A psychologist joins a survivor to hunt a predator

- Recall@10: 0.0000
- First relevant rank: not found
- Relevant titles:
  - The Women and the Murderer (Documentaries)
  - Kiss the Girls (Thrillers)
- Top-3 recommendations:
  - 1. Prey (International Movies), score=0.5046
  - 2. Hunt for the Wilderpeople (Action & Adventure), score=0.4578
  - 3. The Victims' Game (Crime TV Shows), score=0.4452

Failure type: model limitation. `Kiss the Girls` matches the psychologist-and-survivor description closely but is not retrieved. `The Women and the Murderer` is only a partial match and should be reviewed.

### Two minds share one body and face a breaking point

- Recall@10: 0.0000
- First relevant rank: not found
- Relevant titles:
  - Jonathan (Dramas)
- Top-3 recommendations:
  - 1. Face 2 Face (Dramas), score=0.4071
  - 2. The Mind, Explained (Docuseries), score=0.4011
  - 3. The Body Remembers When the World Broke Open (Dramas), score=0.3976

Failure type: model limitation. The source description explicitly states that two siblings share one body; the top results match isolated title words such as "face", "mind", and "body".

### A teenager dreams of flying beyond a forced future

- Recall@10: 0.0000
- First relevant rank: not found
- Relevant titles:
  - Sitara: Let Girls Dream (Children & Family Movies)
- Top-3 recommendations:
  - 1. The Dreamer (Dramas), score=0.4650
  - 2. Memories of a Teenager (Dramas), score=0.4463
  - 3. Imperial Dreams (Dramas), score=0.4288

Failure type: model limitation. The source description directly connects the girl's wish to become a pilot with her father's marriage plan, but the top results match "dream" and "teenager" without that situation.

### A quirky friend helps an aspiring actor make a film

- Recall@10: 0.0000
- First relevant rank: not found
- Relevant titles:
  - The Disaster Artist (Comedies)
- Top-3 recommendations:
  - 1. Classmates Minus (Comedies), score=0.5374
  - 2. Friend Request (Horror Movies), score=0.5352
  - 3. Friendship (Action & Adventure), score=0.5258

Failure type: model limitation. The source description matches an aspiring actor, an unusual friend, and making a movie; the top results primarily match the word "friend".

## Review Summary

This draft assigns 8 of the 10 cases primarily to model limitation and 2 to possible label incompleteness. No primary data issue or ambiguous-query case was identified in this sample. These are review judgments based on the saved item descriptions and retrieved results, not an automatic diagnosis or a claim that the sample represents all 50 queries.

A check of the raw and processed CSVs found zero occurrences of the suspected strings `tripinto`, `murderhe`, and `newlife`; no preprocessing change is justified by those examples.

## Proposed Follow-up

1. Review the relevance sets for `Manusangada`, the television-footage query, the childhood-cancer query, and the psychologist query; change labels only after confirming candidates against descriptions.
2. Compare title-plus-description embeddings against description-only embeddings on the same fixed query and relevance set to test whether title-word overlap affects these cases.
3. Tune FAISS IVF `nprobe` and report the recall/latency trade-off; the current saved run uses the default configuration.
4. Investigate raw-data cleaning only if a concrete malformed value is found in the source file; the three suspected glued tokens were not found.
