from src.ingest.build_sentence_dataset import build_sentence_chunks


def estimate_tokens(text):
    # грубая оценка: 1 token ≈ 4 chars
    return len(text) // 4


chunks, metadata = build_sentence_chunks("data/processed")

lengths = [len(c) for c in chunks]
tokens = [estimate_tokens(c) for c in chunks]

print("=== CHUNK STATS ===")
print(f"Total chunks: {len(chunks)}")
print(f"Avg chars: {sum(lengths)/len(lengths):.1f}")
print(f"Avg tokens: {sum(tokens)/len(tokens):.1f}")
print(f"Max tokens: {max(tokens)}")
print(f"Min tokens: {min(tokens)}")

print("\n=== SAMPLE CHUNKS ===")
for i in range(5):
    print(f"\n--- CHUNK {i} ---")
    print(f"chars: {len(chunks[i])}")
    print(chunks[i][:500])
