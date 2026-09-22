# rag-amazon-reviewer

## Decyzje projektowe

### Chunking — nie stosujemy

Recenzje są krótkie (po czyszczeniu w `src/data_loader.py` mają maksymalnie ~2000 znaków),
więc mieszczą się w całości w oknie modelu embeddingów. Dzielenie ich na fragmenty
rozbijałoby spójną opinię na kawałki bez kontekstu (np. „nie polecam” oddzielone od tego,
czego dotyczy). Dlatego **jedna recenzja = jeden `Document`** (`src/documents.py`).
