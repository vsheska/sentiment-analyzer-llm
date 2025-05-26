from transformers import pipeline

sentiment_pipeline = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)


def predict_llm(review):
    result = sentiment_pipeline(review)[0]
    return f"{result['label']} ({result['score']:.2f})"
