import gradio as gr
from models.bert_model import predict_bert
from models.llm_model import predict_llm

def analyze_review(review):
    # Placeholder predictions
    bert_result = predict_bert(review)
    llm_result = predict_llm(review)

    return bert_result, llm_result

iface = gr.Interface(
    fn=analyze_review,
    inputs=gr.Textbox(lines=4, placeholder="Enter a product review..."),
    outputs=[
        gr.Label(label="BERT Sentiment"),
        gr.Label(label="LLM Sentiment")
    ],
    title="Product Review Sentiment Analyzer",
    description="Compare sentiment predictions from a fine-tuned BERT model and a large language model (LLM)."
)

if __name__ == "__main__":
    iface.launch()