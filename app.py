import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    return model, tokenizer

model, tokenizer = load_model()

st.title("Text Summarization with Transformers")

st.write("Paste your article text below and get a summarized version.")

input_text = st.text_area("Article Text", height=300)

if st.button("Summarize"):
    if input_text.strip():
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=300,
            min_length=40,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
            repetition_penalty=2.5,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.subheader("Summary:")
        st.write(summary)
    else:
        st.error("Please enter some text to summarize.")
