from transformers import pipeline

# Load QA pipeline (you can replace this with a summarization or chat model too)
qa = pipeline("text2text-generation", model="google/flan-t5-small")  # Fast and small

def generate_answer(df, question):
    # Very basic logic. Replace with your LLM logic
    if "fail" in question.lower():
        try:
            fail_times = df[df['Status'].str.lower() == 'fail']['Timestamp'].value_counts()
            return str(fail_times.idxmax()) if not fail_times.empty else "No failures found."
        except:
            return "Could not determine failure timing."
    return "This is a placeholder answer for: " + question
