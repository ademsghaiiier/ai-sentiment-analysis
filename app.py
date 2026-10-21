import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🧠"
)

st.title("🧠 AI Sentiment Analyzer")

user_text = st.text_area(
    "Enter Text",
    placeholder="Write your text here..."
)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

sentiment_prompt = ChatPromptTemplate.from_template(
    """
    Analyze the sentiment of this text.

    Return only:
    Positive, Negative, or Neutral

    Text:
    {text}
    """
)

topic_prompt = ChatPromptTemplate.from_template(
    """
    Identify the main topic.

    Return only the topic.

    Text:
    {text}
    """
)

question_prompt = ChatPromptTemplate.from_template(
    """
    Generate one follow-up question related
    to the text.

    Text:
    {text}
    """
)

if st.button("Analyze 🚀"):

    if not user_text:
        st.warning("Please enter some text.")
    else:

        sentiment = llm.invoke(
            sentiment_prompt.format_messages(
                text=user_text
            )
        ).content

        topic = llm.invoke(
            topic_prompt.format_messages(
                text=user_text
            )
        ).content

        question = llm.invoke(
            question_prompt.format_messages(
                text=user_text
            )
        ).content

        st.subheader("Results")

        st.write("### Original Text")
        st.info(user_text)

        st.write("### Sentiment")
        st.success(sentiment)

        st.write("### Topic")
        st.success(topic)

        st.write("### Follow-up Question")
        st.info(question)