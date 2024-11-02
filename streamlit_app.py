import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain.chat_models import ChatOpenAI

# Initialize the OpenAI LLM with gpt-4o-mini model using the API key from secrets
llm = ChatOpenAI(openai_api_key=st.secrets["IS883-OpenAIKey-RV"], model="gpt-4o-mini")

# Streamlit app setup
st.title("Airline Experience Feedback")

# User input for feedback
user_feedback = st.text_input("Share with us your experience of the latest trip.", "")

# Create a template to classify the feedback type
classification_template = """Classify the feedback into one of the following categories:
1. "negative_airline" if the feedback is negative and caused by the airline (e.g., lost luggage).
2. "negative_other" if the feedback is negative but not caused by the airline (e.g., weather delay).
3. "positive" if the feedback is positive.

Please respond with only one word: "negative_airline", "negative_other", or "positive".

Feedback:
{feedback}
"""

# Create the classification chain
classification_prompt = PromptTemplate(input_variables=["feedback"], template=classification_template)
classification_chain = LLMChain(llm=llm, prompt=classification_prompt, output_parser=StrOutputParser())

# Define response chains for each feedback type with richer responses
negative_airline_chain = PromptTemplate.from_template(
    """We apologize for the inconvenience caused by our services. Our customer service team will contact you shortly."""
) | llm

negative_other_chain = PromptTemplate.from_template(
    """We're sorry for the inconvenience. However, the situation was beyond our control. We appreciate your understanding."""
) | llm

positive_chain = PromptTemplate.from_template(
    """Thank you for your positive feedback! We're glad you had a great experience with us."""
) | llm

# Set up branching logic
branch = RunnableBranch(
    (lambda x: isinstance(x["feedback_type"], str) and "negative_airline" in x["feedback_type"].lower(), negative_airline_chain),
    (lambda x: isinstance(x["feedback_type"], str) and "negative_other" in x["feedback_type"].lower(), negative_other_chain),
    positive_chain,
)

# Combine classification and branching in the full chain
full_chain = {"feedback_type": classification_chain, "feedback": lambda x: x["feedback"]} | branch

# Run the chains if user feedback is provided
if user_feedback:
    try:
        # Run the full chain with user feedback as input
        response = full_chain.invoke({"feedback": user_feedback})
        
        # Extract and display only the "content" part of the response
        if isinstance(response, dict) and "content" in response:
            st.write(response["content"])
        else:
            st.write(response)  # Fallback in case the response is not structured as expected
    except Exception as e:
        st.error(f"An error occurred while processing your feedback: {e}")
