import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain.chat_models import ChatOpenAI

# Initialize ChatOpenAI with the correct API key
llm = ChatOpenAI(openai_api_key=st.secrets["IS883-OpenAIKey-RV"], model="gpt-4o-mini")

# Streamlit setup
st.title("Airline Experience Feedback")

# User feedback input
user_feedback = st.text_input("Share with us your experience of the latest trip.", "")

# Refined template to classify feedback type
classification_template = """Classify the feedback into one of the following categories:
1. "negative_airline" if the feedback is negative and specifically related to services provided by the airline (e.g., lost luggage, bad food, rude staff, delayed baggage).
2. "negative_other" if the feedback is negative but due to reasons beyond the airline's control (e.g., weather delay, security checkpoint delay, airport infrastructure issues).
3. "positive" if the feedback is positive.

Please respond with only one word: "negative_airline", "negative_other", or "positive".

Feedback:
{feedback}
"""

# Classification chain
classification_prompt = PromptTemplate(input_variables=["feedback"], template=classification_template)
classification_chain = LLMChain(llm=llm, prompt=classification_prompt, output_parser=StrOutputParser())

# Response chains for feedback types
negative_airline_chain = PromptTemplate.from_template(
    "We apologize for the inconvenience caused by our services. Our customer service team will contact you shortly."
) | llm

negative_other_chain = PromptTemplate.from_template(
    "We're sorry for the inconvenience. However, the situation was beyond our control. We appreciate your understanding."
) | llm

positive_chain = PromptTemplate.from_template(
    "Thank you for your positive feedback! We're glad you had a great experience with us."
) | llm

# Branching logic
branch = RunnableBranch(
    (lambda x: isinstance(x["feedback_type"], str) and x["feedback_type"].lower() == "negative_airline", negative_airline_chain),
    (lambda x: isinstance(x["feedback_type"], str) and x["feedback_type"].lower() == "negative_other", negative_other_chain),
    positive_chain,  # This will handle "positive" classification as a fallback
)

# Combine classification and branching into full chain
full_chain = {"feedback_type": classification_chain, "feedback": lambda x: x["feedback"]} | branch

# Run the chain if user feedback is provided
if user_feedback:
    try:
        # Execute the chain and get response
        response = full_chain.invoke({"feedback": user_feedback})

        # Debug: Show classification result
        classification_result = classification_chain.run({"feedback": user_feedback})
        st.write("Classification result:", classification_result)
        
        # Check if the response is an AIMessage object and display its content
        if hasattr(response, "content"):
            st.write(response.content)
        else:
            st.write("Unexpected response format:", response)  # Fallback for unexpected types
            
    except Exception as e:
        st.error(f"An error occurred while processing your feedback: {e}")
