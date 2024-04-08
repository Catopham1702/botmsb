import streamlit as st
import streamlit_option_menu
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

# Class representing the chatbot using LlamaCpp
class LlamaCppChatbot:
    def __init__(self, model_path, n_gpu_layers=-1, n_batch=4):
        # Callbacks support token-wise streaming
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            callback_manager=self.callback_manager,
            verbose=True,  # Verbose is required to pass to the callback manager
            max_length=200,
        )

    def generate_response(self, user_input):
        # Create prompt from user input
        template = f"Question: {user_input}\n\nAnswer:"
        prompt = PromptTemplate.from_template(template)
        # Invoke LlamaCpp model with the given user input
        llm_chain = prompt | self.llm
        response = llm_chain.invoke({"question": user_input})
        # Append the response to the session state
        st.session_state["messages"].append({"role": "assistant", "content": response})
        return response

# Create an instance of the chatbot
llama_chatbot = LlamaCppChatbot(model_path="D:\Chatlocal\phi2\phi-2.Q4_K_M.gguf")

with st.sidebar:
    selected = streamlit_option_menu.option_menu(
        menu_title="Model",
        options=["Phi-2", "Google's Gemma 2B Instruct", "Stable Code Instruct 3B"],
    )
    if selected == "Phi-2":
        st.title(f"You have selected {selected}")
    if selected == "Google's Gemma 2B Instruct":
        st.title(f"You have selected {selected}")
    if selected == "Stable Code Instruct 3B":
        st.title(f"You have selected {selected}")

st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by CP")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display messages in the chat
for msg in st.session_state.messages:
    st.write(f"{msg['role']}: {msg['content']}")

# User input text box
user_input = st.text_input("You:", value="")

# When the "Send" button is clicked
if st.button("Send"):
    # Add user message to the message list
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Generate response from the chatbot
    response = llama_chatbot.generate_response(user_input)
    # Display the chatbot's response directly on the web interface
    st.write("Assistant:", response)

# Display messages in the chat
for msg in st.session_state.messages:
    st.write(f"{msg['role']}: {msg['content']}")
