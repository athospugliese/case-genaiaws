import streamlit as st
import asyncio
import os
import urllib.parse
from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from pydantic import ValidationError
from streamlit.runtime.scriptrunner import get_script_run_ctx

from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus

st.set_page_config(
    page_title="Desafio Deloitte - DEV GENAI",
    page_icon="🤖",
    menu_items={}
)

LOGO_PATH = "https://cdn.worldvectorlogo.com/logos/deloitte-2.svg"

st.markdown(
    """
    <style>
    /* force sidebar to the right */
    [data-testid="stAppViewContainer"] {
        flex-direction: row-reverse;
    }

    /* general color adjustments */
    body, .css-fg4pbf {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }

    /* green accents: links, buttons, etc. */
    a, .stButton>button, .st-bb, .css-1cpxqw2 {
        color: #26890D !important;
    }
    .stButton>button {
        background-color: #FFFFFF;
        border: 1px solid #26890D !important;
    }
    .stButton>button:hover {
        background-color: #26890D !important;
        color: #FFFFFF !important;
    }

    /* hide streamlit status in top-right corner */
    [data-testid="stStatusWidget"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
#  main function
# ------------------------------------------------------------
async def main() -> None:

    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 80)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("Connecting to agent service..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"Error connecting to agent service: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = get_script_run_ctx().session_id
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # ------------------------------------------------------------
    # right sidebar
    # ------------------------------------------------------------
    with st.sidebar:
        st.image(LOGO_PATH, width=200)

        model_idx = agent_client.info.models.index(agent_client.info.default_model)
        model = st.selectbox(
            "LLM to use",
            options=agent_client.info.models,
            index=model_idx
        )

        agent_client.agent = agent_client.info.default_agent
        st.write(f"Agent to use: `{agent_client.agent}`")


    # ------------------------------------------------------------
    # display existing messages
    # ------------------------------------------------------------
    messages = st.session_state.messages
    if len(messages) == 0:
        WELCOME = "Olá! Sou um chatbot integrado a um motor de busca!"
        with st.chat_message("ai"):
            st.write(WELCOME)

    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # ------------------------------------------------------------
    # new user message
    # ------------------------------------------------------------
    use_streaming = True
    if user_input := st.chat_input():
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                )
                await draw_messages(stream, is_new=True)
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                )
                messages.append(response)
                st.chat_message("ai").write(response.content)
            st.rerun()
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()


# ------------------------------------------------------------
#  function to draw/display messages
# ------------------------------------------------------------
async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    last_message_type = None
    st.session_state.last_message = None

    streaming_content = ""
    streaming_placeholder = None

    while msg := await anext(messages_agen, None):
        if isinstance(msg, str):
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()
            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue

        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            case "ai":
                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                f"""Tool Call: {tool_call["name"]}""",
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status
                            status.write("Input:")
                            status.write(tool_call["args"])

                        for _ in range(len(call_results)):
                            tool_result: ChatMessage = await anext(messages_agen)
                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            case "custom":
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("Mensagem inesperada vinda do Agente")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            case _:
                st.error(f"ChatMessage Inesperada type: {msg.type}")
                st.write(msg)
                st.stop()


# ------------------------------------------------------------
#   share/resume chat dialog
# ------------------------------------------------------------
@st.dialog("Compartilhe")
def share_chat_dialog() -> None:
    session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
    st_base_url = urllib.parse.urlunparse(
        [session.client.request.protocol, session.client.request.host, "", "", "", ""]
    )
    if not st_base_url.startswith("https") and "localhost" not in st_base_url:
        st_base_url = st_base_url.replace("http", "https")
    chat_url = f"{st_base_url}?thread_id={st.session_state.thread_id}"
    st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
    st.info("Copie a URL acima para compartilhar ou retomar este chat")


# ------------------------------------------------------------
#  feedback function - like/dislike
# ------------------------------------------------------------
async def handle_feedback() -> None:
    """buttons deslike or like for agent response, iteration w/ feeedback from langsmith."""
    latest_run_id = st.session_state.messages[-1].run_id
    agent_client: AgentClient = st.session_state.agent_client

    ### dict to prevent duplicate submissions
    if "feedback_sent" not in st.session_state:
        st.session_state.feedback_sent = {}

    if latest_run_id not in st.session_state.feedback_sent:
        st.session_state.feedback_sent[latest_run_id] = None

    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Like", key=f"like_{latest_run_id}"):
            if st.session_state.feedback_sent[latest_run_id] != "like":
                st.session_state.feedback_sent[latest_run_id] = "like"
                try:
                    await agent_client.acreate_feedback(
                        run_id=latest_run_id,
                        key="human-feedback-like-dislike",
                        score=1.0,  
                        kwargs={"comment": "User pressed like"},
                    )
                    st.toast("Feedback recorded: Like")  
                except AgentClientError as e:
                    st.error(f"Error recording feedback: {e}")
                    st.stop()

    with col2:
        if st.button("👎 Dislike", key=f"dislike_{latest_run_id}"):
            if st.session_state.feedback_sent[latest_run_id] != "dislike":
                st.session_state.feedback_sent[latest_run_id] = "dislike"
                try:
                    await agent_client.acreate_feedback(
                        run_id=latest_run_id,
                        key="human-feedback-like-dislike",
                        score=0.0,  
                        kwargs={"comment": "User pressed dislike"},
                    )
                    st.toast("Feedback recorded: Dislike") 
                except AgentClientError as e:
                    st.error(f"Error recording feedback: {e}")
                    st.stop()


# ------------------------------------------------------------
# Run app!
# ------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())