import streamlit as st
from dotenv import load_dotenv
import os
import asyncio
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, Agent, Runner, function_tool, RunConfig
from ddgs import DDGS
from twilio.rest import Client

# ✅ Load secrets
load_dotenv()

# ✅ User database (static for now)
@function_tool
def get_user_data(min_age: int) -> list[dict]:
    "Retrieve user data based on a minimum age"
    users = [
        {"name": "Muneeb", "age": 2},
        {"name": "Zainscity", "age": 25},
        {"name": "Azan", "age": 19},
    ]
    return [user for user in users if user["age"] >= min_age]

# ✅ DuckDuckGo Search Tool
@function_tool
def search_duckduckgo(query: str) -> list[dict]:
    "Search the web using DuckDuckGo"
    with DDGS() as ddgs:
        results = ddgs.text(query)
        return [
            {"title": r["title"], "href": r["href"], "body": r["body"]}
            for r in results[:5]
        ]

# ✅ WhatsApp sender
def send_whatsapp_message(message: str):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_whatsapp = os.getenv("TWILIO_WHATSAPP_NUMBER")
    to_whatsapp = os.getenv("MY_WHATSAPP_NUMBER")

    if not all([account_sid, auth_token, from_whatsapp, to_whatsapp]):
        raise Exception("❌ Twilio credentials missing or incomplete.")

    client = Client(account_sid, auth_token)
    client.messages.create(
        body=message,
        from_=from_whatsapp,
        to=to_whatsapp
    )

# ✅ Async runner
async def run_agent_async(user_query: str):
    MODEL_NAME = "gemini-2.0-flash"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    external_client = AsyncOpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    model = OpenAIChatCompletionsModel(
        model=MODEL_NAME,
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=False  # Enable to show reasoning steps
    )

    rishtey_wali_agent = Agent(
        name="Auntie",
        model=model,
        instructions="You are a warm and wise 'Rishtey Wali Auntie' who helps people find marriage matches in a caring, funny and auntie-style way.",
        tools=[get_user_data, search_duckduckgo]
    )

    result = await Runner.run(
        starting_agent=rishtey_wali_agent,
        input=user_query,
        run_config=config
    )

    return result

# ==========================
# ✅ Streamlit App UI
# ==========================

st.set_page_config(page_title="🤖 Rishtey Wali Auntie", layout="centered")

st.title("💍 Rishtey Wali Auntie – AI Matchmaker")
st.markdown("Meet your personal rishta expert who finds matches and gives social media info in classic desi style.")

with st.expander("🛠️ Customize your matchmaking"):
    min_age = st.slider("Minimum Age", 18, 40, 20)
    platforms = st.multiselect("Platforms to search:", ["LinkedIn", "Instagram", "Facebook", "TikTok"], default=["Instagram", "Facebook"])
    custom_input = st.text_input("Custom request", 
        value=f"Find a match of {min_age} minimum age and tell me the details from {', '.join(platforms)}.")

    send_to_whatsapp = st.checkbox("📲 Send result to WhatsApp", value=True)
    show_debug = st.checkbox("🐞 Show .env debug info")

if st.button("🔍 Find Match"):
    with st.spinner("Auntie is searching for rishta... 🕵️‍♀️"):
        try:
            result = asyncio.run(run_agent_async(custom_input))

            st.success("🎯 Rishta Found!")
            st.subheader("💬 Auntie says:")
            st.markdown(result.final_output)

            # Optional: Show intermediate steps
            if hasattr(result, 'steps'):
                with st.expander("📜 Auntie's Reasoning"):
                    for i, step in enumerate(result.steps):
                        st.markdown(f"**Step {i+1}:** {step.input}")
                        st.code(step.output)

            if send_to_whatsapp:
                send_whatsapp_message(result.final_output)
                st.info("✅ Message sent to your WhatsApp.")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ==========================
# 🐞 Debug Info
# ==========================

if show_debug:
    st.markdown("### 🔐 Environment Debug Info")
    st.code({
        "TWILIO_ACCOUNT_SID": os.getenv("TWILIO_ACCOUNT_SID"),
        "TWILIO_AUTH_TOKEN": "********",
        "TWILIO_WHATSAPP_NUMBER": os.getenv("TWILIO_WHATSAPP_NUMBER"),
        "MY_WHATSAPP_NUMBER": os.getenv("MY_WHATSAPP_NUMBER"),
        "GEMINI_API_KEY": "********" if os.getenv("GEMINI_API_KEY") else None
    })
