import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8001").rstrip("/")

st.set_page_config(page_title="Inventory Purchasing", page_icon="📦", layout="wide")
st.title("Inventory & Purchasing Assistant")
st.caption("Local-first inventory queries and human-approved supplier purchase orders.")

with st.sidebar:
    st.subheader("API")
    st.code(API_BASE)
    if st.button("Refresh inventory"):
        st.rerun()

chat_tab, inventory_tab, approval_tab, trace_tab = st.tabs(
    ["Agent chat", "Inventory", "Approvals", "Reasoning trace"]
)

with chat_tab:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for entry in st.session_state.messages:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
    prompt = st.chat_input("Ask about stock or explicitly request a draft purchase order")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            response = requests.post(
                f"{API_BASE}/chat",
                json={"message": prompt, "thread_id": st.session_state.thread_id},
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            st.session_state.thread_id = payload["thread_id"]
            answer = payload.get("response", "")
            if payload.get("pending_approval"):
                pending = payload["pending_approval"]
                answer += (
                    f"\n\n**Approval required:** {pending['order_id']} — "
                    f"${pending['total']:.2f}. Choose YES or NO in the Approvals tab."
                )
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        except requests.HTTPError as exc:
            response = exc.response
            try:
                error = response.json().get("error", {})
                message = error.get("message", response.text)
                code = error.get("code")
            except (ValueError, AttributeError):
                message = response.text if response is not None else str(exc)
                code = None
            if code == "missing_api_key":
                st.error(
                    f"Chat is not configured: {message} Add GOOGLE_API_KEY to .env "
                    "or configure a running Ollama model."
                )
            elif code == "llm_quota_exceeded":
                st.warning(
                    f"Gemini quota: {message} The inventory and approvals endpoints remain available."
                )
            else:
                st.error(f"API error ({response.status_code if response else 'unknown'}): {message}")
        except requests.RequestException as exc:
            st.error(f"Could not connect to the API: {exc}")

with inventory_tab:
    query = st.text_input("Filter inventory by SKU, product, or supplier")
    try:
        response = requests.get(
            f"{API_BASE}/inventory",
            params={"search": query} if query else {},
            timeout=15,
        )
        response.raise_for_status()
        st.dataframe(response.json(), use_container_width=True, hide_index=True)
    except requests.RequestException as exc:
        st.error(f"Could not load inventory: {exc}")

with approval_tab:
    try:
        response = requests.get(f"{API_BASE}/approvals", timeout=15)
        response.raise_for_status()
        pending_orders = response.json()
        if not pending_orders:
            st.info("No purchase orders are awaiting approval.")
        for order in pending_orders:
            with st.container(border=True):
                st.markdown(
                    f"**{order['order_id']}** · {order['supplier_name']} · "
                    f"${order['total_cost_cents'] / 100:.2f}"
                )
                left, right = st.columns(2)
                for column, decision in ((left, "YES"), (right, "NO")):
                    if column.button(decision, key=f"{order['order_id']}-{decision}"):
                        result = requests.post(
                            f"{API_BASE}/approvals/{order['order_id']}",
                            json={"decision": decision, "thread_id": order["thread_id"]},
                            timeout=30,
                        )
                        if result.ok:
                            st.success(result.json()["response"])
                            st.rerun()
                        else:
                            st.error(f"Approval failed ({result.status_code}): {result.text}")
    except requests.RequestException as exc:
        st.error(f"Could not load pending approvals: {exc}")

with trace_tab:
    try:
        trace = requests.get(
            f"{API_BASE}/logs",
            params={"thread_id": st.session_state.get("thread_id")} if st.session_state.get("thread_id") else {},
            timeout=15,
        )
        trace.raise_for_status()
        for item in trace.json():
            st.markdown(f"**{item['created_at']} · {item['agent']} / {item['event']}**")
            st.json(item["details"])
    except requests.RequestException as exc:
        st.error(f"Could not load agent trace: {exc}")
