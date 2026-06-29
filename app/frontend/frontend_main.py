"""
Support Ticket Priority Classifier - Streamlit Frontend
Minimalist form-based UI for support agents to classify tickets in real-time.
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Optional, Dict

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎟️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# STYLING & CONSTANTS
# ============================================================================

API_BASE_URL = "http://localhost:8000"  # Change if running on different host
PRIORITY_COLORS = {
    "Low": "#10b981",      # Green
    "Medium": "#3b82f6",   # Blue
    "High": "#ef4444",     # Red
    "Critical": "#f59e0b"  # Amber
}

PRIORITY_EMOJI = {
    "Low": "✅",
    "Medium": "⚠️",
    "High": "🔴",
    "Critical": "🚨"
}

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "ticket_history" not in st.session_state:
    st.session_state.ticket_history = []

if "api_status" not in st.session_state:
    st.session_state.api_status = "unknown"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_api_health() -> bool:
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def classify_ticket(subject: str, body: str) -> Optional[Dict]:
    """
    Send ticket to API for classification.
    
    Args:
        subject: Ticket subject line
        body: Ticket description/body
        
    Returns:
        Dictionary with priority, confidence, department, or None if failed
    """
    try:
        payload = {
            "subject": subject,
            "body": body
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("❌ API request timed out. Check if backend is running.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_BASE_URL}. Is the backend running?")
        return None
    except Exception as e:
        st.error(f"❌ Error classifying ticket: {str(e)}")
        return None


def format_confidence(confidence_dict: Dict[str, float]) -> str:
    """Format confidence scores for display."""
    scores = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
    return " | ".join([f"{name}: {pct*100:.0f}%" for name, pct in scores])


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("## 🎟️ Support Ticket Priority Classifier")
    st.markdown("Automatically classify incoming support tickets into priority tiers")
    
    # API Health Check (sidebar)
    with st.sidebar:
        st.markdown("### System Status")
        if check_api_health():
            st.success("✅ Backend API: Connected")
            st.session_state.api_status = "connected"
        else:
            st.error("❌ Backend API: Offline")
            st.markdown(
                "**Troubleshooting:**\n"
                "1. Start the FastAPI backend: `python app/api/main.py`\n"
                "2. Ensure it's running on `http://localhost:8000`\n"
                "3. Check that the model is loaded correctly"
            )
            st.session_state.api_status = "offline"
    
    # Main form
    st.markdown("---")
    st.markdown("### Classify a Ticket")
    
    # Input fields
    col1, col2 = st.columns([1, 1])
    
    with col1:
        subject = st.text_input(
            label="Subject",
            placeholder="e.g., Cannot login to account",
            max_chars=200,
            help="Brief subject line of the ticket"
        )
    
    with col2:
        ticket_type = st.selectbox(
            label="Ticket Type",
            options=[
                "Technical issue",
                "Billing inquiry",
                "Refund request",
                "Cancellation request",
                "Product inquiry"
            ],
            help="Category of the ticket (for department routing)"
        )
    
    body = st.text_area(
        label="Ticket Description",
        placeholder="Paste the customer's message here...",
        height=150,
        max_chars=3000,
        help="Full ticket description or customer message"
    )
    
    # Character count
    char_count = len(body)
    st.caption(f"📝 {char_count} / 3000 characters")
    
    # Classify button
    st.markdown("---")
    classify_button = st.button(
        label="🚀 Classify Ticket",
        use_container_width=True,
        type="primary"
    )
    
    # Process classification
    if classify_button:
        if not subject or not body:
            st.warning("⚠️ Please fill in both Subject and Description fields")
        elif st.session_state.api_status == "offline":
            st.error("❌ Backend API is offline. Cannot classify ticket.")
        else:
            with st.spinner("🔄 Classifying ticket..."):
                result = classify_ticket(subject, body)
            
            if result:
                # Add to history
                st.session_state.ticket_history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "subject": subject,
                    "priority": result["priority"],
                    "confidence": result["confidence"],
                    "department": result["department"]
                })
                
                # Display results
                st.markdown("---")
                st.markdown("### ✅ Classification Results")
                
                priority = result["priority"]
                confidence_max = max(result["confidence"].values())
                department = result["department"]
                
                # Three-column layout for main metrics
                col1, col2, col3 = st.columns(3, gap="large")
                
                with col1:
                    st.metric(
                        label="Priority",
                        value=f"{PRIORITY_EMOJI.get(priority, '')} {priority}",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        label="Confidence",
                        value=f"{confidence_max*100:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        label="Department",
                        value=department
                    )
                
                # Confidence breakdown
                st.markdown("#### Confidence Breakdown")
                confidence_data = result["confidence"]
                
                # Create visual bars
                col_left, col_right = st.columns([1, 2])
                
                with col_left:
                    st.markdown("**Priority Level**")
                    for level in ["Critical", "High", "Medium", "Low"]:
                        pct = confidence_data.get(level, 0)
                        st.markdown(f"- {level}")
                
                with col_right:
                    st.markdown("**Score**")
                    for level in ["Critical", "High", "Medium", "Low"]:
                        pct = confidence_data.get(level, 0)
                        bar_length = int(pct * 30)
                        bar = "█" * bar_length + "░" * (30 - bar_length)
                        st.markdown(f"`{bar}` {pct*100:.1f}%")
                
                # Action buttons
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📋 Copy to clipboard", use_container_width=True):
                        st.success("✅ Copied!")
                
                with col2:
                    if st.button("🔄 Classify another", use_container_width=True):
                        st.rerun()
                
                with col3:
                    st.caption("💾 History saved below")
    
    # Ticket history sidebar
    if st.session_state.ticket_history:
        st.markdown("---")
        st.markdown("### 📊 Classification History")
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        history = st.session_state.ticket_history
        priorities = [t["priority"] for t in history]
        
        col1.metric("Total Classified", len(history))
        col2.metric("Critical", priorities.count("Critical"))
        col3.metric("High", priorities.count("High"))
        col4.metric("Avg Confidence", f"{sum(max(t['confidence'].values()) for t in history) / len(history) * 100:.0f}%")
        
        st.markdown("#### Recent Classifications")
        
        # Display history as expandable items
        for i, ticket in enumerate(reversed(history)):
            with st.expander(
                label=f"[{ticket['timestamp']}] {ticket['subject'][:50]}... → {PRIORITY_EMOJI.get(ticket['priority'], '')} {ticket['priority']}"
            ):
                st.markdown(f"**Subject:** {ticket['subject']}")
                st.markdown(f"**Priority:** {ticket['priority']}")
                st.markdown(f"**Department:** {ticket['department']}")
                st.markdown(f"**Confidence:** {format_confidence(ticket['confidence'])}")
        
        # Clear history button
        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state.ticket_history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 12px;'>"
        "Support Ticket Priority Classifier v1.0 | "
        "Built with Streamlit + FastAPI + Bi-LSTM"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
