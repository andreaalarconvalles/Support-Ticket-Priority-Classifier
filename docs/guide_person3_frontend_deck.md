# 🎫 Deep Learning Project — Your Guide
## Role: Frontend & Presentation Lead
**Project:** Support Ticket Priority Classifier | IE University Deep Learning Final Project

---

> **Your job in one sentence:** Build the Streamlit interface that any non-technical person can use to classify a support ticket, and own the presentation deck from slide one to demo moment.

---

## ⚠️ Read This First

You work in parallel with the rest of the team from Day 1. You do **not** need to wait for the model or the API to start building. You build against a mock API response from Day 1, and swap it for the real one on Day 3.

The JSON the API will always return looks like this — memorise this structure, your entire frontend is built around it:

```json
{
  "priority": "High",
  "confidence": {
    "Low": 0.05,
    "Medium": 0.12,
    "High": 0.78,
    "Critical": 0.05
  },
  "department": "Technical Support"
}
```

---

## 📦 What You Need to Install

```bash
pip install streamlit requests pandas
```

That is genuinely all you need. No TensorFlow, no Keras, no ML libraries.

---

---

# DAY 1 — Build the Streamlit Shell + Presentation Outline

## Step 1 — Create the App Structure

Create `/frontend/app.py`. This is your entire frontend in one file.

Start with a **hardcoded mock** — no API call yet. This lets you design and polish the UI completely independently.

```python
# frontend/app.py

import streamlit as st
import requests

# ─── CONFIG ───────────────────────────────────────────────────────
API_URL = "http://localhost:8000/predict"
USE_MOCK = True   # Set to False on Day 3 when API is ready

PRIORITY_CONFIG = {
    "Low":      {"color": "#6B7280", "bg": "#F3F4F6", "emoji": "🟢", "label": "P4 — Low"},
    "Medium":   {"color": "#D97706", "bg": "#FFFBEB", "emoji": "🟡", "label": "P3 — Medium"},
    "High":     {"color": "#EA580C", "bg": "#FFF7ED", "emoji": "🟠", "label": "P2 — High"},
    "Critical": {"color": "#DC2626", "bg": "#FEF2F2", "emoji": "🔴", "label": "P1 — Critical"},
}

# ─── MOCK FUNCTION (remove on Day 3) ─────────────────────────────
def mock_predict(subject, body):
    """Simulates an API response for UI development."""
    text = (subject + body).lower()
    if any(w in text for w in ["down", "urgent", "critical", "outage", "revenue"]):
        priority = "Critical"
        probs = {"Low": 0.02, "Medium": 0.05, "High": 0.11, "Critical": 0.82}
    elif any(w in text for w in ["error", "broken", "failing", "issue"]):
        priority = "High"
        probs = {"Low": 0.05, "Medium": 0.15, "High": 0.72, "Critical": 0.08}
    elif any(w in text for w in ["slow", "help", "question", "how"]):
        priority = "Medium"
        probs = {"Low": 0.10, "Medium": 0.65, "High": 0.20, "Critical": 0.05}
    else:
        priority = "Low"
        probs = {"Low": 0.75, "Medium": 0.18, "High": 0.05, "Critical": 0.02}
    
    dept_map = {
        "Low": "General Support",
        "Medium": "Customer Service",
        "High": "Technical Support",
        "Critical": "Incident Response"
    }
    return {"priority": priority, "confidence": probs, "department": dept_map[priority]}

# ─── REAL API FUNCTION ─────────────────────────────────────────────
def call_api(subject, body):
    try:
        response = requests.post(
            API_URL,
            json={"subject": subject, "body": body},
            timeout=10
        )
        return response.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None

# ─── PAGE LAYOUT ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)

# Header
st.title("🎫 Support Ticket Priority Classifier")
st.markdown("Paste a support ticket below. The model will classify its priority and route it to the right team.")
st.divider()

# Input fields
col1, col2 = st.columns([1, 1])
with col1:
    subject = st.text_input(
        "Ticket Subject",
        placeholder="e.g. Production database completely down"
    )
with col2:
    st.write("")  # spacing

body = st.text_area(
    "Ticket Body",
    placeholder="Describe the issue in detail...",
    height=150
)

# Submit button
submitted = st.button("🔍 Classify Ticket", type="primary", use_container_width=True)

# ─── RESULT SECTION ────────────────────────────────────────────────
if submitted:
    if not subject and not body:
        st.warning("Please enter a subject or body before classifying.")
    else:
        with st.spinner("Classifying..."):
            if USE_MOCK:
                result = mock_predict(subject, body)
            else:
                result = call_api(subject, body)
        
        if result:
            priority = result["priority"]
            confidence = result["confidence"]
            department = result["department"]
            cfg = PRIORITY_CONFIG[priority]
            top_confidence = confidence[priority]
            
            st.divider()
            
            # Priority badge
            st.markdown(
                f"""
                <div style="
                    background: {cfg['bg']};
                    border: 2px solid {cfg['color']};
                    border-radius: 12px;
                    padding: 20px 24px;
                    text-align: center;
                    margin-bottom: 16px;
                ">
                    <div style="font-size: 36px">{cfg['emoji']}</div>
                    <div style="font-size: 28px; font-weight: 800; color: {cfg['color']}">
                        {cfg['label']}
                    </div>
                    <div style="font-size: 14px; color: #6B7280; margin-top: 4px">
                        {top_confidence*100:.1f}% confidence · Routed to: <strong>{department}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Confidence breakdown
            st.markdown("**Confidence across all classes:**")
            for p_name, p_val in sorted(confidence.items(), key=lambda x: x[1], reverse=True):
                p_cfg = PRIORITY_CONFIG[p_name]
                col_label, col_bar = st.columns([1, 4])
                with col_label:
                    st.markdown(f"<span style='color:{p_cfg['color']}'>{p_name}</span>",
                                unsafe_allow_html=True)
                with col_bar:
                    st.progress(p_val)
```

Run it:

```bash
cd frontend
streamlit run app.py
```

You should see the UI in your browser at `http://localhost:8501`. Test it with different subjects to see the mock responses change.

---

## Step 2 — Add the Demo Ticket Buttons

Add this section to the app so during the live demo you can one-click load pre-written tickets instead of typing under pressure.

Add this **above** the input fields:

```python
st.markdown("#### 💡 Try a demo ticket:")
demo_col1, demo_col2, demo_col3, demo_col4 = st.columns(4)

DEMO_TICKETS = {
    "🔴 Critical": {
        "subject": "URGENT: Production database completely down",
        "body": "All users are locked out of the platform. We cannot process payments. This started 15 minutes ago and is affecting all regions. We are losing approximately €5,000 per minute."
    },
    "🟠 High": {
        "subject": "Login page throwing 500 errors for 30% of users",
        "body": "Since the last deployment at 09:00, roughly a third of our users are getting 500 errors on the login screen. Other pages seem fine. Affecting EU region only."
    },
    "🟡 Medium": {
        "subject": "Dashboard charts not loading correctly",
        "body": "The revenue chart on the main dashboard is showing last month's data instead of current month. Users have noticed. Not urgent but needs fixing before end of week."
    },
    "🟢 Low": {
        "subject": "How do I change my notification settings?",
        "body": "Hi, I would like to stop receiving the weekly summary email. I have looked in my account settings but cannot find where to change this. Could you help?"
    },
}

selected_demo = None
with demo_col1:
    if st.button("🔴 Critical", use_container_width=True):
        selected_demo = "🔴 Critical"
with demo_col2:
    if st.button("🟠 High", use_container_width=True):
        selected_demo = "🟠 High"
with demo_col3:
    if st.button("🟡 Medium", use_container_width=True):
        selected_demo = "🟡 Medium"
with demo_col4:
    if st.button("🟢 Low", use_container_width=True):
        selected_demo = "🟢 Low"

if selected_demo:
    ticket = DEMO_TICKETS[selected_demo]
    subject = ticket["subject"]
    body = ticket["body"]
```

> **Note on Streamlit state:** You may need to use `st.session_state` to make the demo buttons pre-fill the text inputs. If the buttons don't pre-fill, ask Juan José for 10 minutes of help — it's a 5-line fix.

---

## Step 3 — Start the Presentation Deck

Open Google Slides or PowerPoint. Create these slides in order — fill in what you can, leave placeholders for numbers that come from Juan José on Day 3.

**Slide 1 — Title**
"AI-Powered Support Ticket Triage" | Group name | IE University Deep Learning

**Slide 2 — The Business Problem (your strongest slide)**
- Companies receive thousands of support tickets daily
- Manual triage takes 2–5 minutes per ticket
- SLA breaches cost thousands in penalties per incident
- Misrouted tickets waste 10–20 minutes each
- Use a real stat: *"A team handling 1,000 tickets/day spends 50+ hours just sorting, not solving"*

**Slide 3 — Our Solution**
One sentence: "A Bi-LSTM model that reads incoming ticket text and classifies it by priority in under one second."
Include a simple flow diagram: Ticket Text → Bi-LSTM Model → Priority Badge + Routing

**Slide 4 — Architecture (Juan José fills this in)**
Leave placeholders for: model diagram, layer sizes, why Bidirectional LSTM, GloVe vs trained embeddings

**Slide 5 — Dataset**
- Source: Kaggle Customer Support Ticket Dataset (~8.5K tickets)
- 4 priority classes: Low / Medium / High / Critical
- Include the class distribution bar chart from Person 2's EDA

**Slide 6 — Results (Juan José fills in Day 3)**
Placeholders for: Accuracy, Macro F1-Score, Confusion Matrix image

**Slide 7 — Live Demo**
Just a title slide that says "Live Demo" — this is where you switch to the Streamlit app

**Slide 8 — Business Impact**
Quantify: time saved per ticket, annual cost saving for a 1,000 ticket/day operation, SLA compliance improvement

**Slide 9 — Thank You + Q&A**

---

# DAY 2 — Polish UI + Draft Presentation

- Polish the Streamlit styling — make the priority badges look clean and professional
- Add a footer to the app: "IE University · Deep Learning Final Project · 2026"
- Draft the narration script for each slide (who says what)
- Make sure every team member has speaking parts — this is 20% of your grade
- Prepare 2 backup screenshots of the app working in case of technical issues on presentation day

---

# DAY 3 — Connect Real API + Final Deck

## Switching from Mock to Real API

When Person 2 confirms the API is running, do just two things in `app.py`:

```python
USE_MOCK = False   # Line 10 — change this
```

Restart Streamlit. Test all four demo buttons. Verify the results make sense (Critical ticket → Critical label).

If any demo ticket returns a wrong priority, do not panic. Note it, and prepare a verbal explanation: *"The model occasionally misclassifies ambiguous tickets — in production this would go to a human review queue."* This is honest and shows you understand model limitations.

## Finishing the Deck

Get these from Juan José on Day 3 morning:
- Final accuracy number (currently sitting at **65%**)
- Final Macro F1-Score  
- Confusion matrix image (he'll have this from his evaluation code)

Insert them into Slides 4 and 6. The presentation is now complete.

---

# DAY 4 — Rehearsal

Run the full presentation twice with a timer. Strict 15-minute limit — your professor will cut you off.

Suggested time split:
- Business problem: 2 minutes
- Architecture + dataset: 3 minutes
- Results + metrics: 2 minutes
- Live demo: 5 minutes
- Business impact + Q&A: 3 minutes

**Demo script (do this exactly in this order during the live demo):**
1. Open Streamlit app, show the clean empty interface briefly
2. Click the 🔴 Critical button — let the result appear — pause for effect
3. Click the 🟢 Low button — show the contrast
4. Type a custom ticket live: "The checkout button on our website is completely broken. We cannot process any orders." — let it classify
5. Explain what the confidence bars mean in plain language

**Prepare answers to these likely Q&A questions:**
- *"What happens when the model is wrong?"* → Confidence score below 70% flags for human review. The model is a triage assistant, not a replacement for human judgment.
- *"Why Bi-LSTM and not a Transformer/BERT?"* → Bi-LSTM is within scope for this course (predictive architecture), trains efficiently on our dataset size, and the performance difference for short ticket text is marginal.
- *"How was the dataset labeled?"* → Synthetic dataset generated with priority labels. Real-world deployment would fine-tune on company-specific labeled tickets.

---

## ✅ Your Final Deliverables

By presentation morning, confirm all of these:

- [ ] Streamlit app runs on `streamlit run app.py`
- [ ] All 4 demo buttons work and show correct priority
- [ ] App is connected to real API (USE_MOCK = False)
- [ ] Presentation deck is complete with real metrics
- [ ] Every team member has a speaking script
- [ ] Full 15-minute run-through done at least twice
- [ ] Backup screenshots saved in case of demo failure
- [ ] GitHub README is clean and complete

---

## 🚨 Common Issues and Fixes

**Streamlit won't connect to the API** → Make sure Person 2's FastAPI is running (`uvicorn api:app --port 8000`) before starting Streamlit. Both must run simultaneously in separate terminals.

**Demo buttons don't pre-fill the text inputs** → Use `st.session_state`. Quick fix:
```python
if 'subject' not in st.session_state:
    st.session_state.subject = ""
subject = st.text_input("Subject", key="subject")
# Then when button clicked: st.session_state.subject = demo_ticket["subject"]
```

**Streamlit reloads and loses the result** → This is normal Streamlit behavior. Use `st.session_state` to store the last result and re-display it after reloads.

**API returns 500 error** → Check Person 2's terminal for the error message. Most likely a tokenizer loading issue on their end — not your problem to fix.
