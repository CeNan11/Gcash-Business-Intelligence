# Implementation Plan: GCash BI Dashboard Upgrade

## Objective

Transform the existing dashboard into a high-end Business Intelligence tool that not only visualizes data but provides **actionable insights**, **smart categorization**, and **predictive foresight**. This aligns with the "Customer Churn Analysis" and "Product Performance" topics of the course.

## 1. Backend Upgrades (`app.py`)

### A. Smart Review Categorization (The "Why" Engine)

Instead of just "Positive/Negative", we will classify reviews into business-relevant categories using keyword analysis.

- **Categories**:
  - 🛠️ **Technical Stability** (Crash, lag, slow, bug, error)
  - 💸 **Transactions & Money** (Send money, cash in, load, refund, transfer)
  - 🔒 **Security & Account** (OTP, login, verify, mpin, blocked, hack)
  - 📞 **Customer Service** (Support, agent, ticket, help, reply)
  - 📱 **User Experience** (UI, design, easy, confusing, update)

### B. Automated Insights Engine

A new logic layer that scans the data and generates human-readable bullet points.

- **Trend Detection**: "Sentiment has dropped 12% in the last 30 days."
- **Spike Alert**: "Unusual spike in 'Security' related complaints detected in Version 5.45."
- **Success Spotting**: "Version 5.42 had the highest user retention score."

### C. Enhanced Prediction

- Refine the Linear Regression to be more robust.
- Add a "Confidence" metric (e.g., "High Confidence" if the trend is stable).

## 2. Frontend Upgrades (`templates/index.html`)

### A. "Executive Summary" Section

- Replace the simple "Dashboard Overview" title with a dynamic **Insights Panel**.
- Display the top 3 most critical findings (e.g., "⚠️ Urgent: Login issues increasing").

### B. Topic Analysis Visualization

- Add a **Radar Chart** or **Horizontal Bar Chart** showing the distribution of the 5 categories (Technical, Transactions, Security, etc.).
- This allows the business to see _where_ the problem lies immediately.

### C. Visual Polish

- Apply a "Glassmorphism" touch to the cards for a premium feel.
- Improve the color palette to be more "Corporate Tech" (Deep Blues, Slate Greys, Vibrant Alerts).

## 3. Workflow

1.  **Modify `app.py`**: Add categorization logic and the insights generator.
2.  **Update `index.html`**: Implement the new Insights Panel and Topic Chart.
3.  **Verify**: Ensure the app runs and the insights make sense.

---

**Status**: Ready to Execute.
