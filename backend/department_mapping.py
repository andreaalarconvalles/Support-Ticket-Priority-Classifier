"""Ticket Type -> department lookup for the API contract's `department` field.

Decision (Phase 3 / API contract): the dataset has no real department column, and
Ticket Type is already known per ticket (not something that needs inferring from
text), so department is a plain lookup rather than a second model output. The
Bi-LSTM only needs to predict `priority`.
"""

TICKET_TYPE_TO_DEPARTMENT = {
    "Technical issue": "Technical Support",
    "Billing inquiry": "Billing",
    "Refund request": "Billing",
    "Cancellation request": "Customer Retention",
    "Product inquiry": "Sales",
}


def get_department(ticket_type: str) -> str:
    """Map a Ticket Type value to its department, defaulting to General Support."""
    return TICKET_TYPE_TO_DEPARTMENT.get(ticket_type, "General Support")
