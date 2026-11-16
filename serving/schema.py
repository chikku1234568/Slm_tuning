"""Shared constants for serving: the system prompt, JSON schema and field list.

These are copied *exactly* from SLM_Narration_to_JSON_Colab_Smoke.ipynb.
Why it matters: the model learned to answer this exact system prompt during
fine-tuning. If serving sends a different prompt, quality drops. If you change
the prompt or schema in your training notebook, change it here too.
"""

# JSON schema for one narration line (10 fields). Used two ways:
#   1. sent to vLLM for structured (guided) decoding -> output is always valid JSON
#   2. checked again client-side with `jsonschema` as a safety net
SCHEMA = {"type": "object", "additionalProperties": False,
 "required": ["category", "sub_category", "counterparty_name", "counterparty_type", "is_emi", "lender_name",
              "is_bounce_or_return", "is_reversal", "p2p_or_p2m", "confidence"],
 "properties": {
  "category": {"type": "string"}, "sub_category": {"type": "string"},
  "counterparty_name": {"type": ["string", "null"]},
  "counterparty_type": {"enum": ["merchant", "individual", "lender", "employer", "bank", "unknown"]},
  "is_emi": {"type": "boolean"}, "lender_name": {"type": ["string", "null"]},
  "is_bounce_or_return": {"type": "boolean"}, "is_reversal": {"type": "boolean"},
  "p2p_or_p2m": {"enum": ["P2P", "P2M", "NA"]},
  "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}

FIELDS = list(SCHEMA["required"])

# Exact system prompt used in training (same string for every request ->
# vLLM's prefix cache can reuse its KV cache across requests).
SYS = 'Convert the bank statement narration into JSON with keys: ' + ', '.join(FIELDS) + '. Output JSON only.'

# A few sample statement narrations for demos (synthetic, same style as training data).
SAMPLES = [
    "UPI/P2M/412345678901/SWIGGY/PAYMENT FROM PHONE/YESB",
    "UPI/P2A/498765432109/RAHUL SHARMA/UPI/SBIN",
    "NACH-DR-BAJAJFIN-1234567890",
    "ACH RTN CHG TATACAP 12345678",
    "REV-IMPS-123456789012-FAILED TXN",
    "ATM WDL/123456/MUMBAI ANDHERI",
    "NEFT CR-HDFC0000123-INFOSYS LTD-SALARY JAN",
    "RTGS/UTIB1234567890/TCS LTD/SAL",
    "ECS/IDFCFIRST/LOAN EMI 654321",
    "MMT/IMPS/123456789012/PRIYA N/KKBK",
]
