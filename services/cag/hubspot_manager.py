"""
HubSpot Integration Module
Creates a contact dynamically using the name detected from the conversation.
No default contact - contact is only created once we know the user's name.
"""

import os
import json
import time
import io
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class HubSpotManager:

    def __init__(self, access_token=None):
        self.access_token = access_token or os.getenv('HUBSPOT_API_KEY')
        if not self.access_token:
            raise ValueError("❌ HUBSPOT_API_KEY not found in .env file")

        self.base_url = "https://api.hubapi.com"
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

        # Set later, once we know the user's name
        self.contact_id  = None
        self.user_name   = None
        self.user_email  = None   # optional - set via set_user_email()

        self.utterances       = []
        self.session_start_ts = None
        self.start_time       = None
        self.llm_summary      = None

        print("✅ HubSpot Manager initialized (waiting for user name…)")

    # ──────────────────────────────────────────────────────────────────
    # SESSION LIFECYCLE
    # ──────────────────────────────────────────────────────────────────

    def start_session(self):
        """Start a new session. Contact is resolved later when name is known."""
        self.utterances       = []
        self.session_start_ts = time.time()
        self.start_time       = datetime.utcnow()
        self.user_name        = None
        self.user_email       = None
        self.contact_id       = None
        self.llm_summary      = None
        print(f"\n🎙️ New session started at {datetime.now().strftime('%H:%M:%S')}")
        return self

    def set_user_name(self, name: str):
        """
        Called once the chatbot has detected / confirmed the user's name.
        Immediately creates or finds the HubSpot contact.
        """
        self.user_name = name.strip() if name else None
        if not self.user_name:
            return

        print(f"👤 Name detected: '{self.user_name}' — resolving HubSpot contact…")

        # Only use email if a real one was set via set_user_email()
        # Never generate fake placeholder emails — HubSpot rejects them
        self.contact_id = self._get_or_create_contact_by_name(
            self.user_name, self.user_email  # user_email may be None
        )

        if self.contact_id:
            print(f"✅ Contact ready: '{self.user_name}' (id={self.contact_id})")
        else:
            print(f"⚠️  Could not create contact for '{self.user_name}'")

    def set_user_email(self, email: str):
        """Optionally set a real email before or after set_user_name."""
        self.user_email = email.strip() if email else None

    def set_llm_summary(self, summary: str):
        """Store the LLM-generated session summary."""
        self.llm_summary = summary.strip() if summary else None

    def add_utterance(self, speaker_id: str, speaker_name: str, text: str):
        """Add one line of dialogue with auto timestamp."""
        if self.session_start_ts is None:
            self.start_session()

        elapsed_ms = int((time.time() - self.session_start_ts) * 1000)
        utterance = {
            "speaker": {"id": speaker_id, "name": speaker_name},
            "text": text,
            "startTimeMillis": elapsed_ms,
            "endTimeMillis": elapsed_ms + 1000,
        }
        self.utterances.append(utterance)
        return utterance

    def end_session(self, summary_text=None):
        """
        Save everything to HubSpot.
        If no name was ever detected, session is NOT saved.
        """
        if not self.utterances:
            print("📭 No utterances to save — skipping HubSpot.")
            return False

        if not self.user_name:
            print("⚠️  No user name detected — cannot create contact. Session NOT saved.")
            return False

        if not self.contact_id:
            print(f"⚠️  Contact for '{self.user_name}' could not be resolved — session NOT saved.")
            return False

        print(f"\n💾 Saving session to HubSpot…")
        print(f"   👤 Name      : {self.user_name}")
        print(f"   📧 Email     : {self.user_email}")
        print(f"   🆔 Contact ID: {self.contact_id}")
        print(f"   📝 Utterances: {len(self.utterances)}")
        print(f"   🧠 Summary   : {'yes' if self.llm_summary else 'no'}")

        if summary_text is None:
            summary_text = self._build_full_document()

        file_id = self._upload_summary_file(summary_text)
        call_id = self._create_call_engagement(summary_text, file_id)

        if call_id and self._upload_transcript(call_id):
            print(f"✅ Session saved! Call ID: {call_id}")
            return True

        print("❌ Failed to fully save session to HubSpot")
        return False

    # ──────────────────────────────────────────────────────────────────
    # CONTACT HELPERS
    # ──────────────────────────────────────────────────────────────────

    def _get_or_create_contact_by_name(self, full_name: str, email: str = None):
        """Find contact by email or name, or create a new one."""
        parts = full_name.strip().split()
        first = parts[0] if parts else full_name
        last  = " ".join(parts[1:]) if len(parts) > 1 else ""

        # 1. Try email match (only if a real email was provided)
        if email:
            contact_id = self._find_contact_by_email(email)
            if contact_id:
                print(f"   📋 Existing contact found (email match)")
                return contact_id

        # 2. Try name match
        contact_id = self._find_contact_by_name(first, last)
        if contact_id:
            print(f"   📋 Existing contact found (name match)")
            return contact_id

        # 3. Create new — only send email if it's a real one
        return self._create_contact(email, first, last)

    def _find_contact_by_email(self, email: str):
        url = f"{self.base_url}/crm/v3/objects/contacts/search"
        payload = {
            "filterGroups": [{
                "filters": [{"propertyName": "email", "operator": "EQ", "value": email}]
            }]
        }
        try:
            res = requests.post(url, headers=self.headers, json=payload)
            if res.status_code == 200:
                results = res.json().get("results", [])
                if results:
                    return results[0]["id"]
        except Exception as e:
            print(f"⚠️ Email search error: {e}")
        return None

    def _find_contact_by_name(self, first: str, last: str):
        url = f"{self.base_url}/crm/v3/objects/contacts/search"
        filters = [{"propertyName": "firstname", "operator": "EQ", "value": first}]
        if last:
            filters.append({"propertyName": "lastname", "operator": "EQ", "value": last})
        payload = {"filterGroups": [{"filters": filters}]}
        try:
            res = requests.post(url, headers=self.headers, json=payload)
            if res.status_code == 200:
                results = res.json().get("results", [])
                if results:
                    return results[0]["id"]
        except Exception as e:
            print(f"⚠️ Name search error: {e}")
        return None

    def _create_contact(self, email, first: str, last: str):
        url = f"{self.base_url}/crm/v3/objects/contacts"
        props = {"firstname": first}
        if last:
            props["lastname"] = last
        if email:  # only add email if it's a real one
            props["email"] = email
        payload = {"properties": props}
        try:
            res = requests.post(url, headers=self.headers, json=payload)
            if res.status_code == 201:
                cid = res.json().get("id")
                print(f"   ✨ New contact created: {first} {last} (id={cid})")
                return cid
            else:
                print(f"⚠️ Contact creation HTTP {res.status_code}: {res.text[:200]}")
        except Exception as e:
            print(f"❌ Contact creation error: {e}")
        return None

    # ──────────────────────────────────────────────────────────────────
    # DOCUMENT BUILDER
    # ──────────────────────────────────────────────────────────────────

    def _build_full_document(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("CHAT SESSION REPORT")
        lines.append(f"Date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"User    : {self.user_name or 'Unknown'}")
        lines.append(f"Email   : {self.user_email or 'N/A'}")
        duration_s = int(time.time() - self.session_start_ts) if self.session_start_ts else 0
        lines.append(f"Duration: {duration_s // 60:02d}m {duration_s % 60:02d}s")
        lines.append("=" * 60)

        lines.append("")
        lines.append("─── CONVERSATION TRANSCRIPT ───")
        lines.append("")
        for u in self.utterances:
            ts_ms   = u["startTimeMillis"]
            mm      = int(ts_ms // 60000)
            ss      = int((ts_ms % 60000) // 1000)
            speaker = u["speaker"]["name"]
            text    = u["text"]
            lines.append(f"[{mm:02d}:{ss:02d}] {speaker}: {text}")

        lines.append("")
        lines.append("─── AI SESSION SUMMARY ───")
        lines.append("")
        lines.append(self.llm_summary or "(No LLM summary available)")

        lines.append("")
        lines.append("=" * 60)
        lines.append("END OF REPORT")
        lines.append("=" * 60)

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────
    # HUBSPOT API CALLS
    # ──────────────────────────────────────────────────────────────────

    def _upload_summary_file(self, summary_text: str):
        url      = "https://api.hubapi.com/files/v3/files"
        slug     = (self.user_name or "Unknown").replace(" ", "_")
        filename = f"Session_{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        file_data    = io.BytesIO(summary_text.encode("utf-8"))
        options      = {
            "access": "PRIVATE",
            "overwrite": "false",
            "duplicateValidationStrategy": "NONE",
            "duplicateValidationScope": "ENTIRE_PORTAL",
        }
        files = {
            "file":    (filename, file_data, "text/plain"),
            "options": (None, json.dumps(options), "application/json"),
        }
        file_headers = {"Authorization": f"Bearer {self.access_token}"}
        try:
            res = requests.post(url, headers=file_headers, files=files)
            if res.status_code == 201:
                fid = res.json().get("id")
                print(f"📎 File uploaded: {filename} (id={fid})")
                return fid
            else:
                print(f"⚠️ File upload HTTP {res.status_code}: {res.text[:200]}")
        except Exception as e:
            print(f"❌ File upload error: {e}")
        return None

    def _create_call_engagement(self, summary: str, file_id=None):
        url    = f"{self.base_url}/crm/v3/objects/calls"
        APP_ID = "30746066"

        title = f"Chat – {self.user_name} – {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        properties = {
            "hs_call_title":       title,
            "hs_call_direction":   "INBOUND",
            "hs_call_disposition": "f240bbac-87c9-4f6e-bf70-924b57d47db7",
            "hs_timestamp":        datetime.utcnow().isoformat() + "Z",
            "hs_call_body":        summary[:5000],
            "hs_call_app_id":      APP_ID,
            "hs_call_duration": (
                int((time.time() - self.session_start_ts) / 60)
                if self.session_start_ts else 1
            ),
        }
        if file_id:
            properties["hs_attachment_ids"] = str(file_id)

        payload = {
            "properties": properties,
            "associations": [{
                "to":    {"id": self.contact_id},
                "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 194}],
            }],
        }
        try:
            res = requests.post(url, headers=self.headers, json=payload)
            if res.status_code == 201:
                cid = res.json().get("id")
                print(f"📞 Call engagement created (id={cid})")
                return cid
            else:
                print(f"⚠️ Call creation HTTP {res.status_code}: {res.text[:200]}")
        except Exception as e:
            print(f"❌ Call creation error: {e}")
        return None

    def _upload_transcript(self, call_id: str) -> bool:
        url     = f"{self.base_url}/crm/v3/extensions/calling/transcripts"
        payload = {
            "engagementId": call_id,
            "transcriptCreateUtterances": self.utterances,
        }
        try:
            res = requests.post(url, headers=self.headers, json=payload)
            if res.ok:
                print(f"📜 Transcript uploaded ({len(self.utterances)} utterances)")
                return True
            else:
                print(f"⚠️ Transcript HTTP {res.status_code}: {res.text[:200]}")
        except Exception as e:
            print(f"❌ Transcript upload error: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────────────────────────────────

_hubspot_instance = None

def get_hubspot_manager() -> HubSpotManager:
    global _hubspot_instance
    if _hubspot_instance is None:
        try:
            _hubspot_instance = HubSpotManager()
        except Exception as e:
            print(f"❌ Failed to initialize HubSpot: {e}")
            _hubspot_instance = None
    return _hubspot_instance


# ──────────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hubspot = get_hubspot_manager()

    if hubspot:
        print("\n🔍 Testing dynamic contact creation…")

        hubspot.start_session()

        # Messages BEFORE name is known
        hubspot.add_utterance("ai",   "AI Assistant", "Hello! What's your name?")
        hubspot.add_utterance("user", "User",          "My name is Balsem")

        # ← Name detected → contact created NOW
        hubspot.set_user_name("Balsem")

        hubspot.add_utterance("user", "Balsem", "Our website is too slow, losing sales")
        hubspot.add_utterance("ai",   "AI Assistant",
                              "I recommend our Website Performance Optimization package.")

        hubspot.set_llm_summary(
            "Balsem reported website performance issues causing lost sales. "
            "Recommended the Website Performance Optimization package ($499/month)."
        )

        hubspot.end_session()