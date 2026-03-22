"""
CAG Architecture - Main Script with Conversation Memory + HubSpot Integration
Interactive chatbot that remembers your name and conversation history,
then saves everything (name, full discussion, LLM summary) to HubSpot.
"""

import sys
import os
import argparse
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from cag_config import CAGConfig
from cag_system import CAGSystemWithMemory

# ── HubSpot integration ────────────────────────────────────────────────────
try:
    from hubspot_manager import get_hubspot_manager
    HUBSPOT_ENABLED = True
except ImportError:
    HUBSPOT_ENABLED = False
    print("⚠️  hubspot_manager not found – HubSpot saving disabled")


def print_banner():
    print("\n" + "=" * 80)
    print("🤖 CAG CHATBOT WITH MEMORY + HUBSPOT")
    print("   Personal AI Assistant that remembers you!")
    print("   💭 Conversation Memory | 👤 User Profiles | 🎯 Context Aware | 📊 HubSpot CRM")
    print("=" * 80)


def print_section_header(title: str):
    print("\n" + "=" * 80)
    print(f"📋 {title}")
    print("=" * 80)


# ── HubSpot helpers ────────────────────────────────────────────────────────

def hubspot_start(hubspot):
    """Start a new HubSpot session (safe no-op if hubspot is None)."""
    if hubspot:
        hubspot.start_session()


def hubspot_add(hubspot, speaker_id: str, speaker_name: str, text: str):
    """Add one utterance to HubSpot session."""
    if hubspot:
        hubspot.add_utterance(speaker_id, speaker_name, text)


def hubspot_finish(hubspot, cag_system: "CAGSystemWithMemory"):
    """
    Finalise and push the session to HubSpot.
    Pulls the user name and LLM summary from cag_system automatically.

    Note: HubSpotManager.set_user_name() is idempotent — it skips the
    contact lookup if the name and contact_id are already set, so calling
    it here a second time will never create a duplicate contact.
    """
    if not hubspot:
        return

    # ── 1. User name ────────────────────────────────────────────────────
    user_name = getattr(cag_system.memory.user_profile, "name", None)
    if user_name:
        hubspot.set_user_name(user_name)  # no-op if already resolved this session

    # ── 2. LLM summary ──────────────────────────────────────────────────
    print("\n🧠 Generating LLM session summary for HubSpot...")
    try:
        summary_result = cag_system.generate_session_summary()
        llm_summary = summary_result.get("summary", "")
        # Enrich with the name the LLM itself detected (may differ)
        llm_name = summary_result.get("llm_name") or summary_result.get("user_name")
        if llm_name and not user_name:
            hubspot.set_user_name(llm_name)
        if llm_summary:
            hubspot.set_llm_summary(llm_summary)
            print(f"✅ LLM summary ready ({len(llm_summary)} chars)")
    except Exception as exc:
        print(f"⚠️  Could not generate LLM summary: {exc}")

    # ── 3. Save everything ───────────────────────────────────────────────
    hubspot.end_session()


# ── Interactive mode ───────────────────────────────────────────────────────

def interactive_mode(cag_system: CAGSystemWithMemory, use_streaming: bool = True,
                     hubspot=None):
    mode_name = "Streaming" if use_streaming else "Batch"
    print_section_header(f"INTERACTIVE CHATBOT MODE ({mode_name})")

    stats = cag_system.get_stats()

    print("\n" + "=" * 80)
    if stats["memory"]["user_name"]:
        print(f"👋 Welcome back, {stats['memory']['user_name']}!")
        print(f"   We've had {stats['memory']['total_interactions']} conversations together")
    else:
        print("👋 Welcome! I'm your personal AI assistant.")
        print("   I'll remember our conversations and get to know you better over time.")
    print("=" * 80)

    print("\n💡 FEATURES:")
    print("   ✓ I'll remember your name across sessions")
    print("   ✓ I keep track of our conversation")
    print("   ✓ I provide personalized responses")
    if hubspot:
        print("   ✓ Conversation automatically saved to HubSpot CRM")

    print("\n📝 COMMANDS:")
    print("   • Just type your question and press Enter")
    print("   • 'stats'  - Show conversation statistics")
    print("   • 'reset'  - Clear conversation (keeps your name)")
    print("   • 'forget' - Forget everything (including your name)")
    print("   • 'whoami' - Show your profile")
    print("   • 'help'   - Show this help")
    print("   • 'quit'   - Exit")
    print("=" * 80)

    # Start HubSpot session
    hubspot_start(hubspot)

    while True:
        try:
            user_name = cag_system.memory.user_profile.name
            prompt = f"\n{user_name}: " if user_name else "\nYou: "
            query = input(prompt).strip()

            if not query:
                continue

            # ── Commands ────────────────────────────────────────────────
            if query.lower() in {"exit", "quit", "q", "bye"}:
                if user_name:
                    print(f"\n👋 Goodbye, {user_name}! See you next time!")
                else:
                    print("\n👋 Goodbye! Come back anytime!")
                break

            if query.lower() == "help":
                print("\n📝 AVAILABLE COMMANDS:")
                print("   • Ask me anything!")
                print("   • 'stats'  - Show conversation statistics")
                print("   • 'reset'  - Clear conversation history")
                print("   • 'forget' - Reset everything including your name")
                print("   • 'whoami' - Show your profile")
                print("   • 'quit'   - Exit the chatbot")
                continue

            if query.lower() == "stats":
                show_detailed_stats(cag_system)
                continue

            if query.lower() == "reset":
                cag_system.reset_session()
                print("✅ Conversation cleared (your profile is preserved)")
                # Restart HubSpot session tracking
                hubspot_start(hubspot)
                continue

            if query.lower() == "forget":
                cag_system.reset_all()
                print("✅ All memory cleared (including your profile)")
                print("👋 Nice to meet you! What's your name?")
                hubspot_start(hubspot)
                continue

            if query.lower() == "whoami":
                show_user_profile(cag_system)
                continue

            # ── Track user utterance in HubSpot ─────────────────────────
            display_name = user_name or "User"
            hubspot_add(hubspot, "user", display_name, query)

            # ── Process query ────────────────────────────────────────────
            print("🤖 Assistant: ", end="", flush=True)
            response_text = ""

            if use_streaming:
                stream_error = None
                try:
                    for token in cag_system.stream_query(query):
                        print(token, end="", flush=True)
                        response_text += token
                    print()
                except Exception as e:
                    stream_error = e
                    print(f"\n❌ Streaming error: {e}")
                # If streaming produced nothing, fall back to batch mode
                if stream_error and not response_text:
                    try:
                        result = cag_system.query(query)
                        if result["success"]:
                            response_text = result["answer"]
                            print(f"🤖 (batch fallback): {response_text}")
                    except Exception as e2:
                        print(f"❌ Batch fallback also failed: {e2}")
            else:
                result = cag_system.query(query)
                if result["success"]:
                    response_text = result["answer"]
                    print(response_text)
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")

            # ── Track AI utterance in HubSpot ────────────────────────────
            if response_text:
                hubspot_add(hubspot, "ai", "AI Assistant", response_text)

        except KeyboardInterrupt:
            user_name = cag_system.memory.user_profile.name
            print(f"\n\n👋 Goodbye{', ' + user_name if user_name else ''}!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

    # ── End of conversation: push to HubSpot ────────────────────────────
    hubspot_finish(hubspot, cag_system)


# ── Supporting display functions ───────────────────────────────────────────

def show_session_summary(cag_system: CAGSystemWithMemory):
    print("\n" + "=" * 80)
    print("🧠 SESSION SUMMARY (from LLM memory)")
    print("=" * 80)
    print("⏳ Generating summary...", end="", flush=True)

    result = cag_system.generate_session_summary()
    print("\r" + " " * 40 + "\r", end="")

    user_name = result.get("user_name") or result.get("llm_name") or "Unknown"
    summary = result.get("summary", "No summary available.")

    print(f"\n👤 User Name  : {user_name}")
    print(f"📝 Summary    : {summary}")
    print("=" * 80)


def show_user_profile(cag_system: CAGSystemWithMemory):
    profile = cag_system.memory.user_profile
    print("\n" + "=" * 80)
    print("👤 YOUR PROFILE")
    print("=" * 80)
    print(f"\n📛 Name: {profile.name or 'Not set yet'}")
    print(f"📅 First conversation: {profile.first_interaction[:10]}")
    print(f"🕐 Last conversation: {profile.last_interaction[:10]}")
    print(f"💬 Total interactions: {profile.total_interactions}")
    if profile.preferences:
        print(f"\n⚙️  Preferences:")
        for key, value in profile.preferences.items():
            print(f"   • {key}: {value}")
    print("=" * 80)


def show_detailed_stats(cag_system: CAGSystemWithMemory):
    print_section_header("SYSTEM STATISTICS")
    stats = cag_system.get_stats()

    print(f"\n📚 Knowledge Base:")
    print(f"   • Total entries: {stats['knowledge']['entries']:,}")
    print(f"   • Tokens: {stats['knowledge']['tokens']:,}")

    print(f"\n🎯 Cache:")
    print(f"   • Initialized: {stats['cache']['initialized']}")
    print(f"   • Knowledge tokens: {stats['cache']['knowledge_tokens']:,}")

    if stats.get("memory"):
        mem = stats["memory"]
        print(f"\n💭 Conversation Memory:")
        print(f"   • User name: {mem['user_name'] or 'Not set'}")
        print(f"   • Messages in history: {mem['total_messages']}")
        print(f"   • Total interactions: {mem['total_interactions']}")
        print(f"   • First interaction: {mem['first_interaction'][:10]}")
        print(f"   • Last interaction: {mem['last_interaction'][:10]}")

    print(f"\n💬 Inference:")
    print(f"   • Total queries: {stats['total_queries']}")
    print(f"   • Session mode: {stats.get('session_mode', 'unknown')}")
    print(f"   • Max new tokens: {stats['config']['max_new_tokens']}")

    if stats.get("gpu_memory"):
        gpu = stats["gpu_memory"]
        print(f"\n🖥️  GPU Memory:")
        print(f"   • Total: {gpu['total_mb']:,}MB")
        print(f"   • Used: {gpu['used_mb']:,}MB ({gpu['utilization']:.1f}%)")
        print(f"   • Free: {gpu['free_mb']:,}MB")

    print("=" * 80)


def run_demo(cag_system: CAGSystemWithMemory, hubspot=None):
    print_section_header("CONVERSATION MEMORY DEMO")
    print("\n🎬 This demo shows: name detection, memory, context, and HubSpot saving.")
    print("=" * 80)

    hubspot_start(hubspot)

    demo_conversations = [
        ("Hello!", "First greeting"),
        ("My name is Alex", "Providing name"),
        ("What are your shipping options?", "Regular query"),
        ("What about returns?", "Follow-up query"),
    ]

    for i, (query, description) in enumerate(demo_conversations, 1):
        print(f"\n{'─' * 80}")
        print(f"📝 Step {i}: {description}")
        print(f"{'─' * 80}")
        print(f"\n👤 User: {query}")
        hubspot_add(hubspot, "user", "Demo User", query)

        print("🤖 Assistant: ", end="", flush=True)
        result = cag_system.query(query)
        if result["success"]:
            print(result["answer"])
            hubspot_add(hubspot, "ai", "AI Assistant", result["answer"])
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")

        input("\nPress Enter to continue...")

    print(f"\n{'=' * 80}")
    print("✅ Demo complete!")
    print("=" * 80)

    hubspot_finish(hubspot, cag_system)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CAG Chatbot with Conversation Memory + HubSpot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cag_main_with_memory.py                # Interactive mode (streaming)
  python cag_main_with_memory.py --no-stream    # Interactive mode (batch)
  python cag_main_with_memory.py --demo         # Run automated demo
  python cag_main_with_memory.py --reset-memory # Clear all saved memory
  python cag_main_with_memory.py --no-hubspot   # Disable HubSpot saving
        """,
    )

    parser.add_argument("--max-context", type=int, default=5000)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--reset-memory", action="store_true")
    parser.add_argument("--no-hubspot", action="store_true",
                        help="Disable HubSpot CRM saving")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    print_banner()

    # ── HubSpot setup ────────────────────────────────────────────────────
    hubspot = None
    if HUBSPOT_ENABLED and not args.no_hubspot:
        print("\n🔗 Connecting to HubSpot CRM...")
        hubspot = get_hubspot_manager()
        if hubspot:
            print("✅ HubSpot CRM connected – session will be saved automatically")
        else:
            print("⚠️  HubSpot unavailable – continuing without CRM saving")
    else:
        print("ℹ️  HubSpot saving disabled")

    # ── CAG config ───────────────────────────────────────────────────────
    config = CAGConfig(
        max_context_tokens=args.max_context,
        max_new_tokens=args.max_new,
        enable_cache_persistence=True,
        enable_conversation_memory=True,
        verbose=True,
    )

    print(f"\n⚙️  Configuration:")
    print(f"   • Max context tokens: {config.max_context_tokens:,}")
    print(f"   • Max new tokens: {config.max_new_tokens}")
    print(f"   • Conversation memory: Enabled")
    print(f"   • Streaming mode: {not args.no_stream}")
    print(f"   • HubSpot saving: {'Enabled' if hubspot else 'Disabled'}")

    try:
        print_section_header("SYSTEM INITIALIZATION")
        cag_system = CAGSystemWithMemory(config)

        if args.reset_memory:
            print("\n🗑️  Resetting all conversation memory...")
            cag_system.memory.reset_all()
            cag_system.memory.save_memory()
            print("✅ Memory reset complete")
            return

        cag_system.initialize(force_cache_rebuild=args.rebuild_cache)

        print("\n" + "─" * 80)
        show_detailed_stats(cag_system)

        if args.demo:
            run_demo(cag_system, hubspot=hubspot)
        else:
            interactive_mode(cag_system, use_streaming=not args.no_stream,
                             hubspot=hubspot)

        # Session summary (console display)
        show_session_summary(cag_system)

        print("\n" + "─" * 80)
        show_detailed_stats(cag_system)

        print("\n🧹 Cleaning up...")
        cag_system.cleanup()
        print("✅ Cleanup complete")

        print("\n" + "=" * 80)
        print("✅ SESSION COMPLETE")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()