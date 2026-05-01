"""
CAG Architecture - Fresh Session Mode
Each run is completely independent - no memory between runs
Perfect for: Testing, demos, or when you want fresh start every time
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from cag_config import CAGConfig
from cag_system import CAGSystemFreshSession


def print_banner():
    """Print application banner"""
    print("\n" + "="*80)
    print("🤖 CAG CHATBOT - FRESH SESSION MODE")
    print("   Every run is a new beginning!")
    print("   🔄 No Memory Between Runs | 🆕 Fresh Start Every Time")
    print("="*80)


def print_section_header(title: str):
    """Print a section header"""
    print("\n" + "="*80)
    print(f"📋 {title}")
    print("="*80)


def interactive_mode(cag_system: CAGSystemFreshSession, use_streaming: bool = True):
    """
    Run interactive chatbot mode with fresh session
    """
    mode_name = "Streaming" if use_streaming else "Batch"
    print_section_header(f"INTERACTIVE MODE - FRESH SESSION ({mode_name})")
    
    # Show welcome message
    print("\n" + "="*80)
    print("👋 Welcome! This is a FRESH SESSION")
    print("   • No memory from previous runs")
    print("   • I'll ask for your name again")
    print("   • Memory only lasts during this session")
    print("   • Everything resets when you exit")
    print("="*80)
    
    print("\n💡 FRESH SESSION FEATURES:")
    print("   ✓ Clean slate every time you run")
    print("   ✓ Perfect for testing or demos")
    print("   ✓ No stored data between runs")
    print("   ✓ Conversation memory during current session only")
    
    print("\n📝 COMMANDS:")
    print("   • Just type your question and press Enter")
    print("   • 'stats'     - Show session statistics")
    print("   • 'reset'     - Restart the current session")
    print("   • 'whoami'    - Show current session info")
    print("   • 'help'      - Show this help")
    print("   • 'quit'      - Exit (memory will be cleared)")
    print("="*80)
    
    while True:
        try:
            # Show appropriate prompt
            user_name = cag_system.memory.user_profile.name
            if user_name:
                prompt = f"\n{user_name}: "
            else:
                prompt = "\nYou: "
            
            query = input(prompt).strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.lower() in {'exit', 'quit', 'q', 'bye'}:
                if user_name:
                    print(f"\n👋 Goodbye, {user_name}!")
                else:
                    print("\n👋 Goodbye!")
                print("   Memory from this session will be cleared.")
                break
            
            if query.lower() == 'help':
                print("\n📝 AVAILABLE COMMANDS:")
                print("   • Ask me anything!")
                print("   • 'stats'  - Show session statistics")
                print("   • 'reset'  - Restart current session")
                print("   • 'whoami' - Show session information")
                print("   • 'quit'   - Exit (memory cleared)")
                continue
            
            if query.lower() == 'stats':
                show_session_stats(cag_system)
                continue
            
            if query.lower() == 'reset':
                cag_system.reset_session()
                print("✅ Session restarted - starting fresh again")
                continue
            
            if query.lower() == 'whoami':
                show_session_info(cag_system)
                continue
            
            # Process query
            print("🤖 Assistant: ", end="", flush=True)
            
            if use_streaming:
                # Streaming mode
                try:
                    for token in cag_system.stream_query(query):
                        print(token, end="", flush=True)
                    print()  # New line after streaming
                except Exception as e:
                    print(f"\n❌ Error: {e}")
            else:
                # Batch mode
                result = cag_system.query(query)
                if result['success']:
                    print(result['answer'])
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        except KeyboardInterrupt:
            user_name = cag_system.memory.user_profile.name
            if user_name:
                print(f"\n\n👋 Goodbye, {user_name}!")
            else:
                print("\n\n👋 Goodbye!")
            print("Session memory cleared.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def show_session_summary(cag_system: CAGSystemFreshSession):
    """Ask the LLM to summarise the session and display the result."""
    print("\n" + "="*80)
    print("🧠 SESSION SUMMARY (from LLM memory)")
    print("="*80)
    print("⏳ Generating summary...", end="", flush=True)

    result = cag_system.generate_session_summary()

    print("\r" + " "*40 + "\r", end="")   # clear the "Generating..." line

    user_name = result.get('user_name') or result.get('llm_name') or "Unknown"
    summary   = result.get('summary', 'No summary available.')

    print(f"\n👤 User Name  : {user_name}")
    print(f"📝 Summary    : {summary}")
    print("="*80)


def show_session_info(cag_system: CAGSystemFreshSession):
    """Display current session information"""
    stats = cag_system.get_stats()
    
    print("\n" + "="*80)
    print("📊 CURRENT SESSION INFORMATION")
    print("="*80)
    
    print(f"\n⏰ Session Details:")
    print(f"   Started: {stats.get('session_start', 'N/A')[:19]}")
    print(f"   Session Mode: Fresh (no persistence)")
    print(f"   Total Queries: {stats['total_queries']}")
    
    if stats.get('memory'):
        mem = stats['memory']
        print(f"\n👤 User Information (This Session Only):")
        if mem['user_name']:
            print(f"   Name: {mem['user_name']}")
        else:
            print(f"   Name: Not yet provided")
        print(f"   Messages: {mem['total_messages']}")
        print(f"   Interactions: {mem['total_interactions']}")
    
    print(f"\n⚠️  IMPORTANT:")
    print(f"   • This session's memory is temporary")
    print(f"   • When you exit, everything will be cleared")
    print(f"   • Next run will be a completely fresh start")
    
    print("="*80)


def show_session_stats(cag_system: CAGSystemFreshSession):
    """Display session statistics"""
    print_section_header("SESSION STATISTICS")
    
    stats = cag_system.get_stats()
    
    # Session info
    print(f"\n⏰ Session:")
    print(f"   • Mode: Fresh (No Persistence)")
    print(f"   • Started: {stats.get('session_start', 'N/A')[:19]}")
    print(f"   • Total queries: {stats['total_queries']}")
    
    # Knowledge stats
    print(f"\n📚 Knowledge Base:")
    print(f"   • Total entries: {stats['knowledge']['entries']:,}")
    print(f"   • Tokens: {stats['knowledge']['tokens']:,}")
    
    # Cache stats
    print(f"\n🎯 Cache:")
    print(f"   • Initialized: {stats['cache']['initialized']}")
    print(f"   • Knowledge tokens: {stats['cache']['knowledge_tokens']:,}")
    
    # Memory stats (current session only)
    if stats.get('memory'):
        mem = stats['memory']
        print(f"\n💭 Session Memory (Temporary):")
        print(f"   • User name: {mem['user_name'] or 'Not provided yet'}")
        print(f"   • Messages: {mem['total_messages']}")
        print(f"   • Interactions: {mem['total_interactions']}")
        print(f"   • ⚠️  Cleared when program exits")
    
    # GPU stats
    if stats.get('gpu_memory'):
        gpu = stats['gpu_memory']
        print(f"\n🖥️  GPU Memory:")
        print(f"   • Total: {gpu['total_mb']:,}MB")
        print(f"   • Used: {gpu['used_mb']:,}MB ({gpu['utilization']:.1f}%)")
        print(f"   • Free: {gpu['free_mb']:,}MB")
    
    print("="*80)


def run_demo(cag_system: CAGSystemFreshSession):
    """Run an automated demo"""
    print_section_header("FRESH SESSION DEMO")
    
    print("\n🎬 This demo shows how fresh session mode works:")
    print("   1. Every run starts fresh")
    print("   2. Asks for name at the beginning")
    print("   3. Memory during session only")
    print("   4. Everything resets on exit")
    print("\n" + "="*80)
    
    demo_conversations = [
        ("Hello!", "First greeting - will ask for name"),
        ("My name is Alex", "Providing name - saved for THIS session"),
        ("What are your shipping options?", "Regular query - uses name from THIS session"),
        ("What about returns?", "Follow-up - maintains context in THIS session"),
    ]
    
    for i, (query, description) in enumerate(demo_conversations, 1):
        print(f"\n{'─'*80}")
        print(f"📝 Step {i}: {description}")
        print(f"{'─'*80}")
        print(f"\n👤 User: {query}")
        print("🤖 Assistant: ", end="", flush=True)
        
        result = cag_system.query(query)
        if result['success']:
            print(result['answer'])
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        input("\nPress Enter to continue...")
    
    print(f"\n{'='*80}")
    print("✅ Demo complete!")
    print("\n💡 Important Notes:")
    print("   • The name 'Alex' was saved during THIS session")
    print("   • When you exit and run again, it will ask for name again")
    print("   • Perfect for demos where you want fresh start each time")
    print("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="CAG Chatbot - Fresh Session Mode (No Persistence)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cag_fresh_session.py                # Interactive mode (streaming)
  python cag_fresh_session.py --no-stream    # Interactive mode (batch)
  python cag_fresh_session.py --demo         # Run automated demo

Note: This mode does NOT save memory between runs.
Each time you run the program, it's a completely fresh start.
        """
    )
    
    # Configuration options
    parser.add_argument(
        '--max-context',
        type=int,
        default=5000,
        help='Maximum context tokens (default: 5000)'
    )
    
    parser.add_argument(
        '--max-new',
        type=int,
        default=256,
        help='Maximum new tokens per response (default: 256)'
    )
    
    parser.add_argument(
        '--rebuild-cache',
        action='store_true',
        help='Force rebuild of knowledge cache'
    )
    
    # Mode options
    parser.add_argument(
        '--no-stream',
        action='store_true',
        help='Disable streaming mode'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run fresh session demo'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Create configuration
    config = CAGConfig(
        max_context_tokens=args.max_context,
        max_new_tokens=args.max_new,
        enable_cache_persistence=True,  # Keep cache, just not conversation memory
        enable_conversation_memory=False,  # Will be overridden to work without persistence
        verbose=True
    )
    
    print(f"\n⚙️  Configuration:")
    print(f"   • Max context tokens: {config.max_context_tokens:,}")
    print(f"   • Max new tokens: {config.max_new_tokens}")
    print(f"   • Session mode: Fresh (No Persistence)")
    print(f"   • Streaming: {not args.no_stream}")
    
    try:
        # Initialize system
        print_section_header("SYSTEM INITIALIZATION")
        cag_system = CAGSystemFreshSession(config)
        cag_system.initialize(force_cache_rebuild=args.rebuild_cache)
        
        # Show initial stats
        print("\n" + "─"*80)
        show_session_stats(cag_system)
        
        # Run appropriate mode
        if args.demo:
            # Run demo
            run_demo(cag_system)
        else:
            # Interactive mode (default)
            interactive_mode(cag_system, use_streaming=not args.no_stream)
        
        # Session summary from LLM memory
        show_session_summary(cag_system)

        # Final stats
        print("\n" + "─"*80)
        show_session_stats(cag_system)
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        cag_system.cleanup()
        print("✅ Cleanup complete")
        print("   Session memory has been cleared")
        
        print("\n" + "="*80)
        print("✅ SESSION ENDED")
        print("   Next run will be a fresh start!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        print("Session memory cleared.")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()