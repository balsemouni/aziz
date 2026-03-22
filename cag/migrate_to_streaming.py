#!/usr/bin/env python3
"""
CAG Streaming Migration Helper
===============================

This script helps you integrate streaming capabilities into your existing CAG system.

Usage:
    python migrate_to_streaming.py --check       # Check what needs to be done
    python migrate_to_streaming.py --backup      # Backup current files
    python migrate_to_streaming.py --migrate     # Perform migration
    python migrate_to_streaming.py --test        # Test after migration
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


class StreamingMigrator:
    """Helper to migrate CAG system to include streaming"""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.backup_dir = self.base_dir / "backup_before_streaming"
        
        self.files_to_check = {
            'model_loader.py': 'Model loader',
            'cag_system.py': 'CAG system',
            'cag_main.py': 'Main application',
            'cag_config.py': 'Configuration'
        }
        
        self.streaming_files = {
            'model_loader_enhanced.py': 'Enhanced model loader',
            'cag_main_enhanced.py': 'Enhanced main application',
            'cag_system_streaming_addon.py': 'Streaming addon methods',
            'INTEGRATION_GUIDE.py': 'Integration guide'
        }
    
    def check_status(self):
        """Check current status and what needs to be done"""
        print("\n" + "="*70)
        print("🔍 CHECKING CAG SYSTEM STATUS")
        print("="*70)
        
        # Check existing files
        print("\n📁 Existing Files:")
        existing = []
        missing = []
        
        for filename, description in self.files_to_check.items():
            filepath = self.base_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                print(f"   ✅ {filename} ({size} bytes) - {description}")
                existing.append(filename)
            else:
                print(f"   ❌ {filename} - {description} (NOT FOUND)")
                missing.append(filename)
        
        # Check streaming files
        print("\n📦 Streaming Enhancement Files:")
        streaming_present = []
        streaming_missing = []
        
        for filename, description in self.streaming_files.items():
            filepath = self.base_dir / filename
            if filepath.exists():
                print(f"   ✅ {filename} - {description}")
                streaming_present.append(filename)
            else:
                print(f"   ❌ {filename} - {description} (NOT FOUND)")
                streaming_missing.append(filename)
        
        # Check if already migrated
        print("\n🔎 Migration Status:")
        if self._is_already_migrated():
            print("   ✅ System appears to have streaming capabilities")
            print("   ℹ️  No migration needed")
            return True
        else:
            print("   ⚠️  System does NOT have streaming capabilities")
            print("   📝 Migration recommended")
            
            if streaming_missing:
                print("\n❌ Missing streaming enhancement files:")
                for f in streaming_missing:
                    print(f"      - {f}")
                print("\n   Please ensure all enhancement files are present!")
                return False
        
        # Print recommendations
        print("\n📋 Next Steps:")
        print("   1. Run: python migrate_to_streaming.py --backup")
        print("   2. Run: python migrate_to_streaming.py --migrate")
        print("   3. Run: python migrate_to_streaming.py --test")
        
        return True
    
    def _is_already_migrated(self):
        """Check if system already has streaming"""
        model_loader_path = self.base_dir / 'model_loader.py'
        
        if not model_loader_path.exists():
            return False
        
        # Check if stream_response method exists
        with open(model_loader_path, 'r') as f:
            content = f.read()
            return 'def stream_response' in content
    
    def backup_files(self):
        """Backup current files before migration"""
        print("\n" + "="*70)
        print("💾 BACKING UP CURRENT FILES")
        print("="*70)
        
        # Create backup directory
        self.backup_dir.mkdir(exist_ok=True)
        print(f"\n📁 Backup directory: {self.backup_dir}")
        
        # Backup each existing file
        backed_up = []
        for filename in self.files_to_check.keys():
            src = self.base_dir / filename
            if src.exists():
                dst = self.backup_dir / filename
                shutil.copy2(src, dst)
                print(f"   ✅ Backed up: {filename}")
                backed_up.append(filename)
        
        if backed_up:
            print(f"\n✅ Backed up {len(backed_up)} files")
            print(f"   Location: {self.backup_dir}")
            print("\n   To restore: copy files from backup/ back to main directory")
        else:
            print("\n⚠️  No files to backup")
        
        return len(backed_up) > 0
    
    def migrate(self):
        """Perform migration to streaming"""
        print("\n" + "="*70)
        print("🚀 MIGRATING TO STREAMING")
        print("="*70)

        if self._is_already_migrated():
            print("\n✅ System already has streaming capabilities — migration skipped.")
            print("   Running --migrate again on an already-migrated codebase would")
            print("   corrupt cag_system.py by inserting duplicate methods.")
            print("   No action taken.")
            return True
        
        # Step 1: Replace model_loader.py
        print("\n📝 Step 1: Updating model_loader.py...")
        if not self._update_model_loader():
            return False
        
        # Step 2: Update cag_system.py
        print("\n📝 Step 2: Updating cag_system.py...")
        if not self._update_cag_system():
            return False
        
        # Step 3: Create enhanced main (optional)
        print("\n📝 Step 3: Creating enhanced main application...")
        self._create_enhanced_main()
        
        print("\n" + "="*70)
        print("✅ MIGRATION COMPLETE!")
        print("="*70)
        print("\nYour system now has streaming capabilities!")
        print("\nNext steps:")
        print("   1. Run: python migrate_to_streaming.py --test")
        print("   2. Try: python cag_main_enhanced.py")
        
        return True
    
    def _update_model_loader(self):
        """Update model_loader.py with streaming"""
        src = self.base_dir / 'model_loader_enhanced.py'
        dst = self.base_dir / 'model_loader.py'
        
        if not src.exists():
            print("   ❌ model_loader_enhanced.py not found!")
            return False
        
        # Copy enhanced version
        shutil.copy2(src, dst)
        print("   ✅ model_loader.py updated with streaming methods")
        return True
    
    def _update_cag_system(self):
        """Add streaming methods to cag_system.py"""
        system_file = self.base_dir / 'cag_system.py'
        addon_file = self.base_dir / 'cag_system_streaming_addon.py'

        if not system_file.exists():
            print("   ❌ cag_system.py not found!")
            return False

        if not addon_file.exists():
            print("   ❌ cag_system_streaming_addon.py not found!")
            return False

        # Read current system file
        with open(system_file, 'r') as f:
            content = f.read()

        # HARD ABORT: if stream_query already exists, do NOT proceed.
        # Inserting text into an already-migrated file would create
        # duplicate method definitions and corrupt the class.
        if 'def stream_query' in content:
            print("   ✅ cag_system.py already has streaming methods — no changes made.")
            return True
        
        # Read addon methods
        with open(addon_file, 'r') as f:
            addon_content = f.read()
        
        # Extract just the methods (skip the documentation parts)
        import_line = "import asyncio"
        methods_start = addon_content.find('def stream_query(')
        
        if methods_start == -1:
            print("   ❌ Could not find streaming methods in addon file")
            return False
        
        # Find where to insert
        class_match = content.rfind('class CAGSystem')
        if class_match == -1:
            print("   ❌ Could not find CAGSystem class")
            return False
        
        # Add import if needed
        if 'import asyncio' not in content:
            # Find import section
            last_import = max(
                content.rfind('import '),
                content.rfind('from ')
            )
            if last_import != -1:
                # Find end of that line
                newline = content.find('\n', last_import)
                content = content[:newline+1] + "import asyncio\n" + content[newline+1:]
                print("   ✅ Added asyncio import")
        
        # Add methods at end of class
        # Find last method or end of class
        methods = addon_content[methods_start:]
        
        # Insert before the final closing or at end
        insert_pos = len(content)
        if content.rstrip().endswith('"""'):
            # Insert before final docstring
            insert_pos = content.rfind('"""')
        
        # Add methods with proper indentation
        indented_methods = '\n    ' + methods.replace('\n', '\n    ')
        content = content[:insert_pos] + indented_methods + '\n' + content[insert_pos:]
        
        # Write back
        with open(system_file, 'w') as f:
            f.write(content)
        
        print("   ✅ Added streaming methods to cag_system.py")
        return True
    
    def _create_enhanced_main(self):
        """Create enhanced main application"""
        src = self.base_dir / 'cag_main_enhanced.py'
        
        if src.exists():
            print("   ✅ cag_main_enhanced.py already exists")
            print("   ℹ️  You can now use: python cag_main_enhanced.py")
            return True
        
        print("   ⚠️  cag_main_enhanced.py not found")
        print("   ℹ️  You'll need to manually update cag_main.py or use the enhanced version")
        return False
    
    def test_migration(self):
        """Test that migration was successful"""
        print("\n" + "="*70)
        print("🧪 TESTING MIGRATION")
        print("="*70)
        
        errors = []
        
        # Test 1: Check model_loader has streaming
        print("\n📝 Test 1: Checking model_loader.py...")
        model_loader = self.base_dir / 'model_loader.py'
        if model_loader.exists():
            with open(model_loader, 'r') as f:
                content = f.read()
                if 'def stream_response' in content:
                    print("   ✅ stream_response method found")
                else:
                    print("   ❌ stream_response method NOT found")
                    errors.append("model_loader.py missing stream_response")
        else:
            print("   ❌ model_loader.py not found")
            errors.append("model_loader.py not found")
        
        # Test 2: Check cag_system has streaming
        print("\n📝 Test 2: Checking cag_system.py...")
        cag_system = self.base_dir / 'cag_system.py'
        if cag_system.exists():
            with open(cag_system, 'r') as f:
                content = f.read()
                if 'def stream_query' in content:
                    print("   ✅ stream_query method found")
                else:
                    print("   ❌ stream_query method NOT found")
                    errors.append("cag_system.py missing stream_query")
                
                if 'import asyncio' in content:
                    print("   ✅ asyncio import found")
                else:
                    print("   ⚠️  asyncio import not found (may cause issues)")
        else:
            print("   ❌ cag_system.py not found")
            errors.append("cag_system.py not found")
        
        # Test 3: Try importing (optional)
        print("\n📝 Test 3: Testing imports...")
        try:
            sys.path.insert(0, str(self.base_dir))
            
            # Try importing (will fail if dependencies missing, that's ok)
            print("   ℹ️  Attempting imports (may fail if dependencies not installed)...")
            
            try:
                from model_loader import ModelLoader
                print("   ✅ ModelLoader imported successfully")
            except ImportError as e:
                print(f"   ⚠️  Could not import ModelLoader: {e}")
                print("      (This is OK if dependencies aren't installed)")
            
        except Exception as e:
            print(f"   ⚠️  Import test skipped: {e}")
        
        # Summary
        print("\n" + "="*70)
        if errors:
            print("❌ MIGRATION TEST FAILED")
            print("\nErrors found:")
            for error in errors:
                print(f"   - {error}")
            print("\nPlease re-run migration or check files manually.")
            return False
        else:
            print("✅ MIGRATION TEST PASSED")
            print("\nYour system is ready to use streaming!")
            print("\nTry it out:")
            print("   python cag_main_enhanced.py")
            print("   python cag_main_enhanced.py --demo")
            return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate CAG system to include streaming capabilities"
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check current status'
    )
    
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Backup current files'
    )
    
    parser.add_argument(
        '--migrate',
        action='store_true',
        help='Perform migration'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test migration'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Do everything (backup, migrate, test)'
    )
    
    parser.add_argument(
        '--dir',
        type=str,
        default='.',
        help='Base directory (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Create migrator
    migrator = StreamingMigrator(args.dir)
    
    # Run requested operations
    if args.all:
        print("🚀 Running full migration process...")
        if not migrator.check_status():
            print("\n❌ Pre-flight check failed!")
            sys.exit(1)
        
        if not migrator.backup_files():
            print("\n⚠️  No files to backup (continuing anyway)")
        
        if not migrator.migrate():
            print("\n❌ Migration failed!")
            sys.exit(1)
        
        if not migrator.test_migration():
            print("\n❌ Migration tests failed!")
            sys.exit(1)
        
        print("\n🎉 Complete! Your system now has streaming capabilities!")
    
    elif args.check:
        migrator.check_status()
    
    elif args.backup:
        migrator.backup_files()
    
    elif args.migrate:
        migrator.migrate()
    
    elif args.test:
        migrator.test_migration()
    
    else:
        # Show usage
        parser.print_help()
        print("\n💡 Quick start:")
        print("   python migrate_to_streaming.py --all")


if __name__ == "__main__":
    main()