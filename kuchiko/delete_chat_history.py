"""
Delete Chat History from Memgraph
Clears user conversations from the database for fresh testing.

Usage:
    python delete_chat_history.py              # Interactive: choose user or all
    python delete_chat_history.py all          # Delete all chat history
    python delete_chat_history.py user <ID>    # Delete specific user
    python delete_chat_history.py stats        # Show what's stored

WARNING: This permanently deletes chat history!
"""

import sys
from neo4j import GraphDatabase, basic_auth

# Memgraph configuration
MEMGRAPH_URI = "bolt://localhost:7687"
MEMGRAPH_USER = "memgraph"
MEMGRAPH_PASS = "memgraph"

# Initialize driver
driver = GraphDatabase.driver(MEMGRAPH_URI, auth=basic_auth(MEMGRAPH_USER, MEMGRAPH_PASS))


def show_user_stats():
    """Show statistics about stored users."""
    
    try:
        with driver.session() as session:
            query = """
            MATCH (u:User)-[:SENT]->(m:Message)
            RETURN 
                u.user_id as user_id,
                u.first_name as first_name,
                u.username as username,
                count(m) as message_count,
                min(m.timestamp) as first_message,
                max(m.timestamp) as last_message
            ORDER BY message_count DESC
            """
            
            result = session.run(query)
            
            print("=" * 80)
            print("Chat History Statistics")
            print("=" * 80)
            print()
            
            users = list(result)
            if not users:
                print("No chat history found.")
                return []
            
            for record in users:
                user_id = record["user_id"]
                name = record["first_name"] or record["username"] or f"User {user_id}"
                msg_count = record["message_count"]
                first = record["first_message"]
                last = record["last_message"]
                
                print(f"👤 {name} (ID: {user_id})")
                print(f"   Messages: {msg_count}")
                print(f"   First: {first[:19] if first else 'N/A'}")
                print(f"   Last: {last[:19] if last else 'N/A'}")
                print()
            
            print("=" * 80)
            return users
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def delete_all_chat_history():
    """Delete ALL chat history from Memgraph."""
    
    print("=" * 80)
    print("⚠️  WARNING: Deleting ALL Chat History")
    print("=" * 80)
    print()
    
    # Show what will be deleted
    stats = show_user_stats()
    
    if not stats:
        print("Nothing to delete.")
        return
    
    print()
    confirm = input("Type 'DELETE ALL' to confirm: ").strip()
    
    if confirm != "DELETE ALL":
        print("❌ Cancelled. No changes made.")
        return
    
    try:
        with driver.session() as session:
            # Delete all messages and SENT relationships
            query = """
            MATCH (u:User)-[s:SENT]->(m:Message)
            DETACH DELETE m
            """
            session.run(query)
            
            # Optionally delete User nodes too
            query2 = """
            MATCH (u:User)
            WHERE NOT (u)-[:SENT]->()
            DELETE u
            """
            session.run(query2)
            
            print()
            print("✅ All chat history deleted successfully!")
            print("=" * 80)
            
    except Exception as e:
        print(f"❌ Error deleting chat history: {e}")


def delete_user_chat_history(user_id: int):
    """Delete chat history for a specific user."""
    
    print(f"=" * 80)
    print(f"⚠️  WARNING: Deleting Chat History for User {user_id}")
    print("=" * 80)
    print()
    
    try:
        with driver.session() as session:
            # Check if user exists and show their messages
            check_query = """
            MATCH (u:User {user_id: $user_id})-[:SENT]->(m:Message)
            RETURN 
                u.first_name as first_name,
                u.username as username,
                count(m) as message_count
            """
            
            result = session.run(check_query, user_id=user_id)
            record = result.single()
            
            if not record:
                print(f"❌ No chat history found for user {user_id}")
                return
            
            name = record["first_name"] or record["username"] or f"User {user_id}"
            msg_count = record["message_count"]
            
            print(f"👤 User: {name}")
            print(f"💬 Messages to delete: {msg_count}")
            print()
            
            confirm = input(f"Type 'DELETE' to confirm: ").strip()
            
            if confirm != "DELETE":
                print("❌ Cancelled. No changes made.")
                return
            
            # Delete messages for this user
            delete_query = """
            MATCH (u:User {user_id: $user_id})-[s:SENT]->(m:Message)
            DETACH DELETE m
            """
            session.run(delete_query, user_id=user_id)
            
            # Delete user node if no more relationships
            cleanup_query = """
            MATCH (u:User {user_id: $user_id})
            WHERE NOT (u)-[:SENT]->()
            DELETE u
            """
            session.run(cleanup_query, user_id=user_id)
            
            print()
            print(f"✅ Deleted {msg_count} messages for user {user_id}")
            print("=" * 80)
            
    except Exception as e:
        print(f"❌ Error: {e}")


def interactive_delete():
    """Interactive mode: let user choose what to delete."""
    
    print("=" * 80)
    print("🗑️  Delete Chat History")
    print("=" * 80)
    print()
    
    # Show current stats
    users = show_user_stats()
    
    if not users:
        print("Nothing to delete.")
        return
    
    print()
    print("Options:")
    print("  1. Delete ALL chat history")
    print("  2. Delete specific user")
    print("  3. Cancel")
    print()
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == "1":
        delete_all_chat_history()
    elif choice == "2":
        user_id = input("Enter user ID: ").strip()
        try:
            user_id = int(user_id)
            delete_user_chat_history(user_id)
        except ValueError:
            print("❌ Invalid user ID")
    else:
        print("❌ Cancelled")


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == "all":
                delete_all_chat_history()
            elif command == "user" and len(sys.argv) > 2:
                user_id = int(sys.argv[2])
                delete_user_chat_history(user_id)
            elif command == "stats":
                show_user_stats()
            else:
                print("Usage:")
                print("  python delete_chat_history.py           # Interactive mode")
                print("  python delete_chat_history.py all       # Delete everything")
                print("  python delete_chat_history.py user <ID> # Delete specific user")
                print("  python delete_chat_history.py stats     # Show statistics")
        else:
            # Default: interactive mode
            interactive_delete()
    
    finally:
        driver.close()
