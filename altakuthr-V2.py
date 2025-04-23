import os
import time
import logging
import os
import pyautogui
import ollama
import requests
import sqlite3
import glob
import re
from threading import Event
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict
from collections import deque

# --- Configuration ---
# Change this to a multimodal model that supports image input via Ollama
OLLAMA_MODEL = 'llava:13b' # LLaVA 13B is a strong contender for powerful multimodal in Ollama
# You will need to run: ollama pull llava:13b
SCREENSHOT_INTERVAL = 1     # How often to take a screenshot (seconds)
FAILSAFE_CORNER = 'topLeft' # PyAutoGUI failsafe corner (move mouse there to stop)
MAX_PROCESSED_COMMANDS = 20 # How many recent commands to remember

# --- Logging Setup ---
logger = logging.getLogger('telepathy')
logger.setLevel(logging.DEBUG)

# Configure root logger (for basic info/errors from system libs)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
root_handler.setFormatter(formatter)
# Prevent duplicate handlers if run multiple times (e.g., in interactive session)
if not root_logger.handlers:
    root_logger.addHandler(root_handler)

# Specific handler for this script's logs
handler = logging.StreamHandler()
file_handler = logging.FileHandler('automation.log', mode='w')
handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
# Check if handlers already exist to avoid duplicates if run interactively
if not logger.handlers:
    logger.addHandler(handler)
    logger.addHandler(file_handler)


# Configure RAG logger (optional, adjust level if needed)
rag_logger = logging.getLogger('rag')
rag_logger.setLevel(logging.INFO)
if not rag_logger.handlers:
    rag_logger.addHandler(handler) # Use the same stream handler as main logger
    rag_logger.addHandler(file_handler) # Use the same file handler


# Load environment variables
load_dotenv()

# --- Database for RAG and History ---
class RagDatabase:
    def __init__(self):
        self.conn = None
        try:
            self.conn = sqlite3.connect('rag.db')
            self._create_tables()
            logger.info("Successfully connected to rag.db")
        except sqlite3.Error as e:
            logger.critical(f"Database connection error: {str(e)}")
            # Depending on criticality, might re-raise or exit

    def _create_tables(self):
        if not self.conn: return
        try:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS knowledge
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 context TEXT NOT NULL,
                 source TEXT,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            self.conn.execute('''CREATE TABLE IF NOT EXISTS command_history
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 command TEXT NOT NULL,
                 success INTEGER,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            self.conn.commit()
            logger.debug("Database tables checked/created.")
        except sqlite3.Error as e:
            logger.error(f"Error creating database tables: {str(e)}")

    def add_context(self, context: str, source: str = None):
        if not self.conn: return
        try:
            self.conn.execute('INSERT INTO knowledge (context, source) VALUES (?, ?)',
                            (context, source))
            self.conn.commit()
            rag_logger.debug(f"Added context from source: {source}")
        except sqlite3.Error as e:
            rag_logger.error(f"Error adding context: {str(e)}")

    def get_relevant_context(self, query: str, limit: int = 3) -> List[Dict]:
        """Retrieves context from the knowledge base relevant to the query."""
        if not self.conn or not query: return []
        cursor = self.conn.cursor()
        try:
            # Using LIKE %query% is a very basic similarity search.
            # For better RAG, you'd integrate embeddings and vector search.
            cursor.execute('''SELECT context, source FROM knowledge
                            WHERE context LIKE ?
                            ORDER BY timestamp DESC LIMIT ?''',
                         (f'%{query}%', limit))
            results = [{'context': row[0], 'source': str(row[1] or '')} for row in cursor.fetchall()]
            rag_logger.debug(f"Retrieved {len(results)} contexts for query: '{query}'")
            return results
        except sqlite3.Error as e:
            rag_logger.error(f"Error getting relevant context: {str(e)}")
            return []

    def log_command(self, command: str, success: bool):
        if not self.conn: return
        try:
            self.conn.execute('INSERT INTO command_history (command, success) VALUES (?, ?)',
                            (command, int(success)))
            self.conn.commit()
            logger.debug(f"Logged command: '{command}' success={success}")
        except sqlite3.Error as e:
            logger.error(f"Error logging command: {str(e)}")

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed.")

# --- Optional: Obsidian Indexing ---
class ObsidianIndexer:
    def __init__(self, vault_path: str, rag_db: RagDatabase):
        self.vault_path = vault_path
        self.rag_db = rag_db
        if self.vault_path and not os.path.exists(self.vault_path):
             logger.warning(f"Obsidian vault path not found: {self.vault_path}")
             self.vault_path = None # Disable indexing if path is invalid

    def index_notes(self):
        if not self.vault_path:
            logger.info("Obsidian indexing skipped due to invalid vault path.")
            return

        logger.info(f"Starting Obsidian indexing for vault: {self.vault_path}")
        md_files = glob.glob(f'{self.vault_path}/**/*.md', recursive=True)
        indexed_count = 0
        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.rag_db.add_context(content, source=file_path)
                    indexed_count += 1
                    # logger.debug(f'Indexed Obsidian note: {file_path}') # Too verbose for info level
            except Exception as e:
                logger.error(f'Error indexing {file_path}: {str(e)}')
        logger.info(f"Finished Obsidian indexing. Indexed {indexed_count} markdown files.")

# --- Optional: Learning System (Basic) ---
class LearningSystem:
    def __init__(self, rag_db: RagDatabase):
        self.rag_db = rag_db
        self.patterns = {} # Most frequent successful commands

    def analyze_usage_patterns(self):
        if not self.rag_db.conn: return {}
        cursor = self.rag_db.conn.cursor()
        try:
            cursor.execute('''SELECT command, COUNT(*) as count
                            FROM command_history
                            WHERE success = 1
                            GROUP BY command
                            ORDER BY count DESC LIMIT 5''') # Get top 5 successful commands
            self.patterns = {row[0]: row[1] for row in cursor.fetchall()}
            logger.debug(f"Analyzed command usage patterns: {self.patterns}")
            return self.patterns
        except sqlite3.Error as e:
             logger.error(f"Error analyzing usage patterns: {str(e)}")
             return {}

    def get_suggested_commands(self) -> List[str]:
        return list(self.patterns.keys())

# --- Core Automation Engine ---
class AutomationEngine:
    def __init__(self, rag_db: RagDatabase):
        self.running = Event()
        self.screenshot_interval = SCREENSHOT_INTERVAL
        self.processed_commands = deque(maxlen=MAX_PROCESSED_COMMANDS) # Track recent commands
        self.rag_db = rag_db # Store the database instance

        # Configure pyautogui
        pyautogui.FAILSAFE = True # Enable failsafe
        pyautogui.PAUSE = 0 # Disable default pause, use explicit sleeps
        pyautogui.FAILSAFEPOINT = pyautogui.Point(0, 0) # Top-left corner by default

    def take_screenshot(self):
        """Captures a screenshot and returns the path."""
        try:
            screenshot_path = f"temp_ss_{int(time.time())}.png"
            pyautogui.screenshot(screenshot_path)
            logger.debug(f"Screenshot captured: {screenshot_path}")
            return screenshot_path
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {str(e)}")
            return None

    def get_foreground_app(self):
        """Attempts to get the name of the foreground application."""
        try:
            import sys
            if sys.platform == "win32":
                import ctypes
                from ctypes import wintypes

                hwnd = ctypes.windll.user32.GetForegroundWindow()
                pid = wintypes.DWORD()
                ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

                PROCESS_QUERY_INFORMATION = 0x0400
                h_process = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
                if h_process:
                    exe_path = (ctypes.c_char * 512)()
                    if ctypes.windll.psapi.GetModuleFileNameExA(h_process, None, ctypes.byref(exe_path), 512):
                        ctypes.windll.kernel32.CloseHandle(h_process)
                        return exe_path.value.decode(errors='ignore').split('\\')[-1]
                    ctypes.windll.kernel32.CloseHandle(h_process)

                # Fallback to window title
                length = ctypes.windll.user32.GetWindowTextLengthA(hwnd)
                buff = ctypes.create_string_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextA(hwnd, buff, length + 1)
                return buff.value.decode(errors='ignore') if buff.value else "Unknown (Win32)"
            elif sys.platform == "darwin": # macOS basic example (might need specific libs like pyobjc)
                 try:
                     from AppKit import NSWorkspace
                     app = NSWorkspace.sharedWorkspace().frontmostApplication()
                     return app.localizedName() if app else "Unknown (macOS)"
                 except ImportError:
                     logger.warning("Install pyobjc-core and pyobjc for better macOS foreground app detection.")
                     return "Unknown (macOS - lib missing)"
            elif sys.platform.startswith("linux"): # Linux basic example (might need wmctrl or similar)
                 try:
                     import subprocess
                     output = subprocess.check_output(["wmctrl", "-G", "-l", "-p"]).decode('utf-8').splitlines()
                     pid = subprocess.check_output(["xdotool", "getwindowfocus", "getwindowpid"]).decode('utf-8').strip()
                     for line in output:
                         if f" {pid} " in line:
                             parts = line.split(maxsplit=4)
                             if len(parts) > 4:
                                 # Attempt to get process name from /proc
                                 try:
                                     process_name = subprocess.check_output(["ps", "-p", pid, "-o", "comm="]).decode('utf-8').strip()
                                     return process_name
                                 except:
                                     return parts[-1].strip() # Fallback to window title/class
                     return "Unknown (Linux)"
                 except (FileNotFoundError, subprocess.CalledProcessError):
                     logger.warning("Install wmctrl and xdotool for better Linux foreground app detection.")
                     return "Unknown (Linux - lib missing)")
            else:
                return f"Unknown ({sys.platform})"

        except Exception as e:
            logger.error(f"Failed to get foreground app: {str(e)}")
            return "Unknown (Error)"

    def process_image(self, screenshot_path):
        """Sends screenshot and context to AI, gets command response."""
        if not screenshot_path:
            return [] # Cannot process without screenshot

        try:
            # --- RAG Integration ---
            # Use foreground app name and recent commands as the query
            current_app = self.get_foreground_app()
            # Simple query combining app name and last command (if any)
            rag_query = current_app
            if self.processed_commands:
                 rag_query += " " + self.processed_commands[-1] # Add last processed command to query

            relevant_contexts = self.rag_db.get_relevant_context(rag_query)

            # Format retrieved contexts for the prompt
            rag_context_str = ""
            if relevant_contexts:
                rag_context_str = "\n\nRelevant Knowledge (from your notes/history):"
                for i, context in enumerate(relevant_contexts):
                    rag_context_str += f"\n--- Context {i+1} (Source: {context['source']}) ---\n"
                    rag_context_str += context['context'].strip()[:500] + "..." # Limit context length
                    rag_context_str += "\n--------------------"
                rag_context_str += "\n\nConsider this knowledge when deciding the next action."
            else:
                 rag_context_str = "\n\nNo specific relevant knowledge found in your notes/history."

            # --- Prompt Template ---
            # Inject the rag_context_str into the prompt
            prompt_template = f'''
You are an AI assistant controlling a computer using pyautogui commands based on a screenshot of the active window.
Your task is to analyze the visual information, the provided context, and your relevant knowledge to determine the single most logical and efficient next automation step required to interact with the UI, and generate the corresponding pyautogui command.

Generate commands using the following strict format only:
COMMAND: ACTION PARAMETERS

Available Actions and Required Parameters:
1.  CLICK: Requires parameters X,Y representing integer pixel coordinates. Example: COMMAND: CLICK 1234,567
2.  TYPE: Requires parameters: The text to type, enclosed in double quotes (e.g., "Hello world"), optionally followed by special key names in curly braces (e.g., {{enter,shift}}). Use standard keyboard key names supported by pyautogui (e.g., enter, esc, tab, space, backspace, delete, ctrl, alt, shift, win, f1-f12, etc.). Example: COMMAND: TYPE "My search query" {{enter}}
3.  HOTKEY: Requires parameters: Key names joined by '+'. Use standard keyboard key names supported by pyautogui. Example: COMMAND: HOTKEY ctrl+s
4.  SCROLL: Requires parameters: An integer amount. Positive value scrolls down, negative scrolls up. Example: COMMAND: SCROLL -200
5.  DRAG: Requires parameters: Two sets of X,Y coordinates separated by a space: START_X,START_Y END_X,END_Y. Example: COMMAND: DRAG 100,100 200,200

Validation Rules:
-   Generated coordinates (X,Y for CLICK, START_X,START_Y, END_X,END_Y for DRAG) must be within the detected screen resolution: {pyautogui.size()}.
-   Text for TYPE must be strictly enclosed in double quotes.
-   Special keys for TYPE, if used, must be strictly enclosed in curly braces {{}}. Multiple keys should be comma-separated.
-   Hotkeys must use '+' as the separator between key names.
-   Only generate commands that are directly actionable and visible in the screenshot. Do not guess or invent elements.
-   Focus on completing one distinct, logical interaction step.
-   Avoid generating the exact same command that was just recently processed if the UI state appears unchanged.

Application Context:
-   Foreground App: {current_app}
-   Screen Resolution: {pyautogui.size()}
-   Recent Commands (Last {MAX_PROCESSED_COMMANDS} attempts): {list(self.processed_commands)}
{rag_context_str} # Inject the retrieved context here

Based on the current UI, context, and relevant knowledge, generate the *single most appropriate* COMMAND:
'''
            # --- End Prompt Template ---

            logger.debug(f"Sending prompt to Ollama model: {OLLAMA_MODEL}")
            response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=prompt_template,
                images=[screenshot_path]
            )
            raw_response = response.get('response', '').strip()
            logger.debug(f"Raw AI response:\n---\n{raw_response}\n---")

            return self.parse_commands(raw_response)

        except requests.exceptions.ConnectionError:
            logger.error("Connection error: Make sure Ollama is running.")
            return []
        except Exception as e:
            logger.error(f"AI processing error: {str(e)}", exc_info=True)
            return []
        finally:
            if screenshot_path and os.path.exists(screenshot_path):
                try:
                    os.remove(screenshot_path)
                    logger.debug(f"Deleted screenshot: {screenshot_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete screenshot {screenshot_path}: {str(e)}")

    def parse_commands(self, response_text):
        """
        Parses the AI's response text, validates commands, and returns a list of valid command strings.
        This function ONLY PARSES and VALIDATES FORMAT, it does NOT execute.
        """
        valid_commands = []
        screen_width, screen_height = pyautogui.size()
        # Define regex for TYPE command: "text content"{optional,keys}
        type_regex = re.compile(r'^"([^"]*)"\s*(\{([^}]+)\})?$')

        for line in response_text.split('\n'):
            line = line.strip()
            # Look specifically for the start of a command line
            if not line.upper().startswith('COMMAND:'):
                continue

            # Extract the part after "COMMAND:" case-insensitively
            cmd_part_lower = line.lower().split('command:', 1)
            if len(cmd_part_lower) < 2:
                 continue # Should not happen with startswith check, but safe

            cmd_part = cmd_part_lower[1].strip()
            if not cmd_part:
                logger.warning(f"Empty command part after 'COMMAND:' in line: '{line}'")
                continue

            # Split action and parameters
            parts = cmd_part.split(' ', 1)
            action = parts[0].upper() if parts else ''
            params = parts[1].strip() if len(parts) > 1 else ''

            if not action:
                logger.warning(f"Empty action detected in line: '{line}'")
                continue

            logger.debug(f"Parsing - Action: '{action}', Raw Params: '{params}'")

            # --- Command Validation and Formatting ---
            if action == 'CLICK':
                try:
                    x_str, y_str = params.split(',', 1)
                    x, y = int(x_str.strip()), int(y_str.strip())
                    if 0 <= x < screen_width and 0 <= y < screen_height:
                        valid_commands.append(f'CLICK {x},{y}')
                        logger.debug(f"Parsed valid CLICK command: CLICK {x},{y}")
                    else:
                        logger.warning(f"Invalid CLICK: Coordinates ({x},{y}) out of screen bounds {screen_width}x{screen_height} in line: '{line}'")
                except (ValueError, IndexError):
                    logger.warning(f"Invalid CLICK parameters format: '{params}' in line: '{line}' (Expected X,Y)")

            elif action == 'TYPE':
                match = type_regex.match(params)
                if match:
                    text_content = match.group(1)
                    special_keys_str = match.group(3)

                    command_str = f'TYPE "{text_content.replace("\"", "\\\"")}"'
                    if special_keys_str:
                        keys = [k.strip() for k in special_keys_str.split(',') if k.strip()]
                        if keys:
                             command_str += f' {{{",".join(keys)}}}'
                        else:
                            logger.warning(f"Invalid TYPE: Empty special key list in braces in line: '{line}'")
                    valid_commands.append(command_str)
                    logger.debug(f"Parsed valid TYPE command: {command_str}")
                else:
                    logger.warning(f"Invalid TYPE parameters format: '{params}' in line: '{line}' (Expected \"text\"{keys})")

            elif action == 'HOTKEY':
                keys = [k.strip().lower() for k in params.split('+') if k.strip()]
                if keys:
                    valid_commands.append(f'HOTKEY {"+".join(keys)}')
                    logger.debug(f"Parsed valid HOTKEY command: HOTKEY {'+'.join(keys)}")
                else:
                     logger.warning(f"Invalid HOTKEY parameters: '{params}' in line: '{line}' (Expected key1+key2+...)")

            elif action == 'SCROLL':
                 try:
                     amount = int(params.strip())
                     valid_commands.append(f'SCROLL {amount}')
                     logger.debug(f"Parsed valid SCROLL command: SCROLL {amount}")
                 except ValueError:
                     logger.warning(f"Invalid SCROLL parameter: '{params}' in line: '{line}' (Expected integer)")

            elif action == 'DRAG':
                try:
                    coords_parts = params.replace(',', ' ').split()
                    if len(coords_parts) == 4:
                        x1, y1, x2, y2 = map(int, coords_parts)
                        if 0 <= x1 < screen_width and 0 <= y1 < screen_height and \
                           0 <= x2 < screen_width and 0 <= y2 < screen_height:
                            valid_commands.append(f'DRAG {x1},{y1} {x2},{y2}')
                            logger.debug(f"Parsed valid DRAG command: DRAG {x1},{y1} {x2},{y2}")
                        else:
                            logger.warning(f"Invalid DRAG: Coordinates out of screen bounds ({x1},{y1} to ({x2},{y2}) in line: '{line}'")
                    else:
                         logger.warning(f"Invalid DRAG parameters format: '{params}' in line: '{line}' (Expected X1,Y1 X2,Y2)")
                except ValueError:
                     logger.warning(f"Invalid DRAG parameters (not integers): '{params}' in line: '{line}'")

            else:
                logger.warning(f"Unrecognized command action '{action}' in line: '{line}'")

        logger.info(f"Finished parsing. Found {len(valid_commands)} valid commands.")
        return valid_commands

    def execute_command(self, command):
        """
        Executes a single, already parsed and validated command string.
        Logs success/failure.
        """
        success = False
        try:
            logger.info(f"Executing command: {command}")
            parts = command.split(' ', 1)
            action = parts[0]
            params = parts[1] if len(parts) > 1 else ''

            # --- Command Execution ---
            if action == 'CLICK':
                x_str, y_str = params.split(',', 1)
                x, y = int(x_str), int(y_str)
                pyautogui.moveTo(x, y, duration=0.1)
                pyautogui.click(x, y)
                logger.debug(f"Executed: Clicked at ({x},{y})")
                success = True

            elif action == 'TYPE':
                 type_regex_exec = re.compile(r'^"([^"]*)"\s*(\{([^}]+)\})?$')
                 match = type_regex_exec.match(params)

                 if match:
                     text_content = match.group(1)
                     special_keys_str = match.group(3)

                     if text_content:
                         pyautogui.write(text_content, interval=0.01)
                         logger.debug(f"Executed: Typed '{text_content}'")

                     if special_keys_str:
                         keys = [k.strip() for k in special_keys_str.split(',') if k.strip()]
                         for key in keys:
                             try:
                                 pyautogui.press(key)
                                 logger.debug(f"Executed: Pressed key '{key}'")
                                 time.sleep(0.05)
                             except Exception as e:
                                  logger.warning(f"Failed to press key '{key}': {e}")
                                  pass
                     success = True
                 else:
                     logger.error(f"Execution failed: Internal TYPE command format error for '{command}'")

            elif action == 'HOTKEY':
                keys = params.split('+')
                pyautogui.hotkey(*keys)
                logger.debug(f"Executed: Hotkey '{params}'")
                success = True

            elif action == 'SCROLL':
                amount = int(params)
                pyautogui.scroll(amount)
                logger.debug(f"Executed: Scroll {amount}")
                success = True

            elif action == 'DRAG':
                coords_str = params.replace(',', ' ').split()
                x1, y1, x2, y2 = map(int, coords_str)
                pyautogui.moveTo(x1, y1, duration=0.05) # Shorter move before drag
                pyautogui.dragTo(x2, y2, duration=0.3) # Shorter drag duration
                logger.debug(f"Executed: Drag from ({x1},{y1}) to ({x2},{y2})")
                success = True

            else:
                logger.error(f"Execution failed: Unknown action '{action}' for command: {command}")

        except pyautogui.FailSafeException:
            logger.critical(f"Fail-safe triggered during execution of '{command}'! Automation stopped. Move mouse to corner {FAILSAFE_CORNER} to disable.")
            self.stop()
            success = False
        except ValueError as e:
             logger.error(f"Execution failed (ValueError) for command '{command}': {str(e)}")
             success = False
        except Exception as e:
            logger.error(f"Execution failed (General Error) for command '{command}': {str(e)}", exc_info=True)
            success = False
        finally:
            if self.rag_db:
                 self.rag_db.log_command(command, success)

            # Add a small delay after each command attempt for stability
            time.sleep(0.5) # Wait a bit after execution


    def run_loop(self):
        """Starts the main automation loop."""
        self.running.set()
        logger.info(f"Starting AI-Automation loop (Interval: {self.screenshot_interval}s, Model: {OLLAMA_MODEL}, Failsafe: {pyautogui.FAILSAFE} at {FAILSAFE_CORNER})")
        logger.info(f"Move mouse to {FAILSAFE_CORNER} corner to trigger failsafe and stop.")

        while self.running.is_set():
            try:
                # 1. Capture current screen state
                ss_path = self.take_screenshot()
                if not ss_path:
                    logger.error("Failed to get screenshot, skipping iteration.")
                    time.sleep(1)
                    continue

                # 2. Process screenshot with AI to get potential commands (RAG integrated here)
                potential_commands = self.process_image(ss_path) # This now includes RAG data in prompt

                # 3. Filter out recently processed commands
                new_commands_to_execute = [
                    cmd for cmd in potential_commands
                    if cmd not in self.processed_commands
                ]

                logger.debug(f"Potential commands from AI: {potential_commands}")
                logger.debug(f"Processed command history ({len(self.processed_commands)}): {list(self.processed_commands)}")
                logger.info(f"New commands to execute: {new_commands_to_execute}")

                # 4. Execute new commands
                if new_commands_to_execute:
                    for cmd in new_commands_to_execute:
                        if not self.running.is_set():
                            break
                        self.execute_command(cmd)
                else:
                    logger.info("No new commands generated by AI for the current state.")

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt detected, stopping loop.")
                break
            except pyautogui.FailSafeException:
                 logger.critical("Fail-safe triggered in main loop!")
                 break
            except Exception as e:
                logger.error(f"Unhandled exception in main loop iteration: {str(e)}", exc_info=True)
                time.sleep(2)

            # Wait for the next interval (unless stopped)
            if self.running.is_set():
                logger.debug(f"Waiting for {self.screenshot_interval} seconds...")
                # Sleep for the interval, but check the event flag periodically
                # to allow quicker shutdown if stop() is called.
                self.running.wait(self.screenshot_interval)


    def stop(self):
        """Sets the event to signal the loop to stop."""
        logger.info("Stopping AI-Automation loop...")
        self.running.clear()

# --- Main Execution ---
if __name__ == '__main__':
    # Configure pyautogui failsafe corner
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFEPOINT = pyautogui.Point(0, 0) # Top-left corner by default

    # Initialize RAG DB
    rag_db = RagDatabase()

    # Optional: Initialize and run Obsidian Indexer
    # Get vault path from environment variables
    obsidian_vault_path = os.getenv('OBSIDIAN_VAULT_PATH')
    if obsidian_vault_path:
        indexer = ObsidianIndexer(obsidian_vault_path, rag_db)
        # Run indexing once at startup if path is set
        indexer.index_notes()
    else:
        logger.info("OBSIDIAN_VAULT_PATH environment variable not set, skipping Obsidian indexing.")


    # Optional: Initialize Learning System (not strictly needed for just running the engine)
    # learning_system = LearningSystem(rag_db)
    # analyzed_patterns = learning_system.analyze_usage_patterns()
    # logger.info(f"Top 5 most successful commands: {analyzed_patterns}")
    # suggested_commands = learning_system.get_suggested_commands()
    # logger.info(f"Suggested commands: {suggested_commands}")


    # Initialize the Automation Engine
    engine = AutomationEngine(rag_db) # Pass the database instance

    # Run the automation loop
    try:
        engine.run_loop()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received during engine startup.")
        engine.stop() # Ensure stop is called if interrupt happens early
    except Exception as e:
        logger.critical(f"Unhandled exception outside of main loop: {str(e)}", exc_info=True)
    finally:
        # Clean up resources
        rag_db.close()
        logger.info("Application finished.")