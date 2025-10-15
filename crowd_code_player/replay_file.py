import pandas as pd
import curses
import time
import argparse

def offset_to_yx(content, offset):
    """Converts a 1D string offset to 2D (y, x) coordinates."""
    # Ensure offset is within the bounds of the content length
    offset = min(len(content), int(offset))
    
    # Find the line number by counting newlines before the offset
    y = content.count('\n', 0, offset)
    
    # Find the column number by finding the last newline before the offset
    last_newline_pos = content.rfind('\n', 0, offset)
    if last_newline_pos == -1:
        x = offset
    else:
        x = offset - last_newline_pos - 1
        
    return y, x

def apply_change(content, offset, length, new_text):
    """Applies a text change to the content string."""
    content = str(content)
    new_text = str(new_text) if pd.notna(new_text) else ""
    offset, length = int(offset), int(length)
    
    # Convert literal \n and \r characters to actual newlines and carriage returns
    new_text = new_text.replace('\\n', '\n').replace('\\r', '\r')
    
    if offset > len(content):
        content += ' ' * (offset - len(content)) # Pad if offset is out of bounds
        
    return content[:offset] + new_text + content[offset + length:]

def replay_trace(stdscr, filepath, speed_factor, long_pause_threshold=120000):
    """Main function to replay the coding trace in the terminal."""
    # --- Curses Setup ---
    curses.curs_set(0) # We'll draw our own cursor
    stdscr.nodelay(1)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE, -1) # For status bar
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE) # For our cursor

    # --- Data Loading ---
    try:
        df = pd.read_csv(filepath).sort_values('Time').reset_index(drop=True)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return

    # --- State Management ---
    file_states = {}
    scroll_states = {} # Tracks the top-line for each file's viewport
    active_file = None
    paused = False
    
    # --- Main Replay Loop ---
    for i in range(len(df)):
        # --- Handle User Input for Playback Control ---
        key = stdscr.getch()
        if key == ord('q'): break
        if key == ord(' '): paused = not paused
        if key == curses.KEY_UP: speed_factor = min(100, speed_factor * 1.5)
        if key == curses.KEY_DOWN: speed_factor = max(0.1, speed_factor / 1.5)
        
        # Handle Paused State
        if paused:
            height, width = stdscr.getmaxyx()
            stdscr.addstr(height - 1, 0, "PAUSED".ljust(width - 1), curses.A_REVERSE)
            stdscr.refresh()
            while paused:
                time.sleep(0.1)
                key = stdscr.getch()
                if key == ord(' '): paused = False
                elif key == ord('q'): return

        # --- Process Event ---
        event = df.iloc[i]
        active_file = event['File']
        
        # Initialize state for new files
        if active_file not in file_states:
            file_states[active_file] = ""
            scroll_states[active_file] = 0

        
        # Apply content change based on event type
        if active_file == "TERMINAL":
            # For terminal, just append text and add a newline
            terminal_text = str(event['Text']) if pd.notna(event['Text']) else ""
            # Convert literal \n and \r characters to actual newlines and carriage returns
            terminal_text = terminal_text.replace('\\n', '\n').replace('\\r', '\r')
            file_states[active_file] += terminal_text + '\n'
        else:
            file_states[active_file] = apply_change(
                file_states[active_file], event['RangeOffset'], 
                event['RangeLength'], event['Text']
            )
            
        # --- Calculate Cursor and Scrolling ---
        content = file_states[active_file]
        cursor_y, cursor_x = offset_to_yx(content, event['RangeOffset'])
        scroll_y = scroll_states[active_file]
        height, width = stdscr.getmaxyx()
        visible_height = height - 2 # Account for status bars

        # Adjust scroll to keep cursor in view
        if active_file == "TERMINAL":
            # For terminal, always scroll to bottom to show latest content
            lines = content.split('\n')
            total_lines = len(lines)
            if total_lines > visible_height:
                scroll_y = max(0, total_lines - visible_height)
        else:
            # For regular files, keep cursor in view
            if cursor_y < scroll_y:
                scroll_y = cursor_y
            elif cursor_y >= scroll_y + visible_height:
                scroll_y = cursor_y - visible_height + 1
        
        scroll_states[active_file] = scroll_y

        # --- Render to Screen ---
        stdscr.clear()
        
        # Display file content with scrolling
        lines = content.split('\n')
        for j in range(visible_height):
            line_idx = scroll_y + j
            if line_idx < len(lines):
                stdscr.addstr(j, 0, lines[line_idx][:width - 1])
        
        # Draw our custom cursor
        display_y = cursor_y - scroll_y
        if 0 <= display_y < visible_height and 0 <= cursor_x < width:
            # Ensure we don't try to draw on a non-existent character
            line_len = len(lines[cursor_y]) if cursor_y < len(lines) else 0
            char_to_draw_under = lines[cursor_y][cursor_x] if cursor_x < line_len else " "
            stdscr.attron(curses.color_pair(2))
            stdscr.addstr(display_y, cursor_x, char_to_draw_under)
            stdscr.attroff(curses.color_pair(2))

        # Status Bar
        status_bar_text = f"File: {active_file} | Time: {event['Time']/1000:.1f}s | Event: {event['Type']} | Speed: {speed_factor:.1f}x"
        stdscr.attron(curses.color_pair(1) | curses.A_REVERSE)
        stdscr.addstr(height - 2, 0, status_bar_text.ljust(width - 1))
        stdscr.attroff(curses.color_pair(1) | curses.A_REVERSE)

        # Help Text
        help_text = "PAUSE/PLAY [space] | FASTER [↑] | SLOWER [↓] | QUIT [q]"
        stdscr.addstr(height - 1, 0, help_text)

        stdscr.refresh()

        # --- Wait for Next Event ---
        if i + 1 < len(df):
            time_delta_ms = df.iloc[i+1]['Time'] - event['Time']
            sleep_duration_s = max(0, time_delta_ms / 1000.0)
            
            # Check for long pauses
            if time_delta_ms > long_pause_threshold:
                # Display long pause message
                height, width = stdscr.getmaxyx()
                pause_message = "Long pause detected. User might be googling, thinking or might have gone for a coffee..."
                stdscr.addstr(height - 3, 0, pause_message.ljust(width - 1), curses.A_REVERSE)
                stdscr.refresh()
                time.sleep(4)  # Show message for 4 seconds
                stdscr.clear()
            else:
                time.sleep(sleep_duration_s / speed_factor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay coding traces from a CSV file in the terminal.")
    parser.add_argument("filepath", help="The path to the source CSV file.")
    parser.add_argument("--speed", type=float, default=20.0, help="Initial playback speed multiplier.")
    parser.add_argument("--long_pause_threshold", type=int, default=120000, help="Threshold for long pause in milliseconds.")
    args = parser.parse_args()

    curses.wrapper(replay_trace, args.filepath, args.speed, args.long_pause_threshold)