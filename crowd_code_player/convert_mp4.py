import pandas as pd
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- Helper Functions (Preserved) ---

def offset_to_yx(content, offset):
    """Converts a 1D string offset to 2D (y, x) coordinates."""
    offset = min(len(content), int(offset))
    y = content.count('\n', 0, offset)
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
    new_text = new_text.replace('\\n', '\n').replace('\\r', '\r')
    if offset > len(content):
        content += ' ' * (offset - len(content))
    return content[:offset] + new_text + content[offset + length:]

# --- Video Rendering Functions ---

def get_monospaced_font(size=14):
    """Attempts to load a nice monospaced font, falls back to default."""
    # List of common monospaced fonts across OSs
    common_fonts = [
        "Consolas.ttf", "Courier New.ttf", "Lucon.ttf", 
        "LiberationMono-Regular.ttf", "DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Monaco.ttf"
    ]
    
    for font_name in common_fonts:
        try:
            return ImageFont.truetype(font_name, size)
        except IOError:
            continue
            
    print("Warning: Could not find a standard TTF font. Using PIL default (might look pixelated).")
    return ImageFont.load_default()

def create_frame(width, height, content, cursor_pos, scroll_y, active_file, status_text, font, char_w, char_h, pause_message=None):
    """Draws a single video frame using PIL."""
    # Create black background
    img = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Calculate layout
    lines = content.split('\n')
    max_visible_lines = (height // char_h) - 2 # Reserve space for status bar
    
    # Draw File Content
    for i in range(max_visible_lines):
        line_idx = scroll_y + i
        if line_idx < len(lines):
            # Draw text
            draw.text((0, i * char_h), lines[line_idx], font=font, fill=(200, 200, 200))
    
    # Draw Cursor
    cursor_y, cursor_x = cursor_pos
    display_y = cursor_y - scroll_y
    
    if 0 <= display_y < max_visible_lines:
        # Cursor X position in pixels
        cursor_px_x = cursor_x * char_w
        cursor_px_y = display_y * char_h
        
        # Draw a white rectangle for the cursor (simulating block cursor)
        draw.rectangle(
            [cursor_px_x, cursor_px_y, cursor_px_x + char_w, cursor_px_y + char_h], 
            fill=(255, 255, 255)
        )
        
        # Redraw the character under the cursor in black
        line_len = len(lines[cursor_y]) if cursor_y < len(lines) else 0
        char_under = lines[cursor_y][cursor_x] if cursor_x < line_len else ""
        if char_under:
            draw.text((cursor_px_x, cursor_px_y), char_under, font=font, fill=(0, 0, 0))

    # Draw Status Bar Background
    bar_y = height - (2 * char_h)
    draw.rectangle([0, bar_y, width, bar_y + char_h], fill=(255, 255, 255))
    
    # Draw Status Text (Black on White)
    draw.text((0, bar_y), status_text, font=font, fill=(0, 0, 0))
    
    # Draw pause message if provided (above status bar)
    if pause_message:
        pause_bar_y = bar_y - char_h
        draw.rectangle([0, pause_bar_y, width, pause_bar_y + char_h], fill=(255, 165, 0))  # Orange background
        draw.text((0, pause_bar_y), pause_message, font=font, fill=(0, 0, 0))
    
    return np.array(img)

def render_video(filepath, output_file, speed_factor, width=1280, height=720, fps=30, long_pause_threshold=120000):
    """Main loop to process data and write MP4."""
    
    print(f"Processing {filepath}...")
    try:
        df = pd.read_csv(filepath).sort_values('Time').reset_index(drop=True)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return

    # Video Writer Setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Graphics Setup
    font = get_monospaced_font(size=18)
    # measure character size
    left, top, right, bottom = font.getbbox("A")
    char_w = right - left
    char_h = bottom - top + 4 # add padding
    
    # State Management
    file_states = {}
    scroll_states = {}
    active_file = "Unknown"
    max_visible_lines = (height // char_h) - 2
    
    print("Rendering frames... this may take a while.")
    
    for i in range(len(df)):
        event = df.iloc[i]
        next_event = df.iloc[i+1] if i + 1 < len(df) else None
        
        active_file = event['File']
        
        # Initialize state if new
        if active_file not in file_states:
            file_states[active_file] = ""
            scroll_states[active_file] = 0
            
        # Apply Logic
        if active_file == "TERMINAL":
            terminal_text = str(event['Text']) if pd.notna(event['Text']) else ""
            terminal_text = terminal_text.replace('\\n', '\n').replace('\\r', '\r')
            file_states[active_file] += terminal_text + '\n'
        else:
            file_states[active_file] = apply_change(
                file_states[active_file], event['RangeOffset'], 
                event['RangeLength'], event['Text']
            )
            
        # Calculate Scrolling & Cursor
        content = file_states[active_file]
        cursor_y, cursor_x = offset_to_yx(content, event['RangeOffset'])
        scroll_y = scroll_states[active_file]
        
        if active_file == "TERMINAL":
            lines = content.split('\n')
            if len(lines) > max_visible_lines:
                scroll_y = max(0, len(lines) - max_visible_lines)
        else:
            if cursor_y < scroll_y:
                scroll_y = cursor_y
            elif cursor_y >= scroll_y + max_visible_lines:
                scroll_y = cursor_y - max_visible_lines + 1
        
        scroll_states[active_file] = scroll_y
        
        # --- Frame Generation ---
        
        # Prepare Status Text
        status_text = f"File: {active_file} | Time: {event['Time']/1000:.1f}s | Speed: {speed_factor}x"
        
        # --- Time Calculation ---
        # Determine how many frames to repeat this image for
        if next_event is not None:
            real_delta_ms = next_event['Time'] - event['Time']
            
            # Check for long pause
            is_long_pause = real_delta_ms > long_pause_threshold
            
            if is_long_pause:
                # For long pauses, show the pause message for 3 seconds instead of the full duration
                pause_message = "Long pause detected. User might be googling, thinking or might have gone for a coffee..."
                
                # Create frame WITH pause message
                frame_with_pause = create_frame(
                    width, height, content, (cursor_y, cursor_x), 
                    scroll_y, active_file, status_text, font, char_w, char_h, pause_message=pause_message
                )
                
                # Show pause message for 3 seconds
                pause_display_frames = fps * 3
                for _ in range(pause_display_frames):
                    video_out.write(frame_with_pause)
                
                # Continue with a brief normal frame (no pause message)
                frames_to_write = fps  # 1 additional second
            else:
                # Apply speed factor
                video_delta_ms = real_delta_ms / speed_factor
                
                # Convert ms to frame count
                frames_to_write = int((video_delta_ms / 1000.0) * fps)
                
                # Ensure at least 1 frame if there's a gap, but allow 0 for instant events
                if frames_to_write < 1 and video_delta_ms > 10: 
                    frames_to_write = 1
        else:
            # Last event, hold for 2 seconds
            frames_to_write = fps * 2
            is_long_pause = False

        # Create the visual frame (numpy array) - normal frame without pause message
        frame_image = create_frame(
            width, height, content, (cursor_y, cursor_x), 
            scroll_y, active_file, status_text, font, char_w, char_h
        )
        
        # Write the frames
        for _ in range(frames_to_write):
            video_out.write(frame_image)
            
        # Simple progress indicator
        if i % 100 == 0:
            print(f"Processed {i}/{len(df)} events...", end='\r')

    video_out.release()
    print(f"\nDone! Video saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render coding traces to MP4.")
    parser.add_argument("filepath", help="The path to the source CSV file.")
    parser.add_argument("output", help="The output path for the MP4 file.")
    parser.add_argument("--speed", type=float, default=20.0, help="Playback speed multiplier.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--long_pause_threshold", type=int, default=120000, help="Threshold for long pause in milliseconds.")
    
    args = parser.parse_args()
    
    render_video(args.filepath, args.output, args.speed, args.width, args.height, long_pause_threshold=args.long_pause_threshold)
