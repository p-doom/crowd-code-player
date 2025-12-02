
from huggingface_hub import HfApi, hf_hub_download
import argparse
from crowd_code_player.convert_mp4 import render_video

parser = argparse.ArgumentParser(description="Render coding traces to MP4.")
parser.add_argument("--speed", type=float, default=20.0, help="Playback speed multiplier.")
parser.add_argument("--width", type=int, default=1280, help="Video width.")
parser.add_argument("--height", type=int, default=720, help="Video height.")
parser.add_argument("--num_videos", type=int, default=1, help="Number of videos to generate from the dataset.")

args = parser.parse_args()

api = HfApi()

repo_id = "p-doom/crowd-code-0.1"
repo_type = "dataset"

print("Fetching file list from repository...")
files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

csv_files = [f for f in files if f.endswith('.csv')]

print(f"Found {len(csv_files)} CSV files\n")

for idx, csv_file in enumerate(csv_files, 1):
    print(f"Processing {idx}/{len(csv_files)}: {csv_file}")
    if idx > args.num_videos:
        break
    
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=csv_file,
            repo_type=repo_type
        )

        output_filename = f"{args.output}_{csv_file.replace('/', '_').replace('.csv', '')}.mp4"
        render_video(file_path, output_filename, args.speed, args.width, args.height)
        
    except Exception as e:
        print(f"  Error processing {csv_file}: {str(e)}")
        print("-" * 80)
        continue

print(f"\nCompleted processing {len(csv_files)} CSV files")
