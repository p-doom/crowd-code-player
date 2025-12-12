import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import numpy as np
import cv2


class IDMDataset(Dataset):
    """Dataset that loads before/after frames directly from MP4 video."""
    
    def __init__(self, labels_dir, video_path, transform=None, max_seq_len=32, cache_frames=False):
        self.labels_dir = Path(labels_dir)
        self.video_path = Path(video_path)
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.cache_frames = cache_frames
        self.frame_cache = {}
        
        # Load labels
        self.labels = []
        labels_file = self.labels_dir / "keystrokes.jsonl"
        with open(labels_file, 'r') as f:
            for line in f:
                self.labels.append(json.loads(line))
        
        # Load metadata
        with open(self.labels_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Load key vocabulary
        with open(self.labels_dir / "key_vocab.json", 'r') as f:
            self.key_vocab = json.load(f)
        
        self.idx_to_key = {v: k for k, v in self.key_vocab.items()}
        self.vocab_size = len(self.key_vocab)
        
        # Add special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.key_vocab[self.pad_token] = self.vocab_size
        self.key_vocab[self.unk_token] = self.vocab_size + 1
        self.vocab_size += 2
        
        self.pad_idx = self.key_vocab[self.pad_token]
        self.unk_idx = self.key_vocab[self.unk_token]
        
        # Get video properties
        cap = cv2.VideoCapture(str(self.video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Pre-cache frames if requested
        if self.cache_frames:
            self._cache_all_frames()
    
    def _cache_all_frames(self):
        print("Caching video frames...")
        
        # Collect all frame numbers we need
        frame_numbers = set()
        for label in self.labels:
            frame_before = label.get("video_frame_before")
            frame_after = label.get("video_frame_after")
            if frame_before is not None:
                frame_numbers.add(frame_before)
            if frame_after is not None:
                frame_numbers.add(frame_after)
        
        frame_numbers = sorted(frame_numbers)
        
        cap = cv2.VideoCapture(str(self.video_path))
        for frame_num in frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                self.frame_cache[frame_num] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        
        print(f"Cached {len(self.frame_cache)} frames")
    
    def _get_frame(self, frame_number):
        """Get a frame from cache or video."""
        if frame_number in self.frame_cache:
            return self.frame_cache[frame_number].copy()
        
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        # Get frame numbers from label
        frame_before_num = label.get("video_frame_before", 0)
        frame_after_num = label.get("video_frame_after", frame_before_num + 1)
        
        # Load frames from video
        frame_before = self._get_frame(frame_before_num)
        frame_after = self._get_frame(frame_after_num)
        
        if self.transform:
            frame_before = self.transform(frame_before)
            frame_after = self.transform(frame_after)
        
        # Encode keystroke sequence
        keystroke_indices = []
        shift_flags = []
        ctrl_flags = []
        
        for ks in label["keystrokes"]:
            key_idx = self.key_vocab.get(ks["key"], self.unk_idx)
            keystroke_indices.append(key_idx)
            shift_flags.append(1 if ks["shift"] else 0)
            ctrl_flags.append(1 if ks["ctrl"] else 0)
        
        # Pad or truncate
        seq_len = len(keystroke_indices)
        if seq_len > self.max_seq_len:
            keystroke_indices = keystroke_indices[:self.max_seq_len]
            shift_flags = shift_flags[:self.max_seq_len]
            ctrl_flags = ctrl_flags[:self.max_seq_len]
            seq_len = self.max_seq_len
        else:
            pad_len = self.max_seq_len - seq_len
            keystroke_indices.extend([self.pad_idx] * pad_len)
            shift_flags.extend([0] * pad_len)
            ctrl_flags.extend([0] * pad_len)
        
        return {
            "frame_before": frame_before,
            "frame_after": frame_after,
            "keystroke_indices": torch.tensor(keystroke_indices, dtype=torch.long),
            "shift_flags": torch.tensor(shift_flags, dtype=torch.float),
            "ctrl_flags": torch.tensor(ctrl_flags, dtype=torch.float),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
        }
    
    def decode_keystrokes(self, indices):
        return [self.idx_to_key.get(idx.item(), self.unk_token) for idx in indices if idx.item() != self.pad_idx]

if __name__ == "__main__":
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = IDMDataset(
        labels_dir="data/vid_output_labels/",
        video_path="data/vid_output.mp4",
        transform=transform,
        max_seq_len=32,
        cache_frames=True 
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    for batch in dataloader:
        print(batch["frame_before"].shape)  
        print(batch["frame_after"].shape)
        print(batch["keystroke_indices"].shape)  
        break
