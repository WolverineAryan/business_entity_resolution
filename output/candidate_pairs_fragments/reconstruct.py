"""Reconstruct output/candidate_pairs.tsv from fragments."""
import os
import glob

def reconstruct():
    frag_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(os.path.dirname(frag_dir), "candidate_pairs.tsv")
    parts = sorted(glob.glob(os.path.join(frag_dir, "candidate_pairs.tsv.part_*")))
    print(f"Reconstructing {target_path} from {len(parts)} parts...")
    with open(target_path, "wb") as f_out:
        for p in parts:
            with open(p, "rb") as f_in:
                f_out.write(f_in.read())
    print(f"Successfully reconstructed {target_path} ({os.path.getsize(target_path):,} bytes)!")

if __name__ == "__main__":
    reconstruct()
