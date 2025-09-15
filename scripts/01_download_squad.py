from datasets import load_dataset
if __name__ == "__main__":
    ds = load_dataset("squad")
    ds.save_to_disk("./data/squad")
    print("Saved to ./data/squad")