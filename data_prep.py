from datasets import load_dataset

def load_and_preprocess_data():
    # Load CNN/DailyMail dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    # Split into train, validation, test
    train_data = dataset['train']
    val_data = dataset['validation']
    test_data = dataset['test']

    return train_data, val_data, test_data

if __name__ == "__main__":
    train, val, test = load_and_preprocess_data()
    print(f"Train size: {len(train)}")
    print(f"Val size: {len(val)}")
    print(f"Test size: {len(test)}")
    print("Sample train data:")
    print(train[0])
