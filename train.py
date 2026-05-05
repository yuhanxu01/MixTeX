from transformers import AutoTokenizer, VisionEncoderDecoderModel, AutoImageProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from torch.utils.data import Dataset
import torch

# 3. Load Model, Tokenizer, and Image Processor (Updated to MixTex/base_ZhEn)
MODEL_NAME = "MixTex/base_ZhEn"

feature_extractor = AutoImageProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, max_len=296)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# 4. Load Dataset (Kept as original)
dataframe = load_dataset("MixTex/Pseudo-Latex-ZhEn-1")

# 5. Define Custom Dataset Class
class MixTexDataset(Dataset):
    def __init__(self, dataframe, tokenizer, feature_extractor, max_length=256):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.dataframe['train'])

    def __getitem__(self, idx):
        # Get image and convert to RGB
        image = self.dataframe['train'][idx]['image'].convert("RGB")
        # Get target LaTeX text
        target_text = self.dataframe['train'][idx]['text']
        
        # Process image
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
        
        # Tokenize text and create labels
        target = self.tokenizer(target_text, padding="max_length", max_length=self.max_length, truncation=True).input_ids
        # Replace pad token ids with -100 to ignore them in loss calculation
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in target]
        
        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}

# Instantiate the dataset
traindataset = MixTexDataset(dataframe, tokenizer, feature_extractor=feature_extractor)

# 6. Define Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=12,
    predict_with_generate=True,
    logging_dir='./logs',
    learning_rate=5e-5,
    save_total_limit=1,
    logging_steps=100,
    save_steps=500,
    num_train_epochs=3,
    fp16=True, # Make sure you are using a GPU that supports fp16
)

# 7. Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=traindataset,
)

# 8. Start Training
trainer.train()
