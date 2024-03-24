import torch
from transformers import BertForMaskedLM, BertConfig
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
import matplotlib.pyplot as plt
import os

spm_model_path = "SP_aze_tokenizer/azerbaijani_spm.model"
sp_model = spm.SentencePieceProcessor(model_file=spm_model_path)

# Define the special token for masking
mask_token = "[MASK]"

mask_token_id = sp_model.piece_to_id(mask_token)

max_length = 128
num_epochs = 10 # 20
batch_size = 8 # 16 or 32
learning_rate = 0.0001


class AzerbaijaniDataset(Dataset):
    def __init__(self, data_file, sp_model, max_length=128):
        self.tokenized_data = []
        with open(data_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip().lower()
                # Tokenize the preprocessed text and truncate/pad tokens to max_length
                tokens = sp_model.encode(line, out_type=int)
                # Apply padding or truncation logic
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                else:
                    tokens += [0] * (max_length - len(tokens))
                self.tokenized_data.append(tokens)

            # print(self.tokenized_data)

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return torch.tensor(self.tokenized_data[idx])


config = BertConfig(
    vocab_size=sp_model.get_piece_size(),
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)


model = BertForMaskedLM(config=config)


data_file = 'main_data/final_data_10000.txt'  # Start with a minimum of around 10,000 sentences
dataset = AzerbaijaniDataset(data_file, sp_model)


train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# Create DataLoaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: pad_sequence(x, batch_first=True))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda x: pad_sequence(x, batch_first=True))


# Define evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            padded_batch = batch.to(device)
            attention_mask = (padded_batch != 0).float().to(device)

            inputs, labels = padded_batch.clone(), padded_batch.clone()
            masked_indices = torch.rand(inputs.shape) < 0.15
            inputs[masked_indices] = mask_token_id

            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss


# Define training function
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device):
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')  # Initialize with a very high value
    # Define the directory for saving checkpoints
    checkpoint_dir = "BertMasked_aze_embedder"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    best_model_path = checkpoint_dir

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            padded_batch = batch.to(device)
            attention_mask = (padded_batch != 0).float().to(device)

            inputs, labels = padded_batch.clone(), padded_batch.clone()
            masked_indices = torch.rand(inputs.shape) < 0.15
            inputs[masked_indices] = mask_token_id

            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        train_loss = total_loss / len(train_dataloader)
        train_losses.append(train_loss)

        val_loss = evaluate_model(model, val_dataloader, device)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")


        # Save the best model
        if val_loss < best_val_loss:
            model.save_pretrained(best_model_path)
            best_val_loss = val_loss

    
    # Load the best model
    best_model = BertForMaskedLM.from_pretrained(best_model_path)
    # best_model.load_state_dict(torch.load(best_model_path))
    return train_losses, val_losses, best_model


# Initialize optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_losses, val_losses, best_model = train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device)

# Plot the training and validation loss curves
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("BertMasked_loss_curve.png")

