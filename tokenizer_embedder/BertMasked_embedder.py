import torch
from transformers import BertForMaskedLM, BertConfig
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
import matplotlib.pyplot as plt

spm_model_path = "sp_az_tokenizer/azerbaijani_spm.model"
sp_model = spm.SentencePieceProcessor(model_file=spm_model_path)
# Define the special token for masking
mask_token_id = sp_model.piece_to_id("[MASK]")
max_length = 128


class AzerbaijaniDataset(Dataset):
    def __init__(self, data_file, sp_model, max_length=128):
        self.tokenized_data = []
        with open(data_file, "r", encoding="utf-8") as file:
            for line in file:
                # Tokenize the text and truncate/pad tokens to max_length
                tokens = sp_model.encode(line.strip(), out_type=int)
                # Apply padding or truncation logic
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                else:
                    tokens += [0] * (max_length - len(tokens))
                self.tokenized_data.append(tokens)

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


data_file = "data/clean_dataset.txt"  # Start with a minimum of around 15,000 sentences
dataset = AzerbaijaniDataset(data_file, sp_model)


train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Training loop
num_epochs = 15  # 10-20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_dataloader:
        # Now, after the loop, pad the tokenized sequences
        padded_batch = pad_sequence(batch, batch_first=True).to(device)

        # Create attention mask
        attention_mask = (padded_batch != 0).float().to(device)  # 1 for real tokens, 0 for padding

        # Clone inputs and labels
        inputs, labels = padded_batch.clone(), padded_batch.clone()

        # Apply masking
        masked_indices = torch.rand(inputs.shape) < 0.15
        inputs[masked_indices] = mask_token_id
       
        
        # print(inputs.shape)
        # print(attention_mask.shape)
        # print(labels.shape)
        # print('---')
        
            
        # Pass input IDs to the model
        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)  # Pass input_ids instead of inputs
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    train_loss = total_loss / len(train_dataloader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_total_loss = 0
    with torch.no_grad():
        for val_batch in val_dataloader:
            padded_val_batch = pad_sequence(val_batch, batch_first=True).to(device)
            attention_mask_val = (padded_val_batch != 0).float().to(device)
            inputs_val, labels_val = padded_val_batch.clone(), padded_val_batch.clone()
            masked_indices_val = torch.rand(inputs_val.shape) < 0.15
            inputs_val[masked_indices_val] = mask_token_id
            val_outputs = model(input_ids=inputs_val, attention_mask=attention_mask_val, labels=labels_val)
            val_loss = val_outputs.loss
            val_total_loss += val_loss.item()
    
    val_loss = val_total_loss / len(val_dataloader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")
    model.train()



# Plotting training and validation loss curves
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
# Save the trained model
model.save_pretrained("bert_mlm_az_model")
