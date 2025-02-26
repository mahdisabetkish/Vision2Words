from Flicker_Data import Flicker8k_loader, data
from model import model
import torch
import matplotlib.pyplot as plt

def inference_with_multinomial(model, image, start_token, end_token, max_seq_len=20, temperature=1.3):
    model.eval()  
    device = next(model.parameters()).device 
    image = image.to(device)  
    caption = [start_token]
    with torch.no_grad():  
        features = model.feature_extractor(image)
        features = model.reducer(features)
        features = features.unsqueeze(1)  # Shape: (1, 1, embedd_size)

        h, c = None, None

        for t in range(max_seq_len):
            input_token = torch.tensor([caption[-1]], device=device).unsqueeze(0)  
            input_token = model.embedding_layer(input_token)  

            lstm_input = torch.cat([features, input_token], dim=2)
            output, (h, c) = model.lstm(lstm_input, (h, c) if h is not None else None)
            logits = model.classifier(output.squeeze(1))  # Shape: (1, vocab_size)
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)  
            predicted_token = torch.multinomial(probs, num_samples=1).item()  # Shape: (1,)
            caption.append(predicted_token)
            if predicted_token == end_token:
                break

    return caption

ix, token, image, caption = next(iter(Flicker8k_loader))
caption = inference_with_multinomial(model, image[0].unsqueeze_(0), 0, 1)
generated_caption = [data.i2w[token] for token in caption]
print("Generated Caption:", " ".join(generated_caption))
plt.imshow(image[0].permute(1, 2, 0))
plt.show()
