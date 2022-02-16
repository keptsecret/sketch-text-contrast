import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader

from prepare_data import SketchDataset
from sketch_text_contrast.text_encoder import TextEncoder
from sketch_text_contrast.image_encoder import ImageEncoder, SketchEncoder
from sketch_text_contrast.download import load_checkpoint

num_channels = 192
text_ctx = 128
xf_width = 512
xf_layers = 16
xf_heads = 8
xf_final_ln = True
xf_padding = True

load_weights = True

"""
Tentative, not sure how data works yet
"""
def train_step(text_encoder, image_encoder, trainloader, criterion, optimizer, epoch):
    running_loss = 0.0
    for (image, label), corr in trainloader:
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        text_encoding = text_encoder(label)
        sketch_encoding = image_encoder(image)
        loss = criterion(sketch_encoding, text_encoding, corr)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

def main():
    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')
    th.set_default_tensor_type('torch.cuda.FloatTensor')

    print("Setting up data")
    batch_size = 1
    trainset = SketchDataset("./test_dir", "captions_val2017.json", device)
    trainloader = DataLoader(trainset, batch_size=batch_size)

    print("Setting up models")    
    text_encoder = TextEncoder(
        model_channels=num_channels,
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        xf_padding=xf_padding,
        device=device
    )

    if load_weights:
        text_encoder.load_state_dict(th.load("./transformer_only_weights.pt"))
        # model.load_state_dict(load_checkpoint("transformer", device))

    # prompt = "a cat with a bat"

    # tokens = text_encoder.tokenizer.encode(prompt)
    # tokens, mask = text_encoder.tokenizer.padded_tokens_and_mask(tokens, text_encoder.text_ctx)
    # # uncond_tokens, uncond_mask = text_encoder.tokenizer.padded_tokens_and_mask([], text_ctx)

    # # Should be safe to ignore classifier-free tokens in this instance
    # # tokens = th.tensor([tokens] * batch_size + [uncond_tokens] * batch_size, device=device)
    # # mask = th.tensor([mask] * batch_size + [uncond_mask] * batch_size, dtype=th.bool, device=device)
    # tokens = th.tensor([tokens] * batch_size, device=device)
    # mask = th.tensor([mask] * batch_size, dtype=th.bool, device=device)

    # text_outputs = text_encoder(tokens, mask)
    # xf_proj, xf_out = text_outputs["xf_proj"], text_outputs["xf_out"]

    # # image_encoder = ImageEncoder(xf_width=xf_width, text_ctx=text_ctx)
    image_encoder = SketchEncoder()
    # img_out = image_encoder(th.normal(0, 1, size=(batch_size, 3, 224, 224)))

    criterion = nn.CosineEmbeddingLoss(margin=0)
    # # criterion = nn.TripletMarginLoss()
    optimizer = th.optim.Adam(image_encoder.parameters(), lr=1e-4, weight_decay=1e-5)
    # loss = criterion(xf_out[0], img_out[0], th.tensor([1]))
    # loss.backward()
    # print(loss)

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            tokens_list = []
            mask_list = []
            for label in labels:
                tokens = text_encoder.tokenizer.encode(label)
                tokens, mask = text_encoder.tokenizer.padded_tokens_and_mask(tokens, text_encoder.text_ctx)
                tokens = th.tensor([tokens] * batch_size, device=device)
                mask = th.tensor([mask] * batch_size, dtype=th.bool, device=device)

                text_outputs = text_encoder(tokens, mask)
                xf_out = text_outputs["xf_out"]
                tokens_list.append(xf_out)
                # mask_list.append(mask)

            tokens = th.stack(tokens_list)
            # mask = th.stack(mask_list)

            # forward + backward + optimize
            sketch_outputs = image_encoder(images)
            
            loss = criterion(th.flatten(sketch_outputs, start_dim=1), th.flatten(tokens, start_dim=1), th.Tensor([1]))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')
            running_loss = 0.0

    print('Finished Training')

if __name__ == "__main__":
    main()
