# This model is now trained on adjusted and scaled values from the text encoder to values between 0-1
# Values returned from the sketch encoder have to be adjusted by * (8 + 5.1) - 5.1

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import json

from prepare_data import SketchDataset
from sketch_text_contrast.text_encoder import TextEncoder
from sketch_text_contrast.image_encoder import SketchEncoder

num_channels = 192
text_ctx = 128
xf_width = 512
xf_layers = 16
xf_heads = 8
xf_final_ln = True
xf_padding = True

load_weights = True

def main():
    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')
    th.set_default_tensor_type('torch.cuda.FloatTensor')

    print("Setting up data")
    BATCH_SIZE = 100
    EPOCHS = 200
    trainset = SketchDataset("/srv/share/psangkloy3/coco/train2017_contour",
        "/srv/share/psangkloy3/coco/annotations/captions_train2017.json",
        device,
        preloaded_annotations="./train_pairs_noh.json")
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, generator=th.Generator(device=device), drop_last=True)

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

    image_encoder = SketchEncoder(resnet=False, all_trainable=True)
    #image_encoder.load_state_dict(th.load("./sketch_encoder_weights_f100mse_noh.pt"))

    criterion = nn.MSELoss()
    #optimizer = th.optim.SGD(image_encoder.parameters(), lr=1e-1, weight_decay=5e-4, momentum=0.9)
    optimizer = th.optim.Adam(image_encoder.parameters(), lr=2e-3, weight_decay=5e-4)
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.5)

    print("Starting training...")
    loss_values = []
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data
            images = images.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            tokens_list = []
            for label in labels:
                tokens = text_encoder.tokenizer.encode(label)
                tokens, mask = text_encoder.tokenizer.padded_tokens_and_mask(tokens, text_encoder.text_ctx)
                tokens = th.tensor([tokens], device=device)
                mask = th.tensor([mask], dtype=th.bool, device=device)

                text_outputs = text_encoder(tokens, mask)
                xf_out = text_outputs["xf_out"]
                tokens_list.append(xf_out)

            tokens = th.stack(tokens_list)
            # squeeze and scale encodings down to between 0-1
            tokens = (th.squeeze(tokens, dim=1) + 5.1) / (8.0+5.1)

            sketch_outputs = image_encoder(images)

            loss = criterion(sketch_outputs, tokens)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if True: #i % 3 == 2:    # print every 3 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.5f}')
                loss_values.append(running_loss)
                with open('loss_history.json', 'w') as f:
                    json.dump(loss_values, f, indent=2)
                running_loss = 0.0
                th.save(image_encoder.state_dict(), f'./checkpoints/sketch_encoder_weights_f100mse_{epoch + 1}.pt')

        scheduler.step()

    print('Finished Training')

    print('Saving model...')
    th.save(image_encoder.state_dict(), "./sketch_encoder_weights_newvgg_f100mse.pt")

if __name__ == "__main__":
    main()
