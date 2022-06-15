import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import json

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

def main():
    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')
    th.set_default_tensor_type('torch.cuda.FloatTensor')

    print("Setting up data")
    BATCH_SIZE = 32
    EPOCHS = 2000
    trainset = SketchDataset("/srv/share/psangkloy3/coco/train2017_contour",
        "/srv/share/psangkloy3/coco/annotations/captions_train2017.json",
        device,
        triplet=True,
        preloaded_annotations="./train_pairs_noh.json")
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, generator=th.Generator(device=device), drop_last=True)
    #trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, drop_last=True)

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

    image_encoder = SketchEncoder()
    image_encoder.load_state_dict(th.load("./sketch_encoder_weights_f100_noh.pt"))
    for params in image_encoder.vgg.parameters():
        params.requires_grad = True

    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = th.optim.SGD(image_encoder.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9)       # (1e-4, 1e-5) seemed okay but slow
    #optimizer = th.optim.Adam(image_encoder.parameters(), lr=1e-3)
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    print("Starting training...")
    loss_values = []
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, pos_labels, neg_labels = data
            images = images.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            pos_tokens_list = []
            for label in pos_labels:
                tokens = text_encoder.tokenizer.encode(label)
                tokens, mask = text_encoder.tokenizer.padded_tokens_and_mask(tokens, text_encoder.text_ctx)
                tokens = th.tensor([tokens], device=device)
                mask = th.tensor([mask], dtype=th.bool, device=device)

                text_outputs = text_encoder(tokens, mask)
                xf_out = text_outputs["xf_out"]
                pos_tokens_list.append(xf_out)

            pos_tokens = th.stack(pos_tokens_list)
            pos_tokens = th.squeeze(pos_tokens, dim=1)

            neg_tokens_list = []
            for label in neg_labels:
                tokens = text_encoder.tokenizer.encode(label)
                tokens, mask = text_encoder.tokenizer.padded_tokens_and_mask(tokens, text_encoder.text_ctx)
                tokens = th.tensor([tokens], device=device)
                mask = th.tensor([mask], dtype=th.bool, device=device)

                text_outputs = text_encoder(tokens, mask)
                xf_out = text_outputs["xf_out"]
                neg_tokens_list.append(xf_out)

            neg_tokens = th.stack(neg_tokens_list)
            neg_tokens = th.squeeze(neg_tokens, dim=1)

            sketch_outputs = image_encoder(images)

            #outputs = dist(th.flatten(sketch_outputs, start_dim=1), th.flatten(tokens, start_dim=1))
            #loss = criterion(th.flatten(sketch_outputs, start_dim=1), th.flatten(tokens, start_dim=1), th.tensor([1]))
            loss = criterion(sketch_outputs, pos_tokens, neg_tokens)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 3 == 2:    # print every 200 mini-batches
                # not noh finished at 10452
                print(f'[{epoch + 9001}, {i + 1:5d}] loss: {running_loss / 3:.5f}')
                loss_values.append(running_loss / 3)
                with open('loss_history.json', 'w') as f:
                    json.dump(loss_values, f, indent=2)
                running_loss = 0.0
                th.save(image_encoder.state_dict(), f'./checkpoints_noh/sketch_encoder_weights_f100_noh_{epoch + 9001}.pt')
            #else:
            #    th.save(image_encoder.state_dict(), f'./checkpoints/sketch_encoder_weights_f100_{epoch + 1}.pt')

        scheduler.step()

    print('Finished Training')

    print('Saving model...')
    th.save(image_encoder.state_dict(), "./sketch_encoder_weights_f100_noh.pt")

if __name__ == "__main__":
    main()
