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

def main():
    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')
    th.set_default_tensor_type('torch.cuda.FloatTensor')

    print("Setting up data")
    BATCH_SIZE = 32
    EPOCHS = 20
    pos_trainset = SketchDataset("/srv/share/psangkloy3/coco/train2017_contour",
        "/srv/share/psangkloy3/coco/annotations/captions_train2017.json",
        device,
        negatives=False,
        preloaded_annotations="./train_pairs.json")
    pos_trainloader = DataLoader(pos_trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        #num_workers=2,
        generator=th.Generator(device=device))

    neg_trainset = SketchDataset("/srv/share/psangkloy3/coco/train2017_contour",
        "/srv/share/psangkloy3/coco/annotations/captions_train2017.json",
        device,
        negatives=True,
        preloaded_annotations="./train_pairs.json")
    neg_trainloader = DataLoader(neg_trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        #num_workers=2,
        generator=th.Generator(device=device))

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
    image_encoder.load_state_dict(th.load("./checkpoints/sketch_encoder_weights_2.pt"))
    for params in image_encoder.vgg.parameters():
        params.requires_grad = True

    #dist = nn.PairwiseDistance(p=1)
    dist = nn.CosineSimilarity(dim=0, eps=1e-06)
    criterion = nn.HingeEmbeddingLoss(margin=1.01)
    # criterion = nn.TripletMarginLoss()
    fast_optimizer = th.optim.Adam(image_encoder.parameters(), lr=1e-3, weight_decay=1e-5)
    slow_optimizer = th.optim.Adam(image_encoder.parameters(), lr=1e-4, weight_decay=1e-5)       # (1e-4, 1e-5) seemed okay but slow

    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        if epoch < 8:
            optimizer = fast_optimizer
        else:
            optimizer = slow_optimizer

        # train positive matches
        running_loss = 0.0
        for i, data in enumerate(pos_trainloader, 0):
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

            sketch_outputs = image_encoder(images)

            outputs = 1.0 - dist(sketch_outputs, tokens)
            #loss = criterion(th.flatten(sketch_outputs, start_dim=1), th.flatten(tokens, start_dim=1), th.tensor([1]))
            loss = criterion(outputs, th.ones_like(outputs))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 400 == 399:    # print every 400 mini-batches
                print(f'pos-[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 400:.5f}')
                running_loss = 0.0

            th.save(image_encoder.state_dict(), f'./checkpoints/sketch_encoder_weights_tuned_{epoch + 1}.pt')

        # train negative matches
        running_loss = 0.0
        for i, data in enumerate(neg_trainloader, 0):
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

            sketch_outputs = image_encoder(images)

            #outputs = dist(th.flatten(sketch_outputs, start_dim=1), th.flatten(tokens, start_dim=1))
            outputs = 1.0 - dist(sketch_outputs, tokens)
            #loss = criterion(th.flatten(sketch_outputs, start_dim=1), th.flatten(tokens, start_dim=1), th.tensor([1]))
            loss = criterion(outputs, th.ones_like(outputs) * -1.0)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 400 == 399:    # print every 400 mini-batches
                print(f'neg-[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 400:.5f}')
                running_loss = 0.0

            th.save(image_encoder.state_dict(), f'./checkpoints/sketch_encoder_weights_tuned_{epoch + 1}.pt')

    print('Finished Training')

    print('Saving model...')
    th.save(image_encoder.state_dict(), "./sketch_encoder_weights_tuned.pt")

if __name__ == "__main__":
    main()
