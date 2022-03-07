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

# def train_step(text_encoder, image_encoder, trainloader, criterion, optimizer, epoch):
#     running_loss = 0.0
#     for (image, label), corr in trainloader:

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         text_encoding = text_encoder(label)
#         sketch_encoding = image_encoder(image)
#         loss = criterion(sketch_encoding, text_encoding, corr)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0

def main():
    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')
    th.set_default_tensor_type('torch.cuda.FloatTensor')

    print("Setting up data")
    BATCH_SIZE = 64
    EPOCHS = 15
    pos_trainset = SketchDataset("/srv/share/psangkloy3/coco/train2017_contour",
        "/srv/share/psangkloy3/coco/annotations/captions_train2017.json",
        device,
        preloaded_annotations="./train_pairs.json")
    pos_trainloader = DataLoader(pos_trainset, batch_size=BATCH_SIZE)

    neg_trainset = SketchDataset("/srv/share/psangkloy3/coco/train2017_contour",
        "/srv/share/psangkloy3/coco/annotations/captions_train2017.json",
        device,
        negatives=True,
        preloaded_annotations="./train_pairs.json")
    neg_trainloader = DataLoader(neg_trainset, batch_size=BATCH_SIZE)

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

    #criterion = nn.CosineEmbeddingLoss(margin=0.5)
    # criterion = nn.TripletMarginLoss()
    dist = nn.PairwiseDistance(p=1, eps=1e-6)
    criterion = nn.HingeEmbeddingLoss(margin=1.0)
    fast_optimizer = th.optim.Adam(image_encoder.parameters(), lr=1e-3, weight_decay=1e-5)
    slow_optimizer = th.optim.Adam(image_encoder.parameters(), lr=1e-4, weight_decay=1e-5)       # (1e-4, 1e-5) seemed okay but slow

    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        if EPOCHS < 8:
            optimizer = fast_optimizer
        else:
            optimizer = slow_optimizer

        running_loss = 0.0
        for i, data in enumerate(pos_trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data

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

            outputs = dist(th.flatten(sketch_outputs, start_dim=1), th.flatten(tokens, start_dim=1))
            #loss = criterion(th.flatten(sketch_outputs, start_dim=1), th.flatten(tokens, start_dim=1), th.tensor([1]))
            loss = criterion(outputs, th.ones_like(outputs))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 100 mini-batches
                print(f'pos:[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.5f}')
                running_loss = 0.0

        running_loss = 0.0
        for i, data in enumerate(neg_trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data

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

            outputs = dist(th.flatten(sketch_outputs, start_dim=1), th.flatten(tokens, start_dim=1))
            #loss = criterion(th.flatten(sketch_outputs, start_dim=1), th.flatten(tokens, start_dim=1), th.tensor([-1]))
            loss = criterion(outputs, th.ones_like(outputs) * -1)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 100 mini-batches
                print(f'neg:[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.5f}')
                running_loss = 0.0

        th.save(image_encoder.state_dict(), f'./checkpoints/sketch_encoder_weights_{epoch + 1}.pt')

    print('Finished Training')

    print('Saving model...')
    th.save(image_encoder.state_dict(), "./sketch_encoder_weights.pt")

if __name__ == "__main__":
    main()
