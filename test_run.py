import torch as th
import torch.nn as nn

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

    prompt = "a cat with a bat"
    batch_size = 1

    tokens = text_encoder.tokenizer.encode(prompt)
    tokens, mask = text_encoder.tokenizer.padded_tokens_and_mask(tokens, text_encoder.text_ctx)
    # uncond_tokens, uncond_mask = text_encoder.tokenizer.padded_tokens_and_mask([], text_ctx)

    # Should be safe to ignore classifier-free tokens in this instance
    # tokens = th.tensor([tokens] * batch_size + [uncond_tokens] * batch_size, device=device)
    # mask = th.tensor([mask] * batch_size + [uncond_mask] * batch_size, dtype=th.bool, device=device)
    tokens = th.tensor([tokens] * batch_size, device=device)
    mask = th.tensor([mask] * batch_size, dtype=th.bool, device=device)

    text_outputs = text_encoder(tokens, mask)
    xf_proj, xf_out = text_outputs["xf_proj"], text_outputs["xf_out"]

    # image_encoder = ImageEncoder(xf_width=xf_width, text_ctx=text_ctx)
    image_encoder = SketchEncoder()
    img_out = image_encoder(th.normal(0, 1, size=(batch_size, 3, 224, 224)))

    criterion = nn.CosineEmbeddingLoss(margin=0)
    loss = criterion(xf_out[0], img_out[0], th.tensor([1]))
    print(loss)

if __name__ == "__main__":
    main()
