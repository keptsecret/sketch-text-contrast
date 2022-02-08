import torch as th

from sketch_text_contrast.text_encoder import TextEncoder
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
    
    model = TextEncoder(
        model_channels=num_channels,
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        xf_padding=xf_padding
    )

    if load_weights:
        model.load_state_dict(th.load("./transformer_only_weights.pt"))
        # model.load_state_dict(load_checkpoint("transformer", device))

    print(model)
    """
    # Sampling parameters
    prompt = "an oil painting of a corgi"
    batch_size = 1
    guidance_scale = 3.0

    # Create the text tokens to feed to the model.
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(
        tokens, text_ctx
    )

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], text_ctx
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=th.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )
    """

if __name__ == "__main__":
    main()
