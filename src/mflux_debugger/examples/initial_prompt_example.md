I would like to do the following:

Begin by reading the readme in mflux_debugger/examples

it should tell you how to use the debugger

I would like you to step through the main chunks of the model:

- initial latent
- text encoder
- transformer
- vae

for both diffusers and mlx, in that order.

The main idea is to see that given similar inputs that they have similar tensor values at different moments in time. This would ensure a correct port from one to the other.

Now, for this to work, we would have to have the same input latents. And since that is randomly generated, even with fixed seed, the two implementations don't agree. So we would have to save the one from diffusers to disk and read that in mlx. Otherwise, we don't have to do any other interventions.

To keep things simple, I would actually just do that very quickly so that this thing is saved to disk with minimal intervention in code. You can actually do it in line in diffusers. And once it's saved, we can comment that line out again. We don't even need to finish the whole script, we can break the process after the save (the first time we run it).

And then in the MLX implementation, we can actually have a line where we load in this in place. It's totally fine to edit the source files here. I think that makes it easier.

For debugging, you should do it here in the chat by calling various endpoints, not by writing helper scripts!

