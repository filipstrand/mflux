#!/bin/zsh -e
# ^ safe to assume Mac devs have zsh installed
#   default since Catalina in 2019

mkdir -p /tmp/mflux-test

mflux-generate \
    --prompt "Luxury food photograph" \
    --model schnell \
    --steps 2 \
    --seed 2 \
    --height 512 \
    --width 512 \
    --output /tmp/mflux-test/luxury_food.png

# generate an image of a blue bird, then use it as input for the following test
mflux-generate \
    --prompt "blue bird, morning, spring" \
    --model schnell \
    --steps 2 \
    --seed 24 \
    --height 512 \
    --width 512 \
    --stepwise-image-output-dir /tmp/mflux-test \
    --output /tmp/mflux-test/sf_blue_bird.png

# use the image from the prior test, generate an image with similar visual structure
mflux-generate-controlnet \
    --prompt "yellow bird, afternoon, snowy mountain" \
    --model schnell \
    --controlnet-image-path /tmp/mflux-test/sf_blue_bird.png \
    --controlnet-strength 0.7 \
    --controlnet-save-canny \
    --steps 2 \
    --seed 42 \
    --height 512 \
    --width 512 \
    --output /tmp/mflux-test/controlnet_sf_yellow_bird.png \
    --stepwise-image-output-dir /tmp/mflux-test
