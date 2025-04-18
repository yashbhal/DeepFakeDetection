#!/bin/bash

# Install required packages if not already installed
pip install datasets tqdm Pillow --quiet

# Backup existing real images
echo "Backing up existing real images..."
mv dataset/train/real dataset/train/real_backup

# Run the download script
python dataset/download_real_images.py

# Compare counts
echo "New real images count:"
ls -l dataset/train/real/ | wc -l
echo "Old real images count:"
ls -l dataset/train/real_backup/ | wc -l

# Ask user if they want to keep the new images or restore backup
read -p "Do you want to keep the new images? (y/n) " answer
if [ "$answer" != "y" ]; then
    rm -rf dataset/train/real
    mv dataset/train/real_backup dataset/train/real
    echo "Restored original images"
else
    rm -rf dataset/train/real_backup
    echo "Kept new images"
fi 