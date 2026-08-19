export STABLEWM_HOME=/workspace
HF_DATASET=galilai-group/lewm-pusht
HF_DATASET_DIR=galilai-group--lewm-pusht

pip install 'stable-worldmodel[train]==0.1.1' transformers==4.57.1 imageio ale-py
pip freeze |grep stable
pip freeze |grep transformers

wandb login --verify
hf auth login

# Start download of dataset in tmux session
HF_DOWNLOAD_SESSION=hf-download
if tmux has-session -t "$HF_DOWNLOAD_SESSION" 2>/dev/null; then
    echo "tmux session '$HF_DOWNLOAD_SESSION' already exists; attach with: tmux attach -t $HF_DOWNLOAD_SESSION"
else
    tmux new-session -d -s "$HF_DOWNLOAD_SESSION" \
        "hf download ${HF_DATASET} --repo-type dataset --local-dir ${STABLEWM_HOME}/datasets/${HF_DATASET_DIR}/; echo; echo 'Download finished. Press Enter to close.'; read"
    echo "Started Hugging Face download in tmux session '$HF_DOWNLOAD_SESSION'."
    echo "Attach with: tmux attach -t $HF_DOWNLOAD_SESSION"
fi
# Attach with 
# tmux attach -t hf-download