#!/bin/bash

set -e

VM_USERNAME="himanshu.yadav"
VM_IP="104.198.201.185"
SSH_KEY_PATH="~/.ssh/id_ed25519"

# --- Main Script Logic ---
echo "🔌 Connecting to VM (${VM_IP})..."
ssh -i "${SSH_KEY_PATH}" "${VM_USERNAME}@${VM_IP}"
