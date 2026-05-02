FROM lmsysorg/sglang:dev

# Create non-root user with specified UID and GID
# NOTE: Replace with your own UID and GID. This is a workaround from https://github.com/microsoft/vscode-remote-release/issues/49#issuecomment-489060908.
ARG HOST_UID=1003
ARG HOST_GID=1003
RUN groupadd -g $HOST_GID devuser && \
    useradd -m -u $HOST_UID -g $HOST_GID -s /bin/zsh devuser

# Give devuser sudo access
RUN apt-get update && apt-get install -y sudo && \
    echo "devuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/devuser && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Set up oh-my-zsh for devuser
# Original scripts seems have bug
RUN if [ -d /root/.oh-my-zsh ]; then cp -r /root/.oh-my-zsh /home/devuser/.oh-my-zsh; fi && \
    if [ -f /root/.zshrc ]; then cp /root/.zshrc /home/devuser/.zshrc; else echo 'export ZSH="$HOME/.oh-my-zsh"' > /home/devuser/.zshrc; echo 'ZSH_THEME="robbyrussell"' >> /home/devuser/.zshrc; echo 'plugins=(git)' >> /home/devuser/.zshrc; echo 'source $ZSH/oh-my-zsh.sh' >> /home/devuser/.zshrc; fi && \
    if [ -f /root/.vimrc ]; then cp /root/.vimrc /home/devuser/.vimrc; else touch /home/devuser/.vimrc; fi && \
    if [ -f /root/.tmux.conf ]; then cp /root/.tmux.conf /home/devuser/.tmux.conf; else touch /home/devuser/.tmux.conf; fi && \
    sed -i 's|/root/.oh-my-zsh|/home/devuser/.oh-my-zsh|g' /home/devuser/.zshrc || true && \
    chown -R devuser:devuser /home/devuser

# Set workspace directory and ownership
WORKDIR /sgl-workspace/sglang
RUN chown -R devuser:devuser /sgl-workspace

# Switch to devuser
USER devuser

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
