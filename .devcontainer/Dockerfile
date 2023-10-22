ARG VARIANT="3.8-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# Update pip
RUN pip install --upgrade pip

#Install poetry official way
USER vscode
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH = "${PATH}:/home/vscode/.local/bin"

## Git setup
RUN git config --global diff.tool vscode && git config --global difftool.vscode.cmd 'code --wait --diff $LOCAL $REMOTE'