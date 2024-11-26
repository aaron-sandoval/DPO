
# Step 3: Install pyenv
if [ ! -d "$HOME/.pyenv" ] && ! command -v pyenv &>/dev/null; then
    echo "Installing pyenv dependencies..."
    # sudo apt update; sudo apt install build-essential libssl-dev \
    # libbz2-dev libreadline-dev libsqlite3-dev \
    # libncursesw5-dev tk-dev libxml2-dev libxmlsec1-dev
    sudo apt update; sudo apt-get install -y build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
    echo "Installing pyenv..."
    curl https://pyenv.run | bash

    # Add pyenv to PATH and initialize it
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$HOME/.pyenv/shims:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"

    # Persist to shell configuration
    echo -e '\n# Pyenv Configuration' >> ~/.bashrc
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$HOME/.pyenv/shims:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    # Ensure python and python3 point to pyenv shim
    export PATH="$HOME/.pyenv/shims:$PATH"
    echo "pyenv installed and added to shell configuration."
else
    echo "pyenv is already installed."
fi


# Step 4: Determine Python version
PYTHON_VERSION=""
if [ -f ".python-version" ]; then
    PYTHON_VERSION=$(cat .python-version)
    echo "Using Python version from .python-version file: $PYTHON_VERSION"
else
    if [ -f "pyproject.toml" ]; then
        echo "Reading Python version from pyproject.toml..."
        PYTHON_VERSION=$(grep -E "python\s*=" pyproject.toml | sed -E 's/.*=\s*"([^"]+)".*/\1/')
        if [ -z "$PYTHON_VERSION" ]; then
            echo "No Python version constraint found in pyproject.toml."
            exit 1
        fi
        echo "Python version constraint from pyproject.toml: $PYTHON_VERSION"

        # Parse version constraint and find the earliest allowable version
        PYTHON_VERSION=$(python3 -c "
import packaging.specifiers, packaging.version
constraint = '$PYTHON_VERSION'
versions = [packaging.version.Version(f'3.{i}.0') for i in range(30)]
matching_versions = sorted(v for v in versions if v in packaging.specifiers.SpecifierSet(constraint))
print(matching_versions[0]) if matching_versions else exit(1)")
        echo "Earliest matching Python version: $PYTHON_VERSION"
    else
        echo "No .python-version or pyproject.toml file found."
        exit 1
    fi
fi
# exit 1
# Step 5: Install Python using pyenv
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo "Installing Python $PYTHON_VERSION..."
    pyenv install "$PYTHON_VERSION"
else
    echo "Python $PYTHON_VERSION is already installed."
fi
echo "Python executable: "$(which python3)
echo "Pip executable: "$(which pip)
# exit 1
# Step 6: Set local Python version if .python-version does not exist
if [ ! -f ".python-version" ]; then
    pyenv local "$PYTHON_VERSION"
    echo "Set local Python version to $PYTHON_VERSION."
fi

# Step 7: Install Poetry
if ! command -v poetry &>/dev/null; then
    echo "Installing Poetry..."
    pip install --user poetry
    export PATH="$HOME/.local/bin:$PATH"
    echo "Poetry installed."
else
    echo "Poetry is already installed."
fi

# Step 8: Install project dependencies using Poetry
echo "Installing project dependencies using Poetry..."
poetry install

echo "Python environment setup complete."

git config --global user.email "nope@nope.com"
git config --global user.name "Nope"
echo "Git configuration complete."