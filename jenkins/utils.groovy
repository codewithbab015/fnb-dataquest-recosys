/* 
   Groovy utility functions used across pipeline stages.
*/

// Clone the Git repository using provided credentials and branch (default: main)
def cloneRepository(gitUrl, credentialsId, branch = '*/main') {
    checkout([
        $class: 'GitSCM',
        branches: [[name: branch]],
        userRemoteConfigs: [[
            url: gitUrl,
            credentialsId: credentialsId
        ]]
    ])
}

// Set up Python virtual environment and install dependencies using pip cache
def pythonEnvironment(venv) {
    if (!fileExists("requirements.txt")) {
        error "The requirements.txt file is missing. Unable to set up Python environment."
    }

    sh """
        echo "Creating Python virtual environment at ${venv}"
        python3 -m venv ${venv}

        echo "Activating virtual environment"
        source ${venv}/bin/activate

        echo "Setting PIP cache directory to /cache"
        export PIP_CACHE_DIR=/cache

        echo "Upgrading pip"
        python3 -m pip install --upgrade pip

        echo "Installing Python dependencies from requirements.txt"
        pip install --cache-dir=\$PIP_CACHE_DIR -r requirements.txt

        echo "Python environment setup complete"
    """
}

return this


// Load shared libraries dynamically from workspace
// Ensure that these libraries existing in the jenkins/ folder
// testLib   = load "jenkins/test.groovy"
// buildLib  = load "jenkins/build.groovy"
// deployLib = load "jenkins/deploy.groovy"