/* Groovy utils functions
Consist of methods that are used outside of stages
*/
// Clone repository with specific branch/tag
def cloneRepository(gitUrl, credentialsId, branch = '*/main'){
    checkout([
        $class: 'GitSCM',
        branches: [[name: branch]],
        userRemoteConfigs: [[
            url: gitUrl,
            credentialsId: credentialsId
        ]]
    ])
}

// Setup Python Environment
def pythonEnvironment(venv) {
    if (!fileExists("requirements.txt")) {
        error "Requirements file does not exist!"
    }
   
    // Create virtual environment
    sh "python3 -m venv ${venv}"
    
    // Install dependencies
    sh """
        source ${venv}/bin/activate
        python3 --version
        python3 -m pip install --upgrade pip
        pip install -r requirements.txt
        echo "✅ Environment setup complete"
    """
}

return this

// Load shared libraries dynamically from workspace
// Ensure that these libraries existing in the jenkins/ folder
// testLib   = load "jenkins/test.groovy"
// buildLib  = load "jenkins/build.groovy"
// deployLib = load "jenkins/deploy.groovy"