
## 1) Create .vscode/settings.json
'''
In VS Code Explorer, right-click your project root → New Folder → name it .vscode.
Inside .vscode, create settings.json and paste the below.
'''

// project_root/.vscode/settings.json
{
  // Add "src" to VSCode’s import resolution and IntelliSense
  "python.analysis.extraPaths": [
    "${workspaceFolder}/src"
  ],

  // (Optional) If you’re using a virtual-env inside your project:
  // "python.pythonPath": "${workspaceFolder}/.venv/bin/python"
}

########################################
## 2) Create .vscode/launch.json
'''
In the same .vscode folder, create launch.json.
Paste the above JSON.
'''


// project_root/.vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "▶ Run prearb.consumer",
      "type": "python",
      "request": "launch",

      // Run as a module: python -m prearb.consumer
      "module": "prearb.consumer",

      // Make sure the working dir is "src" so `prearb/` is visible
      "cwd": "${workspaceFolder}/src",

      // (Optional) Load environment variables from .env if you have one
      "envFile": "${workspaceFolder}/.env"
    }
  ]
}



